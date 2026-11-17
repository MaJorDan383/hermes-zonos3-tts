"""ZONOS3-Mini TTS for Hermes — native Zyphra ggml server, lease-driven, idle-unloading.

Engine choice, measured on this host (RTX 4090, 2026-11-15):

====================  ======================  =====================  ==============
                      Zyphra `servers/ggml`   `audio.cpp` port       winner
====================  ======================  =====================  ==============
silent-output rate    0/12                    4/10 (needs retry)     Zyphra
backbone throughput   376 frames/s            ~120 frames/s          Zyphra
emotion steering      yes (steering bundle)   not implemented        Zyphra
named voices          `/v2/speakers`          request_options        Zyphra
model unload          none (process only)     `--idle-unload-ms`     audio.cpp
====================  ======================  =====================  ==============

Zyphra wins on correctness and capability. The one thing it lacks — an idle
unload — is exactly what this plugin supplies: it owns the server as a child
process and stops it after the idle window. That reclaims *all* the VRAM (the
0.58 GiB of weights plus KV cache, decoder, and CUDA context), which is strictly
more than audio.cpp's model-only unload.

Emotion: SenseVoice reports an emotion per utterance; ZONOS3 turns it into
prosody through its *steering bundle* (`emotions.*`, `mood.*`, `delivery.*`
directions served by `GET /v2/controls`). This plugin maps SenseVoice's labels
onto those directions and sends them as request `controls`. Verified to change
the audio: `emotions.happy` raised zero-crossing rate ~20% and RMS ~12% vs
baseline; `emotions.sad` lowered them ~15%/~21%.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import array
from pathlib import Path
from typing import Any, Optional

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

# Sibling modules. Loaded defensively: a broken emotion half must never stop the
# TTS provider from registering — losing the voice would be far worse than losing
# the affect.
try:  # pragma: no cover - import plumbing
    from . import emotion as emotion_mod
except Exception:  # noqa: BLE001
    try:
        import emotion as emotion_mod          # type: ignore[no-redef]
    except Exception:  # noqa: BLE001
        emotion_mod = None                     # type: ignore[assignment]

try:  # pragma: no cover - import plumbing
    from .sensevoice import SenseVoiceEmotion
    from . import sensevoice as sensevoice_mod
except Exception:  # noqa: BLE001
    try:
        from sensevoice import SenseVoiceEmotion  # type: ignore[no-redef]
        import sensevoice as sensevoice_mod      # type: ignore[no-redef]
    except Exception:  # noqa: BLE001
        SenseVoiceEmotion = None               # type: ignore[assignment]
        sensevoice_mod = None                  # type: ignore[assignment]

try:  # pragma: no cover - import plumbing
    from .ducking import AudioDucker
except Exception:  # noqa: BLE001
    try:
        from ducking import AudioDucker  # type: ignore[no-redef]
    except Exception:  # noqa: BLE001
        AudioDucker = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- defaults

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 1919

# Measured: the server binds in ~2 s (weights load before the listener opens), so
# stopping the process on an idle window costs little and returns every byte of
# VRAM. 5 minutes keeps a conversation warm across pauses without holding the GPU
# through a long away period.
_DEFAULT_UNLOAD_AFTER = 300.0
_DEFAULT_START_TIMEOUT = 120.0
_DEFAULT_REQUEST_TIMEOUT = 300.0
_DEFAULT_RETRY_ATTEMPTS = 3

# -- prosody defaults -------------------------------------------------------
# Always-on shaping, applied to every utterance regardless of detected emotion.
# Measured on this host against a baseline F0 spread of 3.17 semitones:
#   pitch.pitch_range +2 -> 5.93 st (near double: the strongest single lever)
#   expressive.expressive +16 -> 4.25 st
#   delivery.loud 4 -> +4.7 dB peak / +6.8 dB RMS (the engine's only real gain;
#   mood.loudness measured -2.2 dB peak and is deliberately unused)
# Defaults stay modest — audible lift without pushing a voice off its identity.
_DEFAULT_DYNAMIC_RANGE = 1.5     # pitch.pitch_range, safe band ±2
_DEFAULT_EXPRESSIVE = 4.0        # expressive.expressive, safe band ±8
_DEFAULT_VOLUME_BOOST = 2.0      # delivery.loud, safe band 0-4

# Steer-then-hold. A constant hidden-space push integrates over a long line: the
# model copies the style of speech it has already produced, so stacked liveliness
# + valence + arousal climb in pitch and loudness towards shouting. Zyphra measured
# a stacked voice design climbing in 19 of 24 takes with the push held constant,
# and chose onset 4 s / ramp 1 s / hold 0.35 as the largest hold that stayed
# neutral. Their numbers: hold 1.0 = +7.2 st and +7.0 dB after the first 5 s;
# hold 0.35 = +0.6 st / +5.2 dB. We follow their default because our shipped
# expressive 4 + pitch_range 1.5 are exactly the "stacked" case they warn about.
# Only sent when the live bundle advertises `steer_release` — a `steer_hold` below
# 1 without it is a 400.
_DEFAULT_STEER_HOLD = 0.35
_DEFAULT_STEER_ONSET = 4.0
_DEFAULT_STEER_RAMP = 1.0
_UNSET = object()

# Every field `POST /v2/generate` (GenerateRequest) accepts. The object is CLOSED:
# any other key is a 422 ("instance invalid as per false-schema"), measured even
# for a no-op value like `speed: 1.0`. Taken from docs/api.md + the OpenAPI schema,
# so unknown caller kwargs are dropped instead of failing the utterance.
_ENGINE_FIELDS = frozenset({
    "text", "language", "text_normalization", "stream", "stream_decode",
    "stream_codebooks", "stream_codebooks_scope", "format", "seed",
    "temperature", "topk", "top_p", "min_p",
    "repetition_window", "repetition_penalty", "repetition_codebooks",
    "max_tokens", "speaker_audio_base64", "speaker_embedding_base64",
    "speaker_embedding_id", "speaker_reference_codes", "speaker_mode",
    "speaker_blend_embedding_id_a", "speaker_blend_embedding_id_b",
    "speaker_blend_t", "speaker_cfg_scale", "condition_buckets",
    "condition_values", "silence_prefill_frames", "start_immediately",
    "accent", "accent_strength", "controls", "control_spans",
    "control_span_rate", "control_span_follow", "control_span_lead",
    "control_span_lag", "controls_cfg_scale",
    "steer_hold", "steer_onset_seconds", "steer_ramp_seconds",
    "laci", "short_text", "decoder", "headroom_db", "fade_out_ms",
    "decode_context",
})

# --------------------------------------------------------------------------- emotion

# SenseVoice emotion labels -> ZONOS3 steering directions, verified against
# GET /v2/controls on a live server (bundle `studio_mect60k_x10`).
#
# `emotions.*` and `delivery.*` take a dose (1.0 = the bundle's nominal step,
# negative reverses). `mood.*` takes an absolute 0-8 coordinate, so it is
# interpolated from neutral by intensity rather than scaled.
EMOTION_CONTROLS: dict[str, dict[str, float]] = {
    "neutral":   {},
    "happy":     {"emotions.happy": 1.6, "mood.valence": 6.5, "mood.arousal": 5.5},
    "excited":   {"emotions.happy": 2.0, "emotions.ecstatic": 1.2,
                  "mood.arousal": 7.0, "delivery.tempo": 1.5},
    "sad":       {"emotions.sad": 1.6, "mood.valence": 2.5, "mood.arousal": 3.0},
    "angry":     {"emotions.angry": 1.8, "mood.arousal": 6.5, "delivery.loud": 1.0},
    "fearful":   {"emotions.fearful": 1.5, "mood.arousal": 6.5, "mood.valence": 3.0},
    "surprised": {"emotions.surprised": 1.5, "mood.arousal": 6.0},
    "disgusted": {"emotions.disgusted": 1.4, "mood.valence": 2.8},
}

_MOOD_KEYS = frozenset({"mood.valence", "mood.arousal", "mood.dominance",
                        "mood.loudness", "mood.tempo", "mood.clarity",
                        "mood.articulation"})
# The bundle's neutral mood vector (GET /v2/controls -> sets.mood.calibration).
_MOOD_DEFAULT = {"valence": 4.0, "arousal": 4.0, "dominance": 4.0,
                 "loudness": 2.0, "tempo": 4.0, "clarity": 2.0, "articulation": 4.0}


def emotion_controls(emotion: Optional[str], intensity: float = 1.0) -> dict[str, float]:
    """Map a SenseVoice emotion label to ZONOS3 request `controls`.

    *intensity* (0-2) blends between neutral and full expression. Unknown labels
    return ``{}`` — an unrecognised emotion must never break speech, and neutral
    sends no controls at all so the engine uses its own defaults.
    """
    key = (emotion or "neutral").strip().lower()
    spec = EMOTION_CONTROLS.get(key) or {}
    t = max(0.0, min(2.0, float(intensity)))
    if not spec or t <= 0:
        return {}
    out: dict[str, float] = {}
    for name, value in spec.items():
        if name in _MOOD_KEYS:
            neutral = _MOOD_DEFAULT.get(name.split(".", 1)[1], 4.0)
            out[name] = round(neutral + (value - neutral) * t, 2)
        else:
            out[name] = round(value * t, 2)
    return out


# --------------------------------------------------------------------------- client


class EngineClient:
    """Minimal HTTP client for the ZONOS3 server (stdlib only).

    The server is local and speaks plain JSON, so keeping ``pip_dependencies``
    empty is worth more than shaving milliseconds: a plugin that cannot fail to
    install.
    """

    def __init__(self, base_url: str, timeout: float = _DEFAULT_REQUEST_TIMEOUT,
                 token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token
        self.last_error = ""

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(self, path: str, payload: Optional[dict] = None,
                 timeout: Optional[float] = None) -> tuple[int, bytes]:
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=self._headers(),
            method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return resp.status, resp.read()

    def close(self) -> None:
        """Kept for symmetry with the pooled clients; nothing is held open here."""

    def health(self, timeout: float = 2.0) -> bool:
        try:
            status, body = self._request("/v2/health", timeout=timeout)
            return status == 200 and b"ok" in body
        except Exception:  # noqa: BLE001
            return False

    def speakers(self, timeout: float = 8.0) -> list[dict]:
        try:
            status, body = self._request("/v2/speakers", timeout=timeout)
            return json.loads(body).get("speakers", []) if status == 200 else []
        except Exception:  # noqa: BLE001
            return []

    def generate_wav(self, payload: dict, timeout: Optional[float] = None) -> bytes:
        """Buffered WAV synthesis via /v2/generate (stream=false, format=wav)."""
        body = {**payload, "stream": False, "format": "wav"}
        status, raw = self._request("/v2/generate", body,
                                    timeout=timeout or self.timeout)
        if status != 200:
            raise RuntimeError(f"ZONOS3 server returned HTTP {status}")
        if not raw:
            raise RuntimeError("ZONOS3 server returned no audio")
        if raw[:4] != b"RIFF":
            raise RuntimeError(f"ZONOS3 server returned a non-WAV body ({raw[:16]!r})")
        return raw

    def generate_pcm(self, payload: dict, timeout: Optional[float] = None):
        """Streaming synthesis via /v2/generate (stream=true, format=pcm).

        Returns ``(sample_rate, chunks)`` where *chunks* yields **int16 mono**
        blocks. The engine streams **float32** (``X-Audio-Format: float32`` on
        the chunked response) at the decoder's own rate — measured 44100 Hz, not
        the 24000 Hz Hermes assumes — so both the width and the rate are read
        from the response headers and the float32 samples are converted here.
        Handing raw float32 bytes to a consumer that expects int16 produces
        white noise at the wrong speed, not silence, so the conversion is not
        optional.
        """
        body = {**payload, "stream": True, "format": "pcm"}
        req = urllib.request.Request(
            f"{self.base_url}/v2/generate", data=json.dumps(body).encode(),
            headers=self._headers(), method="POST")
        import struct

        resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
        try:
            if resp.status != 200:
                raise RuntimeError(f"ZONOS3 server returned HTTP {resp.status}")
            try:
                rate = int(resp.headers.get("X-Audio-Sample-Rate") or 0)
            except (TypeError, ValueError):
                rate = 0
            if rate <= 0:
                raise RuntimeError(
                    "ZONOS3 streamed audio without X-Audio-Sample-Rate; "
                    "playback rate would be a guess")
            fmt = str(resp.headers.get("X-Audio-Format") or "float32").strip().lower()
            channels = int(resp.headers.get("X-Audio-Channels") or 1)

            def _convert(raw: bytes) -> bytes:
                """float32 (or already-int16) little-endian -> int16 LE."""
                if fmt == "pcm16":
                    usable = len(raw) - (len(raw) % 2)
                    return raw[:usable]
                usable = len(raw) - (len(raw) % 4)
                if usable <= 0:
                    return b""
                # struct is ~2x faster than a Python loop and keeps this stdlib-only.
                samples = struct.unpack("<%df" % (usable // 4), raw[:usable])
                out = array.array("h", (
                    -32768 if v <= -1.0 else 32767 if v >= 1.0 else int(v * 32767.0)
                    for v in samples
                ))
                if sys.byteorder == "big":
                    out.byteswap()
                return out.tobytes()

            def _chunks():
                carry = b""
                frame = 4 if fmt != "pcm16" else 2
                try:
                    while True:
                        block = resp.read(65536)
                        if not block:
                            break
                        block = carry + block
                        usable = len(block) - (len(block) % frame)
                        carry = block[usable:]
                        if usable:
                            pcm = _convert(block[:usable])
                            if pcm:
                                yield pcm
                    # a trailing partial sample is dropped, never misaligned
                finally:
                    # Barge-in departs by breaking out of the consumer's for-loop,
                    # which abandons this generator mid-iteration. Without this the
                    # socket stays open, the engine's writer keeps succeeding, and
                    # it decodes the rest of the utterance into a buffer nobody
                    # reads — burning the GPU exactly when the user is talking.
                    # The engine cancels on a gone response (generate.cpp: "the
                    # response is gone ... stop generating"), so closing here is
                    # what actually stops the work.
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001 - teardown must not raise
                        pass

            return rate, channels, _chunks()
        except Exception:
            resp.close()
            raise



# --------------------------------------------------------------------------- audio


def _samples(raw: bytes):
    """int16 samples of a 16-bit mono WAV body."""
    import array
    samples = array.array("h")
    body = raw[44:]
    samples.frombytes(body[: len(body) - (len(body) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def wav_is_silent(raw: bytes) -> bool:
    """True when a WAV body carries no audible samples.

    The native engine measured 0/12 silent, but a silent result is a *successful*
    HTTP response with a well-formed header, so samples are the only signal that
    distinguishes it from speech. Checking costs one pass over a few hundred KB
    and turns the failure users report as "read-aloud does nothing" into a retry.
    """
    if len(raw) < 44 or raw[:4] != b"RIFF":
        return True
    return not any(_samples(raw))


def wav_peak(raw: bytes) -> int:
    return max((abs(x) for x in _samples(raw)), default=0)


def _split_for_engine(text: str, budget: int) -> list[str]:
    """Split *text* into engine-sized pieces at sentence, then clause, boundaries.

    The engine's generation cap is per request and it does not report hitting it,
    so a long line is either spoken in full as several takes or silently cut. The
    split prefers ``.!?`` followed by whitespace, falls back to ``,;:`` and then to
    a hard cut on whitespace, so a piece never lands mid-word. Nothing is dropped:
    the join of the pieces is the original text apart from collapsed spacing.

    Marker-safe: the start marker (U+E002) and the pause markers (U+E003-U+E006)
    are only legal on a checkpoint that has them, and only as the first character
    (start) or after a word and its glued punctuation (pause). A naive split can
    leave one of those private-use characters at the head of a later piece, which
    the engine answers with a 400. So any orphaned marker in a non-first piece is
    stripped, and a first-piece start marker is preserved.
    """
    import re

    # Private-use markers the checkpoint may tokenize. Everything else in this
    # range is left alone: a caller's own text may legitimately contain it.
    _MARKERS = "\ue002\ue003\ue004\ue005\ue006"

    text = text.strip()
    if len(text) <= budget:
        return [text]

    boundaries = [r'(?<=[.!?])\s+', r'(?<=[,;:])\s+', r'\s+']
    pieces: list[str] = []
    rest = text
    while len(rest) > budget:
        cut = -1
        for pattern in boundaries:
            candidates = [m.start() for m in re.finditer(pattern, rest[:budget])]
            if candidates:
                cut = candidates[-1]
                break
        if cut <= 0:
            cut = budget            # one gigantic token; nothing sane to split on
        piece = rest[:cut].strip()
        if piece:
            pieces.append(piece)
        rest = rest[cut:].lstrip()
    if rest.strip():
        pieces.append(rest.strip())

    # Keep a start marker only at the very beginning; drop markers that a cut
    # stranded at the head of a later piece (they would be a 400 there).
    return [
        piece if index == 0 else piece.lstrip(_MARKERS).lstrip()
        for index, piece in enumerate(pieces)
    ]


def _merge_wav(parts: list[bytes]) -> bytes:
    """Concatenate WAV bodies into one WAV, keeping the first header's format.

    Concatenating whole files would leave a 44-byte header mid-stream that a
    player treats as audio (audible click, and the duration lies). Splicing only
    the PCM data and rewriting the size fields produces one honest container.
    """
    if not parts:
        return b""
    if len(parts) == 1:
        return parts[0]
    import struct
    first = parts[0]
    if len(first) < 44 or first[:4] != b"RIFF":
        return first
    pcm = bytearray()
    for part in parts:
        # Walk to the data chunk rather than assuming 44 bytes: some encoders
        # emit a LIST/INFO chunk before it.
        if len(part) < 44 or part[:4] != b"RIFF":
            continue
        offset = 12
        while offset + 8 <= len(part):
            chunk_id = part[offset:offset + 4]
            chunk_size = struct.unpack("<I", part[offset + 4:offset + 8])[0]
            if chunk_id == b"data":
                pcm += part[offset + 8: offset + 8 + chunk_size]
                break
            offset += 8 + chunk_size + (chunk_size & 1)
    header = bytearray(first[:first.find(b"data") + 8])
    struct.pack_into("<I", header, 4, 36 + len(pcm))
    struct.pack_into("<I", header, len(header) - 4, len(pcm))
    return bytes(header) + bytes(pcm)


# --------------------------------------------------------------------------- engine


class EngineSupervisor:
    """Owns the ZONOS3 server process and the lease/idle state machine.

    Residency state follows evidence, never intent:

    * a successful synthesis proves the model is resident -> ``warm``;
    * a lease alone means at most ``loading``;
    * ``unloaded`` only after the process is verifiably gone.

    Hermes treats TTS leases as a *set*: ``acquire`` re-warms on every call and
    ``release`` fires once, when the last holder leaves. Counting acquire calls
    leaks a reference per re-acquire and the unload never arms, so the held flag
    is a boolean and ``release`` is authoritative.
    """

    def __init__(self, *, runtime_dir: Path, model_dir: Path,
                 host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT,
                 base_url: str = "", gpu: bool = True,
                 unload_after: float = _DEFAULT_UNLOAD_AFTER,
                 start_timeout: float = _DEFAULT_START_TIMEOUT,
                 request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
                 retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
                 default_voice: str = "", settings_fn=None,
                 server_exe: Optional[Path] = None):
        self.runtime_dir = runtime_dir
        self.model_dir = model_dir
        self.host = host
        self.port = port
        self.gpu = gpu
        self.unload_after = unload_after
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        self.retry_attempts = max(1, int(retry_attempts))
        self.default_voice = default_voice
        self._settings_fn = settings_fn
        self._server_exe = server_exe
        self._log_path = runtime_dir / "engine.log"

        self.client = EngineClient(base_url or f"http://{host}:{port}",
                                   timeout=request_timeout)
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._starting = False
        self._idle_timer: Optional[threading.Timer] = None

        self.state = "idle"   # idle | starting | loading | warm | unloaded | error
        self.last_error = ""
        self.last_use = 0.0
        self.loaded_at = 0.0
        self._held = False
        self._publisher = None
        self._voices: list[dict] = []
        # speaker id -> controls the server reports as inert for it
        self._disabled_cache: dict[str, frozenset[str]] = {}

    # -- settings ---------------------------------------------------------
    def _live(self, key: str, fallback: float) -> float:
        fn = self._settings_fn
        if fn is None:
            return fallback
        try:
            return max(0.0, float(fn(key, fallback)))
        except Exception:  # noqa: BLE001 - settings must never break speech
            return fallback

    def _notify(self) -> None:
        if self._publisher is not None:
            try:
                self._publisher(self)
            except Exception:  # noqa: BLE001 - telemetry only
                pass

    # -- process ----------------------------------------------------------
    @property
    def exe(self) -> Path:
        if self._server_exe is not None:
            return self._server_exe
        name = "zonos3-server.exe" if os.name == "nt" else "zonos3-server"
        return self.runtime_dir / name

    def is_up(self) -> bool:
        return self.client.health()

    def _log_tail(self, lines: int = 10) -> str:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
            return " | ".join(text.strip().splitlines()[-lines:])
        except Exception:  # noqa: BLE001
            return "(no log)"

    def _argv(self) -> list[str]:
        """Server arguments.

        ``--reference-idle-unload`` governs only the raw-reference encoder; the
        backbone is loaded at startup and never released. Process lifetime is
        therefore ours to manage — that is what :meth:`stop_process` does.
        """
        gguf = self.model_dir
        args = [
            str(self.exe),
            str(gguf / "backbone-ptq.gguf"),
            "--port", str(self.port),
            "--host", self.host,
            "--spk-encoder", str(gguf / "spk-encoder.gguf"),
            "--tts-default-voices-dir", str(gguf / "voices"),
        ]
        if self.gpu:
            args.append("--gpu")
        steer = gguf / "steer.gguf"
        if steer.is_file():
            # Enables GET /v2/controls and request `controls` — the emotion path.
            args += ["--steer", str(steer)]
        # Raw-reference cloning companions. The release backbone is conditioned on
        # a raw reference prefix (capabilities: speaker_conditioning_type =
        # "raw_reference_prefix"), so it rejects speaker-embedding files entirely;
        # these two GGUFs are what make `speaker_mode: reference` — real cloning
        # from an arbitrary clip — possible. They are optional: without them the
        # server still answers with bundled voices and we simply do not advertise
        # cloning.
        zac_enc = gguf / "zac-encoder.gguf"
        teacher = gguf / "ssl-teacher.gguf"
        if zac_enc.is_file() and teacher.is_file():
            args += ["--zac-encoder", str(zac_enc), "--ssl-teacher", str(teacher)]
        # The raw-reference encoder is ~3.4 GiB when resident, so let it decay a
        # minute after the last clone rather than holding it for the session.
        args += ["--reference-idle-unload", "60"]
        return args

    @property
    def cloning_supported(self) -> bool:
        """True when the raw-reference encoder companions are present.

        Checked on disk rather than by probing the server: the server only learns
        about them at launch, and a flag that disagrees with the spawned process
        would let the plugin offer cloning the engine cannot perform.
        """
        gguf = self.model_dir
        return (gguf / "zac-encoder.gguf").is_file() and (gguf / "ssl-teacher.gguf").is_file()

    def _spawn_locked(self) -> None:
        if self.is_up() or self._starting:
            return
        if not self.exe.is_file():
            raise RuntimeError(
                f"zonos3-tts: server binary missing at {self.exe}. Build it from "
                "servers/ggml (plugin README) or set settings.server_exe.")
        args = self._argv()
        self._starting = True
        self.state = "starting"
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self._log_path, "ab")
            self._proc = subprocess.Popen(
                args, cwd=str(self.runtime_dir), stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt" else 0)
        finally:
            self._starting = False
        logger.info("zonos3-tts: server started pid=%s port=%s",
                    getattr(self._proc, "pid", "?"), self.port)

    def _wait_up(self, deadline: float) -> bool:
        while time.time() < deadline:
            if self.is_up():
                return True
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"zonos3-server exited during startup: {self._log_tail()}")
            time.sleep(0.2)
        return False

    def ensure_up(self) -> None:
        with self._lock:
            if self.is_up():
                return
            # A concurrent stop is mid-flight; starting a second server now would
            # race it for the port and could leave one orphaned. Wait it out
            # (bounded) before spawning.
            deadline = time.time() + 15.0
            while self.state == "unloading" and time.time() < deadline:
                self._lock.release()
                try:
                    time.sleep(0.1)
                finally:
                    self._lock.acquire()
            if self.is_up():
                return
            self._spawn_locked()
            if not self._wait_up(time.time() + self.start_timeout):
                raise RuntimeError(
                    f"zonos3-tts: server not healthy on port {self.port}: {self._log_tail()}")

    # -- leases -----------------------------------------------------------
    def warm(self) -> None:
        """A surface took a lease: cancel the pending stop and bring the server up.

        Called synchronously from the Read-aloud toggle. Startup is ~2 s because
        the backbone loads before the listener binds, so when this returns the
        model genuinely is ready — the first sentence does not pay for the load.
        """
        with self._lock:
            self._cancel_timer_locked()
            self._held = True
            if self.state in ("idle", "unloaded", "error"):
                self.state = "starting"
        self._notify()
        try:
            self.ensure_up()
            with self._lock:
                if self.state in ("starting", "idle", "unloaded", "error"):
                    # Listening and serving implies the backbone is resident, so
                    # "warm" is evidence here rather than optimism.
                    self.state = "warm"
                    self.loaded_at = time.time()
        except Exception as exc:  # noqa: BLE001 - reported, never raised into the hook
            with self._lock:
                self.state = "error"
                self.last_error = str(exc)
            logger.warning("zonos3-tts: warm failed: %s", exc)
        self._notify()

    def release(self) -> None:
        """The last lease went away: arm the idle stop."""
        stop_now = False
        with self._lock:
            was_held = self._held
            self._held = False
            if not was_held:
                return
            window = self._live("unload_after_seconds", self.unload_after)
            if window <= 0:
                stop_now = True
            else:
                self._cancel_timer_locked()
                t = threading.Timer(window, self._stop_after_idle)
                t.daemon = True
                self._idle_timer = t
                t.start()
        if stop_now:
            self.stop_process()
        else:
            self._notify()

    def _cancel_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _stop_after_idle(self) -> None:
        with self._lock:
            if self._held:
                return
        self.stop_process()

    # -- evidence ---------------------------------------------------------
    def mark_warm(self) -> None:
        with self._lock:
            self.state = "warm"
            self.last_use = time.time()
            self.loaded_at = self.loaded_at or time.time()
            self.last_error = ""
        self._notify()

    def is_ready(self) -> bool:
        with self._lock:
            return self.state == "warm"

    # -- lifecycle --------------------------------------------------------
    # Cross-process manual unload. The dashboard backend (`plugin_api.py`) and
    # this supervisor often live in *different* processes: the route is served by
    # the host that mounted it, while the engine is owned by whichever process
    # registered the plugin. A route cannot call `stop_process()` across that
    # boundary (and must never try — a forked supervisor would spawn a second
    # engine), so a request is handed over as a small JSON file the supervisor's
    # own poll loop consumes.
    _UNLOAD_REQUEST = "_unload_request.json"

    def _request_dir(self) -> Path:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        return home / "cache" / "zonos3-tts"

    def request_unload(self, reason: str = "manual") -> bool:
        """Ask the owning process to unload. Safe to call from any process."""
        try:
            d = self._request_dir()
            d.mkdir(parents=True, exist_ok=True)
            target = d / self._UNLOAD_REQUEST
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"reason": reason, "requested_at": time.time()}),
                           encoding="utf-8")
            os.replace(tmp, target)
            return True
        except Exception as exc:  # noqa: BLE001 - a UI request must not raise
            logger.warning("zonos3-tts: could not write unload request: %s", exc)
            return False

    def _take_unload_request(self) -> bool:
        """Consume a pending request. True when one was waiting.

        Rename-then-delete so two readers cannot both act on one request.
        """
        path = self._request_dir() / self._UNLOAD_REQUEST
        try:
            claim = path.with_suffix(".claimed")
            os.replace(path, claim)
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.debug("zonos3-tts: unload request unreadable: %s", exc)
            return False
        try:
            claim.unlink()
        except OSError:
            pass
        return True

    def poll_unload_request(self) -> bool:
        """Act on a pending manual unload. Returns True when one was honoured."""
        if not self._take_unload_request():
            return False
        with self._lock:
            held = self._held
        if held:
            # A surface still holds the engine. Unloading now would leave a live
            # lease pointing at a dead process, and the next utterance would pay
            # a cold start mid-sentence. Refuse and say so.
            logger.info("zonos3-tts: manual unload refused while %s holds a lease", held)
            return False
        logger.info("zonos3-tts: manual unload requested")
        return self.stop_process()

    def _stop_orphan_engine(self) -> bool:
        """Stop an engine this object did not spawn. True when a process was stopped.

        The supervisor normally owns its child, but it also *adopts* a server that
        is already listening (a previous agent process leaving a child behind, or
        one started by hand). Adoption records no ``_proc``, so an unload that
        only terminates the child leaves the engine resident and the state stuck
        at "warm" forever.

        Scoped tightly on purpose: only a process whose image name matches this
        plugin's server binary is a candidate, and only when it is not this
        process. Killing by name alone would be reckless on a shared host, so the
        match is the server binary's own stem.
        """
        name = self.exe.name if getattr(self, "exe", None) else "zonos3-server.exe"
        stem = Path(name).stem
        stopped = False
        try:
            if os.name == "nt":
                out = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    stdin=subprocess.DEVNULL).stdout or ""
                pids = []
                for line in out.splitlines():
                    parts = [p.strip('"') for p in line.split('","')]
                    if len(parts) >= 2 and parts[0].lower() == name.lower():
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            pass
                for pid in pids:
                    if pid == os.getpid():
                        continue
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True, text=True, timeout=20,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            stdin=subprocess.DEVNULL)
                        logger.info("zonos3-tts: stopped orphan engine pid %d", pid)
                        stopped = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("zonos3-tts: could not stop orphan pid %d: %s", pid, exc)
            else:
                # POSIX: match the server's own binary path, never a bare name.
                out = subprocess.run(["pgrep", "-f", stem], capture_output=True,
                                     text=True, timeout=20).stdout or ""
                for raw in out.split():
                    try:
                        pid = int(raw)
                    except ValueError:
                        continue
                    if pid == os.getpid():
                        continue
                    try:
                        os.kill(pid, 15)
                        stopped = True
                    except OSError:
                        pass
        except Exception as exc:  # noqa: BLE001 - unload must report, not raise
            logger.warning("zonos3-tts: orphan engine stop failed: %s", exc)
        return stopped

    def stop_process(self) -> bool:
        """Stop the server, returning every byte of VRAM it held.

        The server has no route to release its backbone (only the reference
        encoder has an idle reaper), so the process is the unit of unload.
        Stopping it reclaims the weights *and* the KV cache, decoder and CUDA
        context — strictly more than a model-only unload would.
        """
        with self._lock:
            proc, self._proc = self._proc, None
            self._cancel_timer_locked()
            # Mark the transition explicitly. Terminating, waiting on the child
            # and polling the socket all take time, and an observer (a status
            # chip, a lease acquire) must never see "warm" for a server that is
            # already dead — that reads as a loaded model with no process behind
            # it, which nothing would ever correct.
            self.state = "unloading"
        self._notify()
        exited = proc is None or proc.poll() is not None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
                exited = True
            except Exception:  # noqa: BLE001 - escalate
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                    exited = True
                except Exception:  # noqa: BLE001
                    pass
        # A server started outside this object (or an orphan) still holds VRAM,
        # and only a probe can tell us whether it is really gone. The socket can
        # answer for a moment after the process exits, so poll rather than
        # reading once — a single look reported "still warm" for a server that
        # was already dead, and nothing re-checked afterwards.
        if proc is None:
            # No child handle: the engine was adopted over the socket (started by
            # a previous process, or by hand). `exited` is trivially True here,
            # but the engine is still resident, so the loop below would conclude
            # `down = True and not still_up` = False and report "still warm"
            # forever — a manual unload that silently never unloads. Find and stop
            # the actual process instead.
            exited = self._stop_orphan_engine()
            if exited:
                time.sleep(0.3)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if not self.is_up():
                break
            time.sleep(0.2)
        still_up = self.is_up()
        # The child's own exit is the strongest evidence; the probe covers the
        # orphan case only. Reading the probe alone left a window where the
        # socket had closed but the state still said "warm".
        down = exited and not still_up
        with self._lock:
            self.state = "unloaded" if down else "warm"
            if down:
                self.loaded_at = 0.0
        self._notify()
        return down

    def synthesize(self, payload: dict, timeout: Optional[float] = None) -> bytes:
        """Synthesise WAV bytes, restarting the server if the idle policy stopped it."""
        try:
            return self.client.generate_wav(payload, timeout=timeout or self.request_timeout)
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            logger.info("zonos3-tts: engine was down (%s); restarting", exc)
            self.ensure_up()
            return self.client.generate_wav(payload, timeout=timeout or self.request_timeout)

    # -- voices -----------------------------------------------------------
    def voices(self, refresh: bool = False) -> list[dict]:
        if self._voices and not refresh:
            return self._voices
        people = self.client.speakers()
        if people:
            self._voices = people
        return self._voices

    def disabled_controls(self, speaker_id: Optional[str]) -> frozenset[str]:
        """Controls the server says do nothing for *speaker_id*.

        `GET /v2/speakers` reports each voice's `disabled_controls` — the control
        ids whose calibrated dose is 0 for that voice. Sending one is not an
        error, but it spends dose budget on a direction that cannot move.
        Cached per id: the list is static for a loaded checkpoint.
        """
        if not speaker_id:
            return frozenset()
        cached = self._disabled_cache.get(speaker_id)
        if cached is not None:
            return cached
        found: frozenset[str] = frozenset()
        try:
            people = self.voices(refresh=True)
            for person in people:
                if person.get("id") == speaker_id:
                    found = frozenset(
                        str(c) for c in (person.get("disabled_controls") or []))
                    break
        except Exception:  # noqa: BLE001 - a probe must not break speech
            found = frozenset()
        self._disabled_cache[speaker_id] = found
        return found

    def resolve_voice(self, voice: Optional[str]) -> Optional[str]:
        """Map a name ("Savannah") or an id to a speaker id the server accepts.

        Server ids are opaque hashes (``default_de4d191...``), so a readable name
        has to be matched against the labels the server reports. The voice list
        lives on the server, so a cold start has to be up before the lookup is
        meaningful — otherwise the first synthesis of a session silently falls
        back to the server's default voice. Warming here is cheap (the health
        probe is on loopback) and idempotent.
        """
        want = (voice or self.default_voice or "").strip()
        if not want:
            return None
        people = self.voices()
        if not people:
            # Cold engine: no catalogue to match against yet. Bring it up (the
            # caller is about to synthesize anyway) and re-read.
            try:
                self.ensure_up()
                people = self.voices(refresh=True)
            except Exception as exc:  # noqa: BLE001 - a probe failure is not fatal
                logger.debug("zonos3-tts: could not load the voice list: %s", exc)
        for p in people:
            if p.get("id") == want:
                return want
        needle = want.lower()
        for p in people:
            label = str(p.get("label") or "").lower()
            name = str(p.get("original_name") or "").lower()
            if needle == label or needle in label or needle in name:
                return p.get("id")
        for p in people:
            if str(p.get("original_name") or "").lower().startswith(needle):
                return p.get("id")
        logger.warning("zonos3-tts: voice %r not among %d server voices; using default",
                       want, len(people))
        return None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "tts": self.state,
                "tts_leased": 1 if self._held else 0,
                "unload_after": self._live("unload_after_seconds", self.unload_after),
                "pid": getattr(self._proc, "pid", 0) or 0,
                "last_error": self.last_error,
                "endpoint": self.client.base_url,
                "voice": self.default_voice,
                "voices": len(self._voices),
                "retry_attempts": self.retry_attempts,
            }


# --------------------------------------------------------------------------- provider


class Zonos3TTS(TTSProvider):
    """Hermes TTS provider backed by a supervised ZONOS3 server."""

    name = "zonos3"
    display_name = "ZONOS3-Mini"

    def __init__(self, supervisor: EngineSupervisor, *, options: Optional[dict] = None,
                 emotion_aware: bool = True, language: str = "en_us",
                 dynamic_range: float = 0.0, expressive: float = 0.0,
                 volume_boost: float = 0.0, streams_pcm: bool = False,
                 steer_hold: Optional[float] = None,
                 steer_onset: Optional[float] = None,
                 steer_ramp: Optional[float] = None,
                 headroom_db: Optional[float] = None):
        self._sup = supervisor
        self._base_options = dict(options or {})
        self._emotion_aware = emotion_aware
        self._language = language
        self._emotion: Optional[str] = None
        self._emotion_intensity = 1.0
        self._explicit_controls: dict[str, float] = {}
        self._emotion_lock = threading.Lock()
        # Always-on prosody shaping. These are not "emotions" — they set how
        # lively and how loud every utterance is, independent of affect.
        # pitch.pitch_range needs speaker_mode=embedding (a reference-clip voice
        # cannot be re-projected), so it is only emitted for bundled voices.
        self._dynamic_range = float(dynamic_range or 0.0)
        self._expressive = float(expressive or 0.0)
        self._volume_boost = float(volume_boost or 0.0)
        # Steer-then-hold window. `steer_hold >= 1` means "constant push", which
        # is the old behaviour and never needs the bundle's block.
        self._steer_hold = float(steer_hold if steer_hold is not None else _DEFAULT_STEER_HOLD)
        self._steer_onset = float(steer_onset if steer_onset is not None else _DEFAULT_STEER_ONSET)
        self._steer_ramp = float(steer_ramp if steer_ramp is not None else _DEFAULT_STEER_RAMP)
        self._steer_release: Any = _UNSET
        self._max_frames: Optional[int] = None
        # 0 recovers the engine's default 3 dB attenuation for free; None leaves
        # the server default alone rather than assuming it.
        self._headroom_db = None if headroom_db is None else max(0.0, float(headroom_db))
        # Hermes only takes the raw-PCM path when this is true AND
        # stream_sample_rate is valid; the rate is read from the engine's own
        # X-Audio-Sample-Rate header at stream time, so this is just the opt-in.
        self.streams_pcm = bool(streams_pcm)
        self.stream_sample_rate = 44100  # provisional; the stream overwrites it

    def prosody_controls(self, has_clip_voice: bool = False,
                         scale: float = 1.0) -> dict[str, float]:
        """Always-on shaping applied on top of emotion controls.

        Measured on this host (baseline F0 spread 3.17 st):
          pitch.pitch_range +2 -> 5.93 st, expressive +16 -> 4.25 st,
          delivery.loud 4 -> +4.7 dB peak / +6.8 dB RMS.
        ``delivery.loud`` is the engine's only real gain; mood.loudness
        measured no gain (-2.2 dB peak) and is deliberately not used.

        *scale* damps the whole set for a long utterance. See
        :meth:`steer_scale` for why.
        """
        s = max(0.0, min(1.0, float(scale)))
        out: dict[str, float] = {}
        if self._expressive:
            out["expressive.expressive"] = round(self._expressive * s, 3)
        if self._volume_boost:
            out["delivery.loud"] = round(self._volume_boost * s, 3)
        # A reference-clip voice is conditioned on raw audio codes, so moving the
        # embedding cannot apply; the engine 400s instead of ignoring it.
        if self._dynamic_range and not has_clip_voice:
            out["pitch.pitch_range"] = round(self._dynamic_range * s, 3)
        return out

    def steer_scale(self, text: str, hold: Optional[float] = None) -> float:
        """Client-side stand-in for the bundle's `steer_release` (steer-then-hold).

        The guide measured what a constant hidden push does over a long line: the
        model copies the style of the speech it has already generated, so stacked
        liveliness + valence + arousal climb in pitch and loudness — a stacked
        design went +7.2 st and +7.0 dB after the first 5 s in 19 of 24 takes.
        The engine fixes that with `steer_release` (full dose for an onset, ramp,
        then hold), but **this bundle does not ship it** (`/v2/capabilities`
        `steer_release: null`), so we cannot ask for the ramp.

        So the same curve is applied here, per request: estimate how long the text
        will take, and damp the controls to `hold` once past onset + ramp. Short
        utterances are unaffected (they finish before the onset); a long one stops
        accumulating. The estimate is deliberately conservative — speech runs about
        14 characters per second at this voice, measured across this plugin's
        renders.
        """
        h = self._steer_hold if hold is None else float(hold)
        if not (0.0 <= h < 1.0):
            return 1.0
        seconds = max(0.0, len(text) / 14.0)
        onset, ramp = self._steer_onset, max(1e-3, self._steer_ramp)
        if seconds <= onset:
            return 1.0
        if seconds >= onset + ramp:
            return h
        return 1.0 - (1.0 - h) * ((seconds - onset) / ramp)



    # -- lease hooks ------------------------------------------------------
    def warm(self) -> None:
        self._sup.warm()

    def release(self) -> None:
        self._sup.release()

    def is_ready(self) -> bool:
        return self._sup.is_ready()

    def is_available(self) -> bool:
        """No network call — the picker paints this on every render."""
        return self._sup.exe.is_file() and (self._sup.model_dir / "backbone-ptq.gguf").is_file()

    @property
    def voice_compatible(self) -> bool:
        return True

    def list_voices(self) -> list[dict]:
        return [{"id": p.get("id"),
                 "display": p.get("label") or p.get("original_name") or p.get("id"),
                 "language": "", "preview_url": ""}
                for p in self._sup.voices() if p.get("id")]

    def default_voice(self) -> Optional[str]:
        if self._sup.default_voice:
            return self._sup.resolve_voice(self._sup.default_voice)
        people = self._sup.voices()
        return people[0].get("id") if people else None

    # -- emotion bridge ---------------------------------------------------
    def set_emotion(self, emotion: Optional[str], intensity: float = 1.0) -> None:
        """Set the affect for subsequent utterances — the SenseVoice feed point.

        The STT side of a voice pipeline calls this with SenseVoice's detected
        emotion so the spoken reply matches the user's affect. Thread-safe; clear
        it with :meth:`clear_emotion`.

        Note the direction: this steers the *reply* toward the user's emotion. For
        a sympathetic-but-steady reply use :meth:`set_emotion_controls` with
        ``emotion.reply_prosody``, which is what the built-in pipeline does —
        echoing a user's anger back at them escalates rather than soothes.
        """
        with self._emotion_lock:
            self._emotion = emotion
            self._emotion_intensity = intensity
            self._explicit_controls = {}

    def set_emotion_controls(self, controls: dict[str, float]) -> None:
        """Steer the reply with pre-resolved ZONOS3 controls.

        Used by :class:`EmotionPipeline` so the reply's vocal stance is decided
        once, in :mod:`emotion`, rather than re-derived per utterance.
        """
        with self._emotion_lock:
            self._explicit_controls = dict(controls or {})
            if not controls:
                self._emotion = None
                self._emotion_intensity = 1.0

    def clear_emotion(self) -> None:
        with self._emotion_lock:
            self._emotion = None
            self._emotion_intensity = 1.0
            self._explicit_controls = {}

    def current_emotion(self) -> Optional[str]:
        with self._emotion_lock:
            return self._emotion

    @property
    def steering_supported(self) -> bool:
        """True when the running server actually serves a steering bundle.

        Checked against the live server rather than the filesystem: a bundle on
        disk that the server was *not* started with (before it existed, or with a
        different model dir) would otherwise be claimed as available and every
        `controls` request would 400.
        """
        if not (self._sup.model_dir / "steer.gguf").is_file():
            return False
        try:
            status, body = self._sup.client._request("/v2/controls", timeout=3.0)
            return status == 200 and b"directions" in body
        except Exception:  # noqa: BLE001 - no bundle, or the server is down
            return False

    def steering_release(self) -> Optional[dict]:
        """The bundle's `steer_release` defaults, or None when it has no block.

        Sending `steer_hold` below 1 to a server whose bundle lacks this block is
        a 400, so the value has to be read from the live server rather than
        assumed. Cached: the bundle cannot change without an engine restart.
        """
        if self._steer_release is not _UNSET:
            return self._steer_release
        found = None
        try:
            status, body = self._sup.client._request("/v2/controls", timeout=3.0)
            if status == 200:
                block = json.loads(body).get("steer_release")
                if isinstance(block, dict) and block:
                    found = block
        except Exception:  # noqa: BLE001 - treat as absent
            found = None
        self._steer_release = found
        return found

    @property
    def max_frames(self) -> int:
        """The server's generation cap in codec frames (15 fps), cached."""
        if self._max_frames is None:
            frames = 0
            try:
                status, body = self._sup.client._request("/v2/capabilities", timeout=3.0)
                if status == 200:
                    frames = int(json.loads(body).get("max_tokens") or 0)
            except Exception:  # noqa: BLE001
                frames = 0
            self._max_frames = frames
        return self._max_frames

    # -- request ----------------------------------------------------------
    @staticmethod
    def _encode_clip(path: Path, max_seconds: int = 30) -> str:
        """base64 of a 16 kHz mono PCM WAV for `speaker_audio_base64`.

        The engine wants WAV and says so for reference audio; a caller's mp3/flac
        would otherwise fail deep inside the decoder. ffmpeg is resolved through
        the host's own helper first (a bundled ffmpeg is invisible to `which`).
        """
        import base64
        import shutil
        import subprocess
        import tempfile

        if not path.is_file():
            raise RuntimeError(f"zonos3-tts: voice clip not found: {path}")
        ffmpeg = shutil.which("ffmpeg")
        try:
            from tools.transcription_audio import _find_ffmpeg_binary
            ffmpeg = _find_ffmpeg_binary() or ffmpeg
        except Exception:  # noqa: BLE001 - host helper is a convenience
            pass
        if not ffmpeg:
            raise RuntimeError("zonos3-tts: ffmpeg is required to prepare a voice clip")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ref.wav"
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", str(path), "-t", str(max_seconds),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
                capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
            if proc.returncode != 0 or not out.is_file():
                raise RuntimeError(
                    f"zonos3-tts: could not decode {path.name}: {proc.stderr[:200]}")
            return base64.b64encode(out.read_bytes()).decode()

    def _payload(self, text: str, voice: Optional[str], speed: Optional[float],
                 extra: dict, speaker: Optional[str] = None,
                 clip_b64: Optional[str] = None) -> dict:
        """Build a GenerateRequest.

        *speaker* / *clip_b64* let a caller that renders several chunks resolve and
        encode the voice ONCE. Without them each chunk re-resolves the voice (a
        speaker-list round trip) and, for a cloned clip, re-runs ffmpeg — minutes
        of wasted work on a long reply.
        """
        payload: dict[str, Any] = {
            "text": text,
            "language": self._language,
            "stream": False,
            "format": "wav",
        }
        # A voice may be a bundled name/id, OR a path to an audio clip to clone.
        # Cloning is only offered when the raw-reference encoders are present:
        # this checkpoint rejects embedding files outright, so silently accepting
        # a clip and falling back to the default voice would be a lie the user
        # would hear as "it ignored my sample".
        clip_voice = (voice or "").strip()
        if clip_b64 is not None:
            payload["speaker_audio_base64"] = clip_b64
            payload["speaker_mode"] = "reference"
        elif clip_voice and Path(clip_voice).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            if not self._sup.cloning_supported:
                raise RuntimeError(
                    "zonos3-tts: cloning from an audio clip needs zac-encoder.gguf and "
                    "ssl-teacher.gguf in the model directory (see the README); this "
                    "checkpoint cannot take speaker embeddings instead")
            payload["speaker_audio_base64"] = self._encode_clip(Path(clip_voice))
            payload["speaker_mode"] = "reference"
        else:
            resolved = speaker if speaker else self._sup.resolve_voice(voice)
            if resolved:
                payload["speaker_embedding_id"] = resolved
                # `pitch.pitch_range` re-projects the speaker embedding, so the
                # server refuses it (400) unless the mode is `embedding`. The
                # default mode for a bundled voice id is `reference`, which is
                # conditioned on cached raw codes and cannot be re-projected.
                # Requesting embedding mode keeps dynamic range working for
                # every bundled voice without touching the cloning path.
                if any(k.startswith("pitch.") for k in
                       (self.prosody_controls(False) if self.steering_supported else {})):
                    payload["speaker_mode"] = "embedding"

        with self._emotion_lock:
            emotion = self._emotion
            intensity = self._emotion_intensity
            explicit = dict(self._explicit_controls)
        controls: dict[str, float] = {}
        if explicit:
            # Pre-resolved reply prosody from EmotionPipeline wins: it already
            # encoded "meet then settle", which a raw emotion label cannot.
            controls.update(explicit)
        elif self._emotion_aware and emotion:
            wanted = emotion_controls(emotion, intensity)
            if wanted:
                # Sending `controls` to a server without a steering bundle is a
                # hard 400 ("This server has no steering bundle"), which would
                # take the whole utterance down. Check the live capability first.
                if self.steering_supported:
                    controls.update(wanted)
                else:
                    logger.debug(
                        "zonos3-tts: emotion %r not applied — no steering bundle served",
                        emotion)
        # An explicit request beats the ambient mood.
        if isinstance(extra.get("controls"), dict) and self.steering_supported:
            controls.update({k: float(v) for k, v in extra["controls"].items()})
        if speed is not None and abs(float(speed) - 1.0) > 1e-3 and self.steering_supported:
            # ZONOS3 exposes rate as a steering direction, not a sampler knob.
            controls["delivery.tempo"] = round((float(speed) - 1.0) * 4.0, 3)
        # Always-on prosody last, so a configured baseline cannot be silently
        # dropped by an emotion preset that happens to name the same direction.
        if self.steering_supported:
            # Damp the shaping for a long line: this bundle has no `steer_release`,
            # so the engine cannot ramp it and a constant push accumulates.
            scale = self.steer_scale(text)
            for key, value in self.prosody_controls(
                    bool(payload.get("speaker_audio_base64")), scale).items():
                controls.setdefault(key, value)
        if controls:
            # A speaker may declare controls that do nothing for it
            # (`disabled_controls` from GET /v2/speakers — the calibrated dose is 0).
            # Sending them is not an error, but it wastes the dose budget and, when
            # several share a source, dilutes the ones that do work. Drop them.
            disabled = self._sup.disabled_controls(payload.get("speaker_embedding_id"))
            if disabled:
                dropped = [k for k in controls if k in disabled]
                if dropped:
                    logger.debug("zonos3-tts: dropping %d disabled control(s) for this "
                                 "voice: %s", len(dropped), dropped)
                    controls = {k: v for k, v in controls.items() if k not in disabled}
            if controls:
                payload["controls"] = controls
            else:
                payload.pop("controls", None)
            # Steer-then-hold: only meaningful when something is actually being
            # steered, and only sendable when the bundle advertises the block.
            # Without it, `steer_hold < 1` is a hard 400, so the capability is
            # read from the live server rather than assumed from the docs.
            if 0.0 <= self._steer_hold < 1.0 and self.steering_release():
                payload.setdefault("steer_hold", round(self._steer_hold, 3))
                payload.setdefault("steer_onset_seconds", round(self._steer_onset, 2))
                payload.setdefault("steer_ramp_seconds", round(self._steer_ramp, 2))

        # The engine attenuates every request by `headroom_db: 3` (the default) and
        # then runs a soft limiter above -1 dBFS. Passing 0 recovers that 3 dB for
        # free, with no post-processing — measured +3.0 dB peak and +3.0 dB RMS on
        # this host. Values below 0 are a 422, so the floor is 0.
        if self._headroom_db is not None:
            payload.setdefault("headroom_db", float(self._headroom_db))

        # Pass through engine request fields (seed, temperature, guidance_scale,
        # accent, condition_values, …). Every request object on this server is
        # CLOSED: an unknown field is a 422, not a no-op — measured, even
        # `speed: 1.0` and `tag_cfg_scale: 1.0` are rejected. So anything the host
        # hands us that the engine does not accept must be dropped here, or one
        # caller kwarg (the host's own `speed`, `voice_compatible`, …) would take
        # the whole utterance down with a validation error.
        for key, value in extra.items():
            if value is None or key in payload or key.startswith("_"):
                continue
            if _ENGINE_FIELDS and key not in _ENGINE_FIELDS:
                logger.debug("zonos3-tts: dropping non-engine field %r", key)
                continue
            payload[key] = value
        for key, value in self._base_options.items():
            if _ENGINE_FIELDS and key not in _ENGINE_FIELDS:
                logger.debug("zonos3-tts: dropping non-engine request_option %r", key)
                continue
            payload.setdefault(key, value)
        return payload

    def stream(self, text: str, *, voice: Optional[str] = None,
               model: Optional[str] = None, speed: Optional[float] = None,
               format: str = "pcm", **extra: Any):
        """Yield int16 mono PCM chunks for *text* (Hermes streaming contract).

        A generator on purpose: nothing runs until the first chunk is pulled, so
        the engine is only woken when a consumer actually exists, and
        ``self.stream_sample_rate`` can be updated from the response headers
        before the caller opens its output device at a guessed rate.

        The engine streams float32 at its decoder's rate (44100 Hz measured);
        ``EngineClient.generate_pcm`` converts to int16 and reports the real
        rate, so no resampling happens in this plugin.
        """
        if not text or not text.strip():
            return
        payload = self._payload(text, voice, speed, extra)
        chunks = None
        try:
            try:
                rate, _channels, chunks = self._sup.client.generate_pcm(
                    payload, timeout=self._sup.request_timeout)
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                # A stopped engine must not look like a broken stream: restart and
                # retry once, mirroring synthesize().
                logger.info("zonos3-tts: streaming engine was down (%s); restarting", exc)
                self._sup.ensure_up()
                rate, _channels, chunks = self._sup.client.generate_pcm(
                    payload, timeout=self._sup.request_timeout)
            # Record the engine's own rate before yielding, so the host opens the
            # device at 44100 rather than its 24000 default.
            self.stream_sample_rate = int(rate)
            # Iterate explicitly rather than `yield from`: the host barges in by
            # breaking out of its loop, which never calls close() on us. An
            # explicit loop means that break propagates as GeneratorExit HERE, so
            # the `finally` below closes the HTTP response — and a closed response
            # is what makes the engine stop decoding. `yield from` would hand the
            # abandonment to the inner generator without this frame ever learning
            # about it.
            for chunk in chunks:
                yield chunk
        finally:
            if chunks is not None:
                try:
                    chunks.close()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    pass
            self._sup.mark_warm()

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                   model: Optional[str] = None, speed: Optional[float] = None,
                   format: str = "mp3", **extra: Any) -> str:
        if not text or not text.strip():
            raise ValueError("zonos3-tts: text is required")
        out = Path(output_path).with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)

        # The engine caps generation at max_tokens frames (2000 on this release,
        # 15 fps => ~133 s) and does NOT report hitting it: at 522+ words the
        # returned WAV simply stops, and past ~750 words the server RESETS the
        # connection and dies. Both measured. So long text is split at sentence
        # boundaries and concatenated — every second the caller asked for is
        # spoken, one cold engine start, and no crash.
        pieces = _split_for_engine(text, self._char_budget())
        if len(pieces) > 1:
            logger.info("zonos3-tts: %d chars exceeds one engine take; speaking "
                        "%d chunks", len(text), len(pieces))

        # Resolve the voice ONCE for the whole utterance. Each chunk otherwise
        # re-resolves the id (a speakers round trip) and, for a clone, re-runs
        # ffmpeg — which for a 5-chunk reply is 5 clip encodes for identical bytes.
        speaker = None
        clip_b64 = None
        clip_voice = (voice or "").strip()
        if clip_voice and Path(clip_voice).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            if not self._sup.cloning_supported:
                raise RuntimeError(
                    "zonos3-tts: cloning from an audio clip needs zac-encoder.gguf and "
                    "ssl-teacher.gguf in the model directory (see the README)")
            clip_b64 = self._encode_clip(Path(clip_voice))
        else:
            speaker = self._sup.resolve_voice(voice)

        audio = _merge_wav([
            self._synthesize_one(piece, voice, speed, extra,
                                 speaker=speaker, clip_b64=clip_b64)
            for piece in pieces
        ])
        out.write_bytes(audio)
        logger.info("zonos3-tts: %d bytes (peak %d) -> %s",
                    len(audio), wav_peak(audio), out)
        return str(out)

    def _char_budget(self) -> int:
        """Characters that safely fit one engine take.

        Derived from the measured cap rather than guessed: the engine stops at
        2000 frames (133 s) and crashes a few hundred words past that. Speech runs
        ~14 characters per second at this voice's rate, so 110 s of headroom is
        ~1540 chars; the floor keeps a small cap from producing one-word chunks.
        """
        frames = self.max_frames
        if frames and frames > 0:
            seconds = (frames / 15.0) * 0.82   # 18% headroom under the hard stop
            return max(200, int(seconds * 14))
        return 1400

    def _synthesize_one(self, text: str, voice: Optional[str],
                        speed: Optional[float], extra: dict,
                        speaker: Optional[str] = None,
                        clip_b64: Optional[str] = None) -> bytes:
        """One engine request, with the silent-WAV retry ladder."""
        payload = self._payload(text, voice, speed, extra,
                                speaker=speaker, clip_b64=clip_b64)
        # A first clone makes the engine encode the reference (measured ~6.6 s on
        # GPU, minutes on CPU), so it gets the load ceiling rather than the
        # ordinary request timeout — otherwise the very first cloned utterance
        # dies at the socket and reads as a broken integration. Detected from the
        # payload rather than a sentinel key: the engine's request objects are
        # closed and reject unknown fields with a 422.
        first_clone = payload.get("speaker_mode") == "reference"
        op_timeout = self._sup.start_timeout if first_clone else self._sup.request_timeout

        # A silent WAV is a successful response with a valid header, so it has to
        # be caught from the samples and re-sampled with a fresh seed. The native
        # engine measured 0/12 silent, but "read-aloud plays nothing" is exactly
        # the symptom this guards and a retry costs far less than dead air.
        attempts = self._sup.retry_attempts
        audio = b""
        last_error = ""
        for attempt in range(attempts):
            attempt_payload = payload
            if attempt:
                attempt_payload = {**payload,
                                   "seed": (int(time.time() * 1000) + attempt * 7919) % 2147483647}
            try:
                audio = self._sup.synthesize(attempt_payload,
                                             timeout=op_timeout)
            except Exception as exc:  # noqa: BLE001 - dispatcher builds the envelope
                last_error = str(exc)
                if attempt + 1 >= attempts:
                    with self._sup._lock:
                        self._sup.state = "error"
                        self._sup.last_error = last_error
                    raise RuntimeError(
                        f"ZONOS3-Mini synthesis failed: {last_error}") from exc
                time.sleep(0.15 * (attempt + 1))
                continue
            if not wav_is_silent(audio):
                break
            last_error = "engine returned silent audio"
            logger.warning("zonos3-tts: silent audio on attempt %d/%d; retrying",
                           attempt + 1, attempts)
        else:
            with self._sup._lock:
                self._sup.last_error = last_error
            raise RuntimeError(
                f"ZONOS3-Mini returned silent audio after {attempts} attempts")

        self._sup.mark_warm()
        return audio


# --------------------------------------------------------------------------- settings


def _setting(ctx, key: str, default: Any = None) -> Any:
    try:
        value = ctx.get_config(key, default)
    except Exception:  # noqa: BLE001 - older hosts / missing key
        value = default
    return default if value is None else value


def _float_setting(ctx, key: str, default: float) -> float:
    try:
        return float(_setting(ctx, key, default))
    except (TypeError, ValueError):
        return default


def _bool_setting(ctx, key: str, default: bool) -> bool:
    value = _setting(ctx, key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _live_setting_fn(ctx):
    """Re-read settings with a short cache.

    ``ctx.get_config`` re-parses the whole config.yaml per call (~12 ms on a
    large file) and a release reads two settings, so a 2 s memo collapses one
    release's reads while still letting a chip change land promptly.
    """
    cache: dict[str, tuple[float, float]] = {}
    lock = threading.Lock()
    ttl = 2.0

    def _read(key: str, fallback: float) -> float:
        now = time.time()
        with lock:
            hit = cache.get(key)
            if hit and (now - hit[1]) < ttl:
                return hit[0]
        value = _float_setting(ctx, key, float(fallback))
        with lock:
            cache[key] = (value, now)
        return value
    return _read


# --------------------------------------------------------------------------- registration


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent


def _default_runtime_dir() -> Path:
    """Where the built zonos3-server lives: the plugin's own runtime/, else the checkout."""
    name = "zonos3-server.exe" if os.name == "nt" else "zonos3-server"
    for candidate in (_plugin_root() / "runtime",
                      Path(r"C:\Users\MaJor\Github\ZONOS3-Mini\servers\ggml\build-cuda")):
        if (candidate / name).is_file():
            return candidate
    return _plugin_root() / "runtime"


def register(ctx) -> None:  # noqa: ANN001 - host-provided plugin context
    runtime_dir = Path(str(_setting(ctx, "runtime_dir", "") or _default_runtime_dir()))
    model_dir = Path(str(_setting(ctx, "model_dir", "") or
                        (Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
                         / "models" / "zonos3-mini")))
    server_exe = _setting(ctx, "server_exe", "") or None
    options = _setting(ctx, "request_options", None)
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            logger.warning("zonos3-tts: settings.request_options is not valid JSON; ignoring")
            options = None

    sup = EngineSupervisor(
        runtime_dir=runtime_dir,
        model_dir=model_dir,
        host=str(_setting(ctx, "host", _DEFAULT_HOST) or _DEFAULT_HOST),
        port=int(_setting(ctx, "port", _DEFAULT_PORT) or _DEFAULT_PORT),
        base_url=str(_setting(ctx, "base_url", "") or ""),
        gpu=_bool_setting(ctx, "gpu", True),
        unload_after=_float_setting(ctx, "unload_after_seconds", _DEFAULT_UNLOAD_AFTER),
        start_timeout=_float_setting(ctx, "start_timeout_seconds", _DEFAULT_START_TIMEOUT),
        request_timeout=_float_setting(ctx, "request_timeout_seconds", _DEFAULT_REQUEST_TIMEOUT),
        retry_attempts=int(_setting(ctx, "retry_attempts", _DEFAULT_RETRY_ATTEMPTS) or 3),
        default_voice=str(_setting(ctx, "voice", "") or ""),
        settings_fn=_live_setting_fn(ctx),
        server_exe=Path(str(server_exe)) if server_exe else None,
    )

    provider = Zonos3TTS(
        sup,
        options=options if isinstance(options, dict) else None,
        emotion_aware=_bool_setting(ctx, "emotion_aware", True),
        language=str(_setting(ctx, "language", "en_us") or "en_us"),
        dynamic_range=_float_setting(ctx, "dynamic_range", _DEFAULT_DYNAMIC_RANGE),
        expressive=_float_setting(ctx, "expressive", _DEFAULT_EXPRESSIVE),
        volume_boost=_float_setting(ctx, "volume_boost", _DEFAULT_VOLUME_BOOST),
        streams_pcm=_bool_setting(ctx, "streaming", True),
        steer_hold=_float_setting(ctx, "steer_hold", _DEFAULT_STEER_HOLD),
        steer_onset=_float_setting(ctx, "steer_onset_seconds", _DEFAULT_STEER_ONSET),
        steer_ramp=_float_setting(ctx, "steer_ramp_seconds", _DEFAULT_STEER_RAMP),
        headroom_db=(_float_setting(ctx, "headroom_db", -1.0)
                     if _setting(ctx, "headroom_db", None) is not None else None),
    )
    ctx.register_tts_provider(provider)

    try:
        ctx._zonos3_state = sup
        ctx._zonos3_provider = provider
    except Exception:  # noqa: BLE001 - host context may be slotted
        pass

    # -- emotion pipeline --------------------------------------------------
    # Registered before the publisher so the state file can report it.
    pipeline = None
    detector = None
    if SenseVoiceEmotion is not None and emotion_mod is not None:
        try:
            configured_dir = str(_setting(ctx, "sensevoice_model_dir", "")
                                 or os.environ.get("SENSEVOICE_ONNX_DIR", "") or "")
            detector = SenseVoiceEmotion(
                Path(configured_dir) if configured_dir else None,
                enabled=_bool_setting(ctx, "emotion_stt", True),
                engine=str(_setting(ctx, "sensevoice_engine", "onnx") or "onnx"),
                threads=int(_setting(ctx, "sensevoice_threads", 2) or 2),
            )
            pipeline = EmotionPipeline(
                detector, provider,
                enabled=_bool_setting(ctx, "emotion_stt", True),
                annotate=_bool_setting(ctx, "emotion_annotate", True),
                prosody=_bool_setting(ctx, "emotion_reply_prosody", True),
                auto=_bool_setting(ctx, "emotion_auto", True),
            )
            ctx.register_hook("pre_llm_call", pipeline.on_pre_llm_call)
            # pre_transcription fires with the recorded file BEFORE any STT
            # backend runs, which is the one hook that reaches every voice
            # surface (desktop mic, CLI voice mode, gateway) without patching a
            # host call site. It analyses the same audio in parallel and stores
            # the affect; the hook's own return value is ignored here because
            # SenseVoice already produced the transcript we report.
            if _bool_setting(ctx, "emotion_auto", True):
                ctx.register_hook("pre_transcription", pipeline.on_pre_transcription)
            try:
                ctx._zonos3_emotion = pipeline
            except Exception:  # noqa: BLE001
                pass
            _register_emotion_tools(ctx, pipeline)
            state = "available" if detector.is_available() else "weights missing"
            logger.info("zonos3-tts: emotion pipeline on (SenseVoice %s, auto=%s)",
                        state, _bool_setting(ctx, "emotion_auto", True))
        except Exception as exc:  # noqa: BLE001 - voice still works without affect
            logger.warning("zonos3-tts: emotion pipeline unavailable: %s", exc)

    # -- SenseVoice as the STT provider ------------------------------------
    # Optional, and off by default: the host may already be pointed at another
    # provider (local whisper, a cloud API, …). Turning this on makes one
    # SenseVoice pass produce the transcript *and* the affect, so the emotion
    # hook does not have to decode the same audio a second time.
    stt_provider = None
    if detector is not None and _bool_setting(ctx, "stt_enabled", False):
        try:
            stt_provider = sensevoice_mod.make_stt_provider(
                detector,
                on_result=(pipeline.observe_result if pipeline is not None else None),
                display_name="SenseVoice (local)",
            )
            ctx.register_transcription_provider(stt_provider)
            logger.info("zonos3-tts: sensevoice STT provider registered (engine=%s)",
                        detector.engine)
        except Exception as exc:  # noqa: BLE001 - TTS must still work
            logger.warning("zonos3-tts: could not register the sensevoice STT provider: %s", exc)
            stt_provider = None
    try:
        ctx._zonos3_stt = stt_provider
    except Exception:  # noqa: BLE001
        pass

    # -- audio ducking -----------------------------------------------------
    # Registered before the publisher so the state file can report it.
    # Triggered by watching Hermes' own WASAPI session state, not a Python hook:
    # the desktop app plays TTS in its Electron renderer, so the host's
    # playback helpers never run there.
    ducker = None
    if AudioDucker is not None and _bool_setting(ctx, "ducking_enabled", True):
        try:
            ducker = AudioDucker(
                factor=_float_setting(ctx, "ducking_factor", 0.20),
                watch=tuple(
                    w.strip().lower()
                    for w in str(_setting(ctx, "ducking_watch", "hermes.exe") or "hermes.exe").split(",")
                    if w.strip()),
                hold_off=_float_setting(ctx, "ducking_hold_off_seconds", 1.2),
                enabled=True,
            )
            if not ducker.start():
                ducker = None
        except Exception as exc:  # noqa: BLE001 - ducking is optional polish
            logger.warning("zonos3-tts: audio ducking unavailable: %s", exc)
            ducker = None
    try:
        ctx._zonos3_ducker = ducker
    except Exception:  # noqa: BLE001
        pass

    publisher = _StatePublisher(sup, provider, pipeline, ducker)
    sup._publisher = publisher.publish
    publisher.publish(sup)

    stop = threading.Event()

    def _heartbeat() -> None:
        # Two cadences: a fast tick that only looks for a manual-unload request
        # (cheap — one stat/rename), and the slower full republish. A chip
        # button that takes 20 s to act reads as broken, so the fast tick keeps
        # the visible latency around a second without re-serialising state.
        tick = 0
        while not stop.wait(1.0):
            tick += 1
            try:
                if sup.poll_unload_request():
                    publisher.publish(sup)
                    tick = 0
            except Exception:  # noqa: BLE001 - telemetry loop must never die
                logger.debug("zonos3-tts: unload poll failed", exc_info=True)
            if tick >= 20:
                tick = 0
                publisher.publish(sup)

    threading.Thread(target=_heartbeat, name="zonos3-state-heartbeat", daemon=True).start()

    def _shutdown() -> None:
        stop.set()
        sup.client.close()
        # Ducking is a global side effect on other applications; never leave it
        # applied because a plugin unloaded mid-utterance.
        try:
            if ducker is not None:
                ducker.stop()
        except Exception:  # noqa: BLE001
            pass
        sup.stop_process()

    try:
        ctx.on_unload(_shutdown)
    except Exception:  # noqa: BLE001 - older hosts may not expose the hook
        pass

    if not provider.is_available():
        logger.warning(
            "zonos3-tts: server binary or weights missing (exe=%s model=%s). "
            "See the plugin README.", sup.exe, model_dir)


class _StatePublisher:
    """Publish residency facts to a JSON file the dashboard chip reads.

    Three guards keep a dead agent from rendering as live: a heartbeat rewrite,
    write-then-rename so a poll never reads half a file, and a pid stamp the
    reader can prove against.
    """

    def __init__(self, sup: EngineSupervisor, provider: Zonos3TTS,
                 pipeline: Optional["EmotionPipeline"] = None,
                 ducker: Optional[object] = None):
        self.sup = sup
        self.provider = provider
        self.pipeline = pipeline
        self.ducker = ducker

    def publish(self, _sup: EngineSupervisor) -> None:
        try:
            home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
            out_dir = home / "cache" / "zonos3-tts"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                **self.sup.snapshot(),
                "steering_supported": self.provider.steering_supported,
                "emotion_aware": self.provider._emotion_aware,
                "emotion": self.provider.current_emotion(),
                "emotion_pipeline": self.pipeline.snapshot() if self.pipeline else None,
                "ducking": self.ducker.snapshot() if self.ducker else None,
                # Streaming is decided by the provider, not the engine, so it is
                # stamped here. Without it, "is the raw-PCM path live?" is
                # unanswerable from outside the process that owns the object.
                "streams_pcm": bool(getattr(self.provider, "streams_pcm", False)),
                "stream_sample_rate": int(getattr(self.provider, "stream_sample_rate", 0) or 0),
                "updated_at": time.time(),
                "agent_pid": os.getpid(),
            }
            target = out_dir / f"_engine_state.{os.getpid()}.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, target)
            # Also refresh the unsuffixed file so an older reader (a dashboard
            # process started before this change) sees fresh data instead of a
            # fossil. Two writers collide here, which is exactly why the reader
            # prefers the per-pid files.
            legacy = out_dir / "_engine_state.json"
            try:
                legacy_tmp = legacy.with_suffix(".json.tmp")
                legacy_tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(legacy_tmp, legacy)
            except Exception:  # noqa: BLE001 - the per-pid file is authoritative
                pass
        except Exception as exc:  # noqa: BLE001 - telemetry only
            logger.debug("zonos3-tts: could not publish state: %s", exc)


def _register_emotion_tools(ctx, pipeline: "EmotionPipeline") -> None:  # noqa: ANN001
    """Expose the emotion pipeline as callable tools.

    Two tools, because the pipeline has two halves a caller may want separately:

    * ``sensevoice_emotion`` — transcribe an audio file and report the affect, so
      a voice surface (or the agent itself) can feed a recording in;
    * ``voice_emotion_state`` — read/clear the pending affect, for debugging and
      for surfaces that prefer to compose their own annotation.

    Both are read-mostly and cheap: the detector is CPU/int8 and keeps one
    session.
    """

    def _emotion_tool(audio_path: str = "", clear: bool = False, **_ignored) -> str:
        import json as _json
        if clear:
            pipeline.provider.clear_emotion()
            return _json.dumps({"ok": True, "cleared": True})
        if not audio_path:
            return _json.dumps({"ok": False, "error": "audio_path is required"})
        p = Path(audio_path).expanduser()
        if not p.is_file():
            return _json.dumps({"ok": False, "error": f"no such file: {audio_path}"})
        return _json.dumps(pipeline.observe_audio(p))

    def _state_tool(**_ignored) -> str:
        import json as _json
        return _json.dumps({"ok": True, **pipeline.snapshot()})

    try:
        ctx.register_tool(
            "sensevoice_emotion", "voice",
            {
                "name": "sensevoice_emotion",
                "description": (
                    "Transcribe an audio file with SenseVoice and report the "
                    "speaker's emotion and any audio events. Also arms the reply "
                    "voice so the next spoken answer matches the mood. Pass "
                    "clear=true to reset the affect."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {"type": "string",
                                       "description": "Path to an audio file."},
                        "clear": {"type": "boolean",
                                  "description": "Reset the stored affect instead of analysing."},
                    },
                },
            },
            _emotion_tool, description="SenseVoice emotion + transcript",
            emoji="🎭",
        )
        ctx.register_tool(
            "voice_emotion_state", "voice",
            {"name": "voice_emotion_state",
             "description": "Report the currently pending voice emotion (or none).",
             "parameters": {"type": "object", "properties": {}}},
            _state_tool, description="Pending voice emotion state", emoji="🎭",
        )
    except Exception as exc:  # noqa: BLE001 - tools are a convenience, not the core
        logger.debug("zonos3-tts: could not register emotion tools: %s", exc)


def engine_state(ctx) -> Optional[EngineSupervisor]:  # noqa: ANN001 - dashboard half
    return getattr(ctx, "_zonos3_state", None)


# --------------------------------------------------------------------------- emotion wiring


class EmotionPipeline:
    """Bridges STT affect -> model stance -> reply prosody.

    The two halves of this plugin meet here:

    * :class:`sensevoice.SenseVoiceEmotion` produces ``(text, emotion,
      intensity)`` from an audio file;
    * :class:`Zonos3TTS` consumes the *reply* prosody through ``set_emotion``.

    Between them sits the model's turn, which is where "sympathetic, rational,
    productive" actually happens — that is a prompt problem, so this class
    registers a ``pre_llm_call`` hook that appends a stance instruction to the
    user turn. Doing it as a hook rather than by patching a host call site keeps
    the integration supported across Hermes updates.

    State is per-process and deliberately tiny: the most recent detected emotion,
    decayed by age. Voice turns are seconds apart, so a stale emotion from an
    earlier turn would colour an unrelated typed message — hence the TTL.
    """

    # A detected emotion older than this is not applied to a new turn. Speech and
    # the reply it provokes are seconds apart; anything older is a different
    # interaction.
    TTL_SECONDS = 120.0

    def __init__(self, detector: "object", provider: Zonos3TTS, *, enabled: bool = True,
                 annotate: bool = True, prosody: bool = True, auto: bool = True):
        self.detector = detector
        self.provider = provider
        self.enabled = enabled
        self.annotate = annotate
        self.prosody = prosody
        # auto = run detection from the pre_transcription hook (no manual call)
        self.auto = auto
        self._lock = threading.Lock()
        self._emotion = "neutral"
        self._intensity = 0.0
        self._events: list[str] = []
        self._at = 0.0
        self._turn_consumed = True   # nothing pending until real speech arrives

    # -- STT side ---------------------------------------------------------
    def observe_audio(self, audio_path: str | Path) -> dict:
        """Recognise affect in a just-recorded utterance and arm the reply voice.

        Returns a small dict for logging/dashboard; never raises — a broken
        detector must not fail the turn.
        """
        if not self.enabled or not getattr(self.detector, "is_available", lambda: False)():
            return {"ok": False, "reason": "sensevoice unavailable"}
        return self.observe_result(self.detector.recognise(audio_path))

    def observe_result(self, result) -> dict:
        """Accept an already-computed :class:`EmotionResult`.

        The ``sensevoice`` STT provider transcribes with the same detector, so
        when it is the active provider the affect has *already* been recognised
        by the time the transcript exists. Re-running the model here would be a
        second full decode of the same audio for no new information, so the
        provider hands its result straight to this method instead.
        """
        if not self.enabled:
            return {"ok": False, "reason": "emotion pipeline disabled"}
        if not getattr(result, "ok", False):
            return {"ok": False, "reason": getattr(result, "error", "recognition failed")}
        # Reconcile the per-segment emotion tag with the audio events before
        # anything acts on it: a crying speaker often reads as NEUTRAL, while
        # `<|Cry|>` is unambiguous. See emotion.infer_affect.
        events = list(result.events)
        raw_intensity = result.intensity if result.intensity is not None else 0.0
        detected = getattr(result, "detected", True)
        if emotion_mod is not None and hasattr(emotion_mod, "infer_affect"):
            if detected:
                emotion, intensity, why = emotion_mod.infer_affect(
                    result.emotion, events, raw_intensity)
            else:
                # The model declined to name an affect. Only an event can supply
                # one; a bare EMO_UNKNOWN must not steer the reply either way.
                emotion, intensity, why = emotion_mod.infer_affect(
                    "neutral", events, raw_intensity)
                if emotion == "neutral":
                    why = "no affect detected"
        else:
            # Emotion module unavailable: keep the raw label, don't invent one.
            emotion = result.emotion or "neutral"
            intensity, why = raw_intensity, "emotion only"
        if why != "emotion only" and why != "no affect detected":
            logger.debug("zonos3-tts: affect %s -> %s (%.2f): %s",
                         result.emotion, emotion, intensity, why)
        with self._lock:
            self._emotion = emotion
            self._intensity = intensity
            self._events = events
            self._at = time.time()
            self._turn_consumed = False
        # Arm the reply voice now, so the synthesis that follows this turn
        # already sounds right.
        if self.prosody:
            self._apply_reply_voice()
        return {"ok": True, "emotion": emotion, "intensity": intensity or None,
                "events": events, "text": result.text, "affect_source": why}

    def _apply_reply_voice(self) -> None:
        controls = emotion_mod.reply_prosody(self._emotion, self._intensity,
                                            enabled=self.prosody)
        if controls:
            self.provider.set_emotion_controls(controls)
        else:
            self.provider.clear_emotion()

    def _stt_provider_selected(self) -> bool:
        """True when ``stt.provider`` is this plugin's own ``sensevoice`` provider.

        Read from the host config on every call (it is a dict lookup on an already
        parsed config), so switching providers takes effect on the next turn
        instead of only after a restart. Any failure reads as "not selected",
        which falls back to the hook doing its own decode — the safe default.
        """
        try:
            from tools.transcription_tools import _load_stt_config, _get_provider
            return str(_get_provider(_load_stt_config()) or "").strip().lower() == "sensevoice"
        except Exception:  # noqa: BLE001 - host internals, fail open
            return False

    # -- LLM side ---------------------------------------------------------
    def on_pre_transcription(self, **kwargs) -> Optional[dict]:
        """``pre_transcription`` hook: analyse the recorded audio for affect.

        This is what makes emotion automatic. The hook fires before any STT
        backend runs, so it sees the exact file that will be transcribed — no
        extra service in the loop, no patched host call site, and it works for
        the desktop mic, CLI voice mode and the gateway alike.

        Returns ``None`` deliberately: this hook exists to *observe*, and the
        configured STT provider still produces the transcript the user reads.
        Returning a dict here would only override prompt/language/model.

        The analysis runs on a daemon thread. It takes a few hundred ms, and the
        user is waiting on their transcript — blocking the STT dispatch to detect
        a mood would be the wrong trade. The trade-off is that a reply could in
        principle be requested before the analysis lands; the window is small
        (detection ~0.3 s for several seconds of audio) and the failure mode is
        merely "this turn had no affect tag", never a broken turn.
        """
        if not self.enabled or not self.auto:
            return None
        # When the sensevoice STT provider is the *configured* provider it has
        # already recognised this exact file (transcript *and* affect) and
        # reported through observe_result, so decoding it again here would
        # double the CPU for the same answer. Checked live rather than latched at
        # registration: merely registering the provider while `stt.provider`
        # still names something else must keep this hook working, or the emotion
        # feature would silently stop the moment the provider was enabled.
        if self._stt_provider_selected():
            return None
        file_path = kwargs.get("file_path")
        if not file_path:
            return None
        path = Path(str(file_path))
        if not path.is_file():
            return None

        def _run() -> None:
            try:
                self.observe_audio(path)
            except Exception as exc:  # noqa: BLE001 - affect must never break STT
                logger.debug("zonos3-tts: automatic emotion pass failed: %s", exc)

        threading.Thread(target=_run, name="zonos3-emotion-auto", daemon=True).start()
        return None

    def on_pre_llm_call(self, **kwargs) -> Optional[dict]:
        """``pre_llm_call`` hook: append the response stance for this turn.

        Returns ``{"context": ...}`` which the host appends to the *user message*
        (never the system prompt), so it is turn-local and cannot leak into later
        turns through the system prompt.
        """
        if not self.annotate or not self.enabled:
            return None
        with self._lock:
            emotion, intensity = self._emotion, self._intensity
            events = list(self._events)
            age = time.time() - self._at
            pending = not self._turn_consumed
            self._turn_consumed = True
        if not pending or age > self.TTL_SECONDS:
            return None
        guidance = emotion_mod.response_guidance(emotion, intensity)
        if not guidance:
            return None
        line = f"[heard: {emotion} {intensity:.1f}"
        if events:
            interesting = [e for e in events
                           if e.lower() not in {"speech", "speech_noise", "event_unk", "bgm"}]
            if interesting:
                line += " " + ",".join(interesting)
        line += f"] {guidance}"
        return {"context": line}

    # -- introspection ----------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {"emotion": self._emotion, "intensity": self._intensity,
                    "events": list(self._events),
                    "age_seconds": round(time.time() - self._at, 1) if self._at else None,
                    "pending": not self._turn_consumed,
                    "detector_available": bool(getattr(
                        self.detector, "is_available", lambda: False)())}
