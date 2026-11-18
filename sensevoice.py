"""SenseVoice-small speech recognition for the ZONOS3 voice loop.

Runs the **int8 ONNX export already on this machine** (``~/.dsh/speech-to-text/
sensevoice/models/sensevoice-onnx``), so nothing is downloaded and no PyTorch
checkpoint is loaded. Two backends drive the same two files
(``model.int8.onnx`` + ``tokens.txt``):

``onnx`` (default)
    A bare ``onnxruntime`` session plus the Kaldi-style frontend reproduced in
    NumPy below (80 mel, LFR 7/6). No extra dependency at all — ``onnxruntime``
    is already a Hermes dependency — and it is the only backend that exposes the
    CTC **posteriors**, i.e. the margin-based ``intensity`` dial. Measured
    end-to-end here (decode + infer): 354 ms for a 3 s clip, 537 ms for 5 s.

``sherpa``
    sherpa-onnx, a small C++ runtime with the SenseVoice CTC frontend built in
    (``pip install sherpa-onnx``, 28 MB, CPU only). Its feature extractor, thread
    pool and CTC collapse are native, which made it ~3x faster on *inference
    alone* (105 ms vs 414 ms for a 3 s clip); once audio decoding and process
    overhead are included the two engines measure the same, so it is opt-in. It
    reports the emotion label but not the posteriors, so no ``intensity``.

Both backends emit the same :class:`EmotionResult`.

Why this needs no tokenizer
---------------------------
SenseVoice-small is a FunASR SAN-M encoder with a CTC head. The emotion is not a
separate classifier: the model emits **control tokens** inside its own CTC
output — ``<|HAPPY|>``, ``<|SAD|>``, ``<|ANGRY|>``, ``<|NEUTRAL|>``,
``<|FEARFUL|>``, ``<|DISGUSTED|>``, ``<|SURPRISED|>``, ``<|EMO_UNKNOWN|>`` — plus
audio-event tags (``<|Speech|>``, ``<|Laughter|>``, ``<|Applause|>``, ``<|Cry|>``,
``<|Cough|>`` …). They live in the last ~170 rows of ``tokens.txt``. So one
forward pass yields transcript *and* affect: no second model, no extra VRAM.

``text_norm`` is a token id, not a flag
---------------------------------------
The exported graph takes ``language`` and ``text_norm`` as **int32 token ids**
resolved through an embedding table, not booleans. ``text_norm`` must be the id
of ``<|withitn|>`` (14 in this vocabulary) to get inverse text normalisation —
digits, punctuation, capitalisation. The obvious-looking ``1`` selects
``<|woitn|>``, which strips all of that: it returns ``fifty pieces of code``
instead of ``50 pieces of code.``, and drops the sentence-final period in CJK.
Out-of-range values crash inside the graph's Gather (e.g. 16).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SR = 16000


class AudioDecodeError(RuntimeError):
    """Audio could not be turned into samples (empty, truncated, not audio).

    Distinct from a generic RuntimeError so the STT envelope can report a message
    the caller can act on instead of ffmpeg's internal diagnostics.
    """


def _silence_wav(seconds: float = 0.5, rate: int = SR) -> bytes:
    """A minimal mono 16-bit PCM WAV of silence, for warming a backend.

    Built in-process rather than shelled out to ffmpeg, so a preload has no
    external dependency and cannot fail on a host without ffmpeg on PATH.
    """
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * max(0.05, float(seconds))))
    return buf.getvalue()
N_MELS = 80
FRAME_LEN = 0.025
FRAME_SHIFT = 0.010
N_FFT = 512
PREEMPH = 0.97
LFR_M, LFR_N = 7, 6

# Default location of the shared ONNX weights. Deliberately a *reference*, not a
# copy: the point of using this build is to avoid a second 231 MB model on disk.
DEFAULT_MODEL_DIR = Path.home() / ".dsh" / "speech-to-text" / "sensevoice" / "models" / "sensevoice-onnx"

# `<|withitn|>` in this vocabulary (see module docstring): inverse text
# normalisation on. `<|woitn|>` is 1 and silently strips all punctuation/digits.
TEXT_NORM_ON = 14

# SenseVoice language ids for the `language` input. 0 is auto-detect, which is
# what every caller gets unless it passes an explicit hint.
_LANG_IDS = {"auto": 0, "zh": 1, "en": 2, "ja": 3, "ko": 4, "yue": 5}

# SenseVoice emotion labels -> the vocabulary this plugin maps onto ZONOS3.
_EMOTIONS = ("NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED")
_AUDIO_EVENTS = ("Speech", "BGM", "Laughter", "Applause", "Cry", "Sneeze",
                 "Breath", "Cough", "Speech_Noise", "Event_UNK")

# Short alpha control tokens that are normalisation hints, not language tags.
_NON_LANGUAGE_TAGS = {"woitn", "withitn"}

# Decoding threads. Measured on this host (i7-12700K, 12C/20T, int8 ONNX): 2
# threads is the sweet spot — a 3 s clip in 414 ms vs 1028 ms at 1 thread and
# 867 ms at 4. More threads oversubscribe the P/E split and lose badly.
DEFAULT_THREADS = 2


@dataclass
class EmotionResult:
    """One recognition pass: transcript plus affect."""

    text: str = ""
    emotion: str = "neutral"
    # None when the active backend cannot measure it: sherpa reports the label
    # only, while the onnxruntime backend exposes the CTC posteriors behind it.
    intensity: Optional[float] = None
    events: list[str] = field(default_factory=list)
    language: str = ""
    raw_tokens: list[str] = field(default_factory=list)
    error: str = ""
    # False when the model declined to name an affect (`<|EMO_UNKNOWN|>`) or
    # emitted none. FunASR's guidance is not to fold that into NEUTRAL: "no
    # opinion" and "calm" are different evidence, and only the second should
    # steer a reply. `emotion` stays "neutral" so lookups still resolve.
    detected: bool = True

    @property
    def ok(self) -> bool:
        return not self.error

    def annotation(self) -> str:
        """Compact tag for the model, e.g. ``[voice: emotion=angry intensity=0.8 events=cry]``.

        Deliberately terse and machine-shaped: it rides inside the user turn, so
        every token spent here is one the user's own words do not get. Intensity
        is omitted when the backend did not measure it, rather than printing a
        number it does not have.
        """
        if not self.ok or self.emotion in ("neutral", "unknown", ""):
            return ""
        parts = [f"emotion={self.emotion}"]
        if self.intensity is not None:
            parts.append(f"intensity={self.intensity:.1f}")
        if self.events:
            parts.append("events=" + ",".join(self.events))
        return "[voice: " + " ".join(parts) + "]"


# --------------------------------------------------------------------------- frontend


def _mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR,
                    low: float = 20.0, high: Optional[float] = None):
    import numpy as np
    high = high if high is not None else sr / 2

    def hz2mel(f):
        return 1127.0 * np.log1p(np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (np.exp(np.asarray(m) / 1127.0) - 1.0)

    mels = np.linspace(hz2mel(low), hz2mel(high), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre > left:
            fb[i, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            fb[i, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return fb


def _fbank(x):
    """Kaldi-compatible 80-dim log-mel fbank (25 ms / 10 ms, pre-emphasis, Hamming)."""
    import numpy as np
    x = np.append(x[0], x[1:] - PREEMPH * x[:-1])
    frame_len = int(FRAME_LEN * SR)
    shift = int(FRAME_SHIFT * SR)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // shift
    window = np.hamming(frame_len)
    fb = _mel_filterbank()
    # Frame the whole signal once, then one batched FFT instead of a Python loop
    # over frames: identical output, far cheaper on longer clips.
    idx = np.arange(frame_len)[None, :] + shift * np.arange(n_frames)[:, None]
    frames = x[idx] * window
    spec = np.fft.rfft(frames, N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(power @ fb.T + 1e-10)


def _apply_lfr(feat, m: int = LFR_M, n: int = LFR_N):
    """Low Frame Rate stacking + subsampling, the FunASR default."""
    import numpy as np
    pad = (m - 1) // 2
    feat = np.pad(feat, ((pad, pad), (0, 0)), mode="edge")
    starts = list(range(0, feat.shape[0] - m + 1, n))
    out = np.concatenate([feat[i:i + m].reshape(-1) for i in starts])
    return out.reshape(len(starts), m * feat.shape[1])


# --------------------------------------------------------------------------- audio


_FFMPEG_CACHE: list[Optional[str]] = []


def _find_ffmpeg() -> Optional[str]:
    """Resolve ffmpeg once per process.

    ``shutil.which`` walks 679 PATH entries here (~216 ms), so the answer is
    memoised: the binary cannot move between two utterances in the same session,
    and a first call that found nothing is not retried on the next turn (which
    keeps a missing-ffmpeg deployment failing fast instead of stalling).
    """
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE[0]
    found = None
    try:
        from tools.transcription_audio import _find_ffmpeg_binary
        found = _find_ffmpeg_binary()
    except Exception:  # noqa: BLE001 - host helper is a convenience only
        found = None
    if not found:
        found = shutil.which("ffmpeg")
    _FFMPEG_CACHE.append(found)
    return found


def load_audio(path: Path, ffmpeg: Optional[str] = None):
    """``(16000, float32 mono samples)`` for *path*.

    Fast paths first, in cost order:

    1. a mono 16-bit **16 kHz** WAV is read directly (~0.3 ms) — that is what the
       host's own recorder produces;
    2. any WAV **at any rate/channels** is read with ``soundfile`` and resampled in
       NumPy (~3 ms measured vs ~65 ms for an ffmpeg subprocess);
    3. everything else (mp3/ogg/webm/3gp…) goes through ffmpeg.

    About (2): the resample is linear interpolation, which is *not* band-limited
    and would alias a music signal. For 16 kHz speech recognition it measured
    word-identical to ffmpeg across every line tested (numbers, questions, long
    sentences) — the only differences were commas and question marks. It is used
    only for WAV sources, where libsndfile's decoder is exact; compressed
    containers still go to ffmpeg, and ``resample="ffmpeg"`` forces the old path.

    ffmpeg is resolved *after* the fast paths, on purpose: ``shutil.which`` here
    walks 679 PATH entries (~216 ms measured) and would otherwise be paid on
    every utterance that never needs it.
    """
    import numpy as np

    mode = (os.environ.get("ZONOS3_STT_RESAMPLE") or "auto").strip().lower()
    if mode != "ffmpeg":
        try:
            with wave.open(str(path), "rb") as w:
                ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
                if ch == 1 and width == 2 and rate == SR:
                    raw = w.readframes(w.getnframes())
                    if raw:
                        return SR, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        except (wave.Error, EOFError, OSError):
            pass  # not a PCM wav: try soundfile, then ffmpeg

        try:
            import soundfile as sf

            data = None
            src_rate = 0
            with sf.SoundFile(str(path)) as handle:
                if (handle.format or "").upper() in {"WAV", "WAVEX", "W64"}:
                    data = handle.read(dtype="float32", always_2d=True)
                    src_rate = int(handle.samplerate)
        except Exception:  # noqa: BLE001 - any libsndfile problem -> ffmpeg
            data = None
            src_rate = 0
        if data is not None and len(data) and src_rate > 0:
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            if src_rate == SR:
                return SR, mono.astype(np.float32)
            target = int(round(len(mono) * SR / src_rate))
            if target > 0:
                resampled = np.interp(
                    np.linspace(0.0, len(mono) - 1, target),
                    np.arange(len(mono)), mono).astype(np.float32)
                return SR, resampled

    ffmpeg = ffmpeg or _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; SenseVoice needs it to decode this audio")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sv16k.wav"
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        if proc.returncode != 0 or not out.is_file():
            # A bad upload (empty, truncated, or not audio at all) reaches here as
            # an ffmpeg decode failure, and ffmpeg's stderr names a temp file the
            # caller has never heard of. Surface a sentence a person can act on,
            # and keep the raw text in the log for debugging.
            size = path.stat().st_size if path.exists() else 0
            if size == 0:
                raise AudioDecodeError("the recording was empty (0 bytes)")
            if size < 128:
                raise AudioDecodeError(
                    f"the recording was too short to contain audio ({size} bytes)")
            detail = (proc.stderr or "").strip().splitlines()
            first = detail[0] if detail else "unknown decoder error"
            logger.info("sensevoice: ffmpeg could not decode %s (%d bytes): %s",
                        path.name, size, " | ".join(detail[:3]))
            raise AudioDecodeError(
                "the upload could not be decoded as audio — it was empty, "
                f"truncated, or not an audio file ({size} bytes)")
        with wave.open(str(out), "rb") as w:
            raw = w.readframes(w.getnframes())
    if not raw:
        raise AudioDecodeError("the recording decoded to silence (no samples)")
    return SR, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- token parsing


def _load_vocab(tokens_path: Path) -> dict[int, str]:
    vocab: dict[int, str] = {}
    for line in tokens_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        token, _, idx = line.rpartition(" ")
        if token and idx.lstrip("-").isdigit():
            vocab[int(idx)] = token
    return vocab


def _is_control(token: str) -> bool:
    return token.startswith("<|") and token.endswith("|>")


def _clean_tag(tag: str) -> str:
    """``<|NEUTRAL|>`` -> ``NEUTRAL``; anything malformed -> ``""``."""
    tag = (tag or "").strip()
    return tag[2:-2] if _is_control(tag) else ""


def _result_from_tokens(text: str, *, emotion: str = "", event: str = "", language: str = "",
                        intensity: Optional[float] = None,
                        raw_tokens: Optional[list[str]] = None) -> EmotionResult:
    """Assemble an :class:`EmotionResult` from raw CTC tokens or sherpa's parsed fields.

    Shared by both backends so the label mapping is written once: sherpa hands
    over already-split ``emotion``/``event``/``lang``, while the onnxruntime path
    only has *raw_tokens* and this has to pull the control tokens out of them.
    """
    events: list[str] = []
    lang = language if language in _LANG_IDS and language != "auto" else ""
    emo = emotion.lower() if emotion else ""

    def take(name: str) -> None:
        nonlocal emo, lang
        if name in _EMOTIONS:
            if not emo:
                emo = name.lower()
        elif name in _AUDIO_EVENTS:
            if name.lower() not in {e.lower() for e in events}:
                events.append(name)
        elif len(name) <= 6 and name.isalpha() and name not in _NON_LANGUAGE_TAGS:
            if not lang:
                lang = name

    if event:
        take(event)
    for tok in (raw_tokens or []):
        if _is_control(tok):
            take(tok[2:-2])

    # SentencePiece marks word starts with "▁"; a joiner is what human-facing
    # text wants in every language this model covers.
    cleaned = (text or "").replace("▁", " ").strip()
    # EMO_UNKNOWN is the model declining to name an affect — it is NOT neutral and
    # NOT a low-confidence neutral. FunASR's own guidance is explicit: "Send
    # unknown or unparseable results for inspection instead of silently turning
    # them into NEUTRAL", and it warns that `ban_emo_unk=True` only masks the
    # token's decoding score without providing new evidence. Collapsing it to
    # `neutral` here made "the model has no opinion" indistinguishable from "the
    # model says calm", so a caller could not tell a firm neutral from a shrug.
    # `detected=False` carries that distinction; `emotion` stays "neutral" so the
    # prosody table still has an entry.
    unknown = emo == "emo_unknown" or not emo
    return EmotionResult(
        text=cleaned,
        emotion="neutral" if unknown else emo,
        intensity=intensity,
        events=events,
        language=lang,
        raw_tokens=list(raw_tokens or []),
        detected=not unknown,
    )


# --------------------------------------------------------------------------- backends


class _SherpaBackend:
    """sherpa-onnx SenseVoice recogniser — native frontend, fastest path."""

    name = "sherpa"

    def __init__(self, model_path: Path, tokens_path: Path, *, threads: int = DEFAULT_THREADS,
                 use_itn: bool = True):
        import sherpa_onnx
        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_path), tokens=str(tokens_path),
            num_threads=max(1, int(threads)), use_itn=bool(use_itn),
            debug=False, language="auto")
        self._lock = threading.Lock()

    def recognise(self, path: Path, *, language: int = 0, ffmpeg: Optional[str] = None) -> EmotionResult:
        rate, samples = load_audio(path, ffmpeg)
        with self._lock:  # one recognizer, one utterance at a time
            stream = self._rec.create_stream()
            stream.accept_waveform(rate, samples)
            self._rec.decode_stream(stream)
            result = stream.result
        return _result_from_tokens(
            getattr(result, "text", "") or "",
            emotion=_clean_tag(getattr(result, "emotion", "")),
            event=_clean_tag(getattr(result, "event", "")),
            language=_clean_tag(getattr(result, "lang", "")),
            intensity=None,  # sherpa does not surface the CTC posteriors
            raw_tokens=list(getattr(result, "tokens", []) or []),
        )


class _OnnxBackend:
    """Bare onnxruntime session + the NumPy frontend. Exposes CTC posteriors."""

    name = "onnx"

    def __init__(self, model_path: Path, tokens_path: Path, *, threads: int = DEFAULT_THREADS,
                 use_itn: bool = True):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        # Utterances are short and arrive serially; a wide pool both steals cores
        # from the ZONOS3 backbone and, measured, decodes slower anyway.
        opts.intra_op_num_threads = max(1, int(threads))
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.text_norm = TEXT_NORM_ON if use_itn else 1
        self._vocab = _load_vocab(tokens_path)
        self._lock = threading.Lock()

    def recognise(self, path: Path, *, language: int = 0, ffmpeg: Optional[str] = None) -> EmotionResult:
        import numpy as np
        rate, samples = load_audio(path, ffmpeg)
        if rate != SR:  # the ORT graph is fixed at 16 kHz
            raise RuntimeError(f"expected {SR} Hz audio, got {rate} Hz")
        if samples.size == 0:
            raise RuntimeError("audio decoded to zero samples")
        feat = _apply_lfr(_fbank(samples))
        with self._lock:
            logits = self._session.run(None, {
                "x": feat[None].astype(np.float32),
                "x_length": np.array([feat.shape[0]], dtype=np.int32),
                "language": np.array([int(language)], dtype=np.int32),
                "text_norm": np.array([self.text_norm], dtype=np.int32),
            })[0][0]

        vocab = self._vocab
        # Greedy CTC collapse. Id 0 is ``<unk>`` and doubles as the blank this
        # export was trained with; the 25055-wide logits have no extra blank row.
        # (The previous version compared against max(vocab)+1 = 25055, i.e. one
        # past the end, so it could never have skipped a real blank id.)
        prev, ids = -1, []
        for row in logits.argmax(-1):
            i = int(row)
            if i != prev and i != 0:
                ids.append(i)
            prev = i
        tokens = [vocab.get(i, "") for i in ids]

        parsed = _result_from_tokens(
            "".join(t for t in tokens if not _is_control(t)),
            intensity=None, raw_tokens=tokens)

        if parsed.emotion not in ("neutral", "unknown"):
            idx = next((k for k, v in vocab.items() if v == f"<|{parsed.emotion.upper()}|>"), None)
            if idx is not None:
                probs = _row_softmax(logits, idx)
                if probs:
                    # Calibrate against measured posteriors rather than guessing
                    # a divisor. On this host the winning emotion token's
                    # probability ran 0.49-0.93 across real voice samples, so a
                    # fixed /0.35 pinned everything at 1.0 and the dial was
                    # useless. Map 0.45..0.95 onto 0..1 instead: below 0.45 is a
                    # marginal call, at/above 0.95 a decisive one.
                    parsed.intensity = round(min(1.0, max(0.0, (probs[0] - 0.45) / 0.50)), 2)
        return parsed


def _row_softmax(logits, token_id: int) -> list[float]:
    """Probability of *token_id* at each frame where it was the argmax, desc.

    Reading the model's own confidence is the only intensity signal the export
    exposes; a separate regressor would mean another model on disk, which is the
    thing this integration exists to avoid.
    """
    import numpy as np
    out = []
    for row in logits:
        if int(row.argmax()) != token_id:
            continue
        z = row - row.max()
        e = np.exp(z)
        out.append(float(e[token_id] / e.sum()))
    out.sort(reverse=True)
    return out


_BACKENDS = {"sherpa": _SherpaBackend, "onnx": _OnnxBackend}


# --------------------------------------------------------------------------- facade


class SenseVoiceEmotion:
    """Lazy-loaded SenseVoice recogniser that returns transcript + emotion.

    One backend instance, created on first use and reused. CPU-only and
    int8-quantised, so it neither competes with ZONOS3 for VRAM nor keeps the GPU
    busy between utterances; a short clip transcribes in ~0.1-0.2 s here.

    ``engine`` picks the runtime:

    * ``onnx`` (default) — bare onnxruntime. Zero extra dependencies, and the
      only backend that reports the CTC posteriors behind the emotion label, so
      it is the one that can drive ``intensity``. Measured end-to-end (ffmpeg
      decode + infer) on this host: 354 ms for a 3 s clip, 537 ms for 5 s — the
      same as sherpa within noise.
    * ``sherpa`` — sherpa-onnx, if it is installed (``pip install sherpa-onnx``,
      28 MB). Its frontend, thread pool and CTC collapse are native, which made
      it ~3x faster than ORT on *inference alone* (105 ms vs 414 ms for a 3 s
      clip); once decoding and process overhead are included the two are equal,
      so it is opt-in and it cannot supply ``intensity``.
    * ``auto`` — sherpa when importable, else onnx.
    """

    def __init__(self, model_dir: Optional[Path] = None, *, enabled: bool = True,
                 ffmpeg: Optional[str] = None, engine: str = "onnx",
                 threads: int = DEFAULT_THREADS, use_itn: bool = True):
        self.model_dir = Path(model_dir or os.environ.get("SENSEVOICE_ONNX_DIR")
                              or DEFAULT_MODEL_DIR)
        self.enabled = enabled
        self._ffmpeg = ffmpeg
        self._engine = (engine or "onnx").strip().lower()
        self._threads = max(1, int(threads))
        self._use_itn = bool(use_itn)
        self._backend = None
        # True once a successful warm decode has run (distinguishes "session
        # built" from "kernels warmed"; only the latter makes the next real call
        # fast).
        self._warm_decoded = False
        self._lock = threading.Lock()
        self.last_error = ""

    # -- paths ------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.int8.onnx"

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / "tokens.txt"

    @property
    def engine(self) -> str:
        """Engine that will be used (resolved without building a session)."""
        if self._engine in _BACKENDS:
            return self._engine
        try:
            import sherpa_onnx  # noqa: F401
            return "sherpa"
        except Exception:  # noqa: BLE001
            return "onnx"

    def is_available(self) -> bool:
        """Cheap presence check: no network, no session creation."""
        if not self.enabled:
            return False
        module = "sherpa_onnx" if self.engine == "sherpa" else "onnxruntime"
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            return False
        return self.model_path.is_file() and self.tokens_path.is_file()

    # -- backend ----------------------------------------------------------
    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            name = self.engine
            cls = _BACKENDS.get(name)
            if cls is None:
                raise RuntimeError(f"unknown SenseVoice engine {name!r}")
            try:
                self._backend = cls(self.model_path, self.tokens_path,
                                    threads=self._threads, use_itn=self._use_itn)
            except Exception:
                if name != "sherpa":
                    raise
                # An unusable sherpa build is still better served by a working
                # recogniser than by no recogniser.
                logger.warning("sensevoice: sherpa backend failed; using onnxruntime",
                               exc_info=True)
                self._backend = _OnnxBackend(self.model_path, self.tokens_path,
                                             threads=self._threads, use_itn=self._use_itn)
        return self._backend

    # -- inference --------------------------------------------------------
    def preload(self, *, delay: float = 0.0) -> None:
        """Build the inference backend in the background.

        Measured on this host: building the onnxruntime session takes ~3.0 s, and
        the first decode after it costs another ~3 s (ORT kernel warm-up), so the
        first utterance in a fresh process takes ~6 s against ~0.65 s warm — a 9x
        penalty that only the FIRST caller pays.

        That is what makes STT look intermittent on the relay. The relay posts
        straight to `/voice/transcribe` and nothing on that path warms the engine:
        the host's own `warm_stt_provider` returns `action: "noop"` for every
        provider outside `{local, local_command}`, so a plugin STT is never
        pre-loaded, and the relay never calls the `audio/stt-lease` endpoint at
        all. A client that gives up after a few seconds reports "STT didn't work"
        on exactly the first utterance after the gateway has been idle.

        So warm it here, unprompted. Daemon thread, best-effort, never raises: a
        failed preload must not affect a later real request.
        """
        def _go() -> None:
            if delay:
                time.sleep(delay)
            try:
                if not self.is_available():
                    return
                backend = self._get_backend()
                # Session creation alone leaves the first real decode slow (ORT
                # builds kernels per input shape on first run), so run one silent
                # decode through the same frontend. 0.5 s of zeros is enough to
                # take the shape path without producing a transcript.
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                        warm_path = fh.name
                        fh.write(_silence_wav(seconds=0.5))
                    try:
                        backend.recognise(Path(warm_path), language=0, ffmpeg=None)
                        self._warm_decoded = True
                    finally:
                        try:
                            os.unlink(warm_path)
                        except OSError:
                            pass
                except Exception:  # noqa: BLE001 - a partial warm still helps
                    logger.debug("sensevoice: warm decode skipped", exc_info=True)
                logger.debug("sensevoice: backend preloaded (%s)", type(backend).__name__)
            except Exception:  # noqa: BLE001 - warming is an optimisation
                logger.debug("sensevoice: preload failed", exc_info=True)

        thread = threading.Thread(target=_go, name="sensevoice-preload", daemon=True)
        thread.start()

    def recognise(self, audio_path: str | Path, *, language: int = 0,
                  text_norm: bool = True) -> EmotionResult:
        """Transcribe and detect the speaker's emotion in one pass.

        ``intensity`` is populated only by the ``onnx`` engine, where it is a
        heuristic in ``0..1`` derived from how decisively the CTC head picked the
        emotion token (softmax margin at the frames where that token won). It is
        a useful ordering — clearly emotional speech scores higher than flat
        speech — but it is not a calibrated annotation, so downstream code treats
        it as a dial, not a measurement.
        """
        if not self.is_available():
            return EmotionResult(error="SenseVoice ONNX model or runtime unavailable")
        try:
            backend = self._get_backend()
            if isinstance(backend, _OnnxBackend):
                backend.text_norm = TEXT_NORM_ON if text_norm else 1
            return backend.recognise(Path(audio_path), language=language, ffmpeg=self._ffmpeg)
        except Exception as exc:  # noqa: BLE001 - never break the turn
            self.last_error = str(exc)
            logger.warning("sensevoice: recognition failed: %s", exc)
            return EmotionResult(error=str(exc))

    # -- introspection ----------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {"engine": self.engine, "available": self.is_available(),
                "model_dir": str(self.model_dir), "threads": self._threads,
                "itn": self._use_itn, "last_error": self.last_error}


# --------------------------------------------------------------------------- STT provider


def _language_id(language: Optional[str]) -> int:
    """BCP-47 / informal hint -> SenseVoice language id (0 = auto)."""
    if not language:
        return 0
    key = str(language).strip().lower().replace("_", "-").split("-")[0]
    return _LANG_IDS.get(key, 0)


def make_stt_provider(detector: SenseVoiceEmotion, *, on_result=None,
                      display_name: str = "SenseVoice"):
    """Build a host ``TranscriptionProvider`` around an existing detector.

    Imported lazily so this module stays importable outside a Hermes process
    (tests, benchmarks, dashboards) where ``agent`` is not on the path.
    """
    from agent.transcription_provider import TranscriptionProvider

    label = display_name

    class SenseVoiceSTT(TranscriptionProvider):
        """Hermes speech-to-text backed by the local SenseVoice int8 ONNX model.

        One pass produces both the transcript the user sees and the affect the
        reply voice uses, which is why selecting this provider makes the
        ``pre_transcription`` hook's own decode redundant: :meth:`transcribe`
        reports straight to the pipeline via *on_result*.
        """

        name = "sensevoice"

        @property
        def display_name(self) -> str:
            return label

        def __init__(self, det: SenseVoiceEmotion):
            self._det = det

        def is_available(self) -> bool:
            try:
                return bool(self._det.is_available())
            except Exception:  # noqa: BLE001
                return False

        def warm(self) -> None:
            """Mic opened — warm the backend so the first utterance is fast.

            Uses the detector THIS provider was built with, not a fresh instance:
            measured here, warming a separate `SenseVoiceEmotion` had zero effect
            because the provider holds its own object and re-built the ORT
            session anyway (6.9 s cold vs 7.1 s "preloaded").
            """
            self._det.preload()

        def transcribe(self, file_path: str, *, model: Optional[str] = None,
                       language: Optional[str] = None, **extra: Any) -> dict[str, Any]:
            result = self._det.recognise(file_path, language=_language_id(language))
            if not result.ok:
                return {"success": False, "transcript": "", "provider": self.name,
                        "error": result.error}
            if on_result is not None:
                try:
                    on_result(result)
                except Exception:  # noqa: BLE001 - affect must never break STT
                    logger.debug("sensevoice: reporting affect failed", exc_info=True)
            return {"success": True, "transcript": result.text, "provider": self.name,
                    "emotion": result.emotion, "events": list(result.events),
                    "language": result.language,
                    # `emotion` is "neutral" both for a confident neutral and for
                    # "the model declined" — this is the only way a consumer can
                    # tell them apart without re-parsing raw tokens.
                    "emotion_detected": result.detected}

    return SenseVoiceSTT(detector)
