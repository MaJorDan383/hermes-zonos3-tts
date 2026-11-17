"""SenseVoice-small emotion recognition for the ZONOS3 voice loop.

Runs the **int8 ONNX export already on this machine** (``~/.dsh/speech-to-text/
sensevoice/models/sensevoice-onnx``), so the emotion feature costs no extra
disk: the weights are shared with the copy that is already installed. Nothing is
downloaded, and the 893 MB PyTorch checkpoint in Voxta's model folder is *not*
used — that one needs torch + funasr at runtime, neither of which is needed here.

Why the ONNX path works with what Hermes already bundles
--------------------------------------------------------
SenseVoice-small is a FunASR SAN-M encoder with a CTC head. The exported ONNX
takes the raw 560-dim feature frames (80 mel x LFR 7/6) directly, so no
``kaldi_native_fbank`` is required: the frontend is reproduced in NumPy below.
``onnxruntime`` is already a Hermes dependency, so ``pip_dependencies`` stays
empty.

The emotion is not a separate classifier. SenseVoice emits **control tokens**
inside its own CTC output — ``<|HAPPY|>``, ``<|SAD|>``, ``<|ANGRY|>``,
``<|NEUTRAL|>``, ``<|FEARFUL|>``, ``<|DISGUSTED|>``, ``<|SURPRISED|>``,
``<|EMO_UNKNOWN|>`` — plus audio-event tags (``<|Speech|>``, ``<|Laughter|>``,
``<|Applause|>``, ``<|Cry|>``, ``<|Cough|>`` …). They live in the last ~170 rows
of ``tokens.txt``. So one forward pass yields transcript *and* affect, which is
why this is cheap: no second model, no extra GPU memory.

Emotional intensity is approximated from the CTC **posterior margin** on the
winning emotion token: a confident emotion token (a clean peak in the
distribution) reads as a strongly expressed feeling, a marginal one as a mild
cast. That is a heuristic, not a calibrated score, and the docstring on
:func:`recognise` says so.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SR = 16000
N_MELS = 80
FRAME_LEN = 0.025
FRAME_SHIFT = 0.010
N_FFT = 512
PREEMPH = 0.97
LFR_M, LFR_N = 7, 6

# Default location of the shared ONNX weights. Deliberately a *reference*, not a
# copy: the point of using this build is to avoid a second 231 MB model on disk.
DEFAULT_MODEL_DIR = Path.home() / ".dsh" / "speech-to-text" / "sensevoice" / "models" / "sensevoice-onnx"

# SenseVoice emotion labels -> the vocabulary this plugin maps onto ZONOS3.
_EMOTIONS = ("NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED")
_AUDIO_EVENTS = ("Speech", "BGM", "Laughter", "Applause", "Cry", "Sneeze",
                 "Breath", "Cough", "Speech_Noise", "Event_UNK")

# CTC blank for the exported model. The blank id is the last vocabulary index;
# tokens.txt omits it, so it is derived from the vocab size.
_BLANK_FALLBACK = 0


@dataclass
class EmotionResult:
    """One recognition pass: transcript plus affect."""

    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.0
    events: list[str] = field(default_factory=list)
    language: str = ""
    raw_tokens: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def annotation(self) -> str:
        """Compact tag for the model, e.g. ``[voice: emotion=angry intensity=0.8 events=cry]``.

        Deliberately terse and machine-shaped: it rides inside the user turn, so
        every token spent here is one the user's own words do not get.
        """
        if not self.ok or self.emotion in ("neutral", "unknown", ""):
            return ""
        parts = [f"emotion={self.emotion}", f"intensity={self.intensity:.1f}"]
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
    out = np.empty((n_frames, N_MELS), dtype=np.float32)
    for i in range(n_frames):
        frame = x[i * shift: i * shift + frame_len] * window
        spec = np.fft.rfft(frame, N_FFT)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        out[i] = np.log(power @ fb.T + 1e-10)
    return out


def _apply_lfr(feat, m: int = LFR_M, n: int = LFR_N):
    """Low Frame Rate stacking + subsampling, the FunASR default."""
    import numpy as np
    pad = (m - 1) // 2
    feat = np.pad(feat, ((pad, pad), (0, 0)), mode="edge")
    starts = list(range(0, feat.shape[0] - m + 1, n))
    out = np.concatenate([feat[i:i + m].reshape(-1) for i in starts])
    return out.reshape(len(starts), m * feat.shape[1])


# --------------------------------------------------------------------------- engine


class SenseVoiceEmotion:
    """Lazy-loaded SenseVoice recogniser that returns transcript + emotion.

    One ONNX session, created on first use and reused. The session is
    CPU-only and int8-quantised, so it neither competes with ZONOS3 for VRAM nor
    keeps the GPU busy between utterances; a short clip transcribes in well under
    a second (measured ~0.3 s for 7 s of audio on this host).
    """

    def __init__(self, model_dir: Optional[Path] = None, *, enabled: bool = True,
                 ffmpeg: Optional[str] = None):
        self.model_dir = Path(model_dir or os.environ.get("SENSEVOICE_ONNX_DIR")
                              or DEFAULT_MODEL_DIR)
        self.enabled = enabled
        self._ffmpeg = ffmpeg
        self._session = None
        self._vocab: dict[int, str] = {}
        self._lock = threading.Lock()
        self.last_error = ""

    # -- paths ------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.int8.onnx"

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / "tokens.txt"

    def is_available(self) -> bool:
        """Cheap presence check: no network, no session creation."""
        if not self.enabled:
            return False
        try:
            import onnxruntime  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return self.model_path.is_file() and self.tokens_path.is_file()

    # -- vocab / session --------------------------------------------------
    def _load_vocab(self) -> dict[int, str]:
        if self._vocab:
            return self._vocab
        vocab: dict[int, str] = {}
        for line in self.tokens_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            token, _, idx = line.rpartition(" ")
            if token and idx.lstrip("-").isdigit():
                vocab[int(idx)] = token
        self._vocab = vocab
        return vocab

    def _get_session(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            import onnxruntime as ort
            opts = ort.SessionOptions()
            # One thread per call: utterances are short and arrive serially, and
            # a wide pool would steal cores from the ZONOS3 backbone.
            opts.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts,
                providers=["CPUExecutionProvider"])
        return self._session

    # -- audio ------------------------------------------------------------
    def _find_ffmpeg(self) -> Optional[str]:
        if self._ffmpeg:
            return self._ffmpeg
        try:
            from tools.transcription_audio import _find_ffmpeg_binary
            found = _find_ffmpeg_binary()
            if found:
                self._ffmpeg = found
                return found
        except Exception:  # noqa: BLE001 - host helper is a convenience only
            pass
        self._ffmpeg = shutil.which("ffmpeg")
        return self._ffmpeg

    def _decode_wav(self, path: Path):
        """16 kHz mono float32 samples via ffmpeg (handles any container)."""
        import numpy as np
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found; SenseVoice needs it to decode audio")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sv16k.wav"
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", str(path),
                 "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)],
                capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
            if proc.returncode != 0 or not out.is_file():
                raise RuntimeError(f"ffmpeg could not decode {path.name}: {proc.stderr[:200]}")
            import wave
            with wave.open(str(out), "rb") as w:
                raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    # -- inference --------------------------------------------------------
    def recognise(self, audio_path: str | Path, *, language: int = 0,
                  text_norm: int = 1) -> EmotionResult:
        """Transcribe and detect the speaker's emotion in one pass.

        ``intensity`` is a heuristic in ``0..1`` derived from how decisively the
        CTC head picked the emotion token (softmax margin over the runner-up at
        the frames where that token won). It is a useful ordering — clearly
        emotional speech scores higher than flat speech — but it is not a
        calibrated annotation, so downstream code treats it as a dial, not a
        measurement.
        """
        if not self.is_available():
            return EmotionResult(error="SenseVoice ONNX model or onnxruntime unavailable")
        try:
            import numpy as np
            samples = self._decode_wav(Path(audio_path))
            if samples.size == 0:
                return EmotionResult(error="audio decoded to zero samples")
            feat = _apply_lfr(_fbank(samples))
            session = self._get_session()
            logits = session.run(None, {
                "x": feat[None].astype(np.float32),
                "x_length": np.array([feat.shape[0]], dtype=np.int32),
                "language": np.array([language], dtype=np.int32),
                "text_norm": np.array([text_norm], dtype=np.int32),
            })[0][0]

            vocab = self._load_vocab()
            blank = len(vocab) if len(vocab) not in vocab else max(vocab) + 1

            # Greedy CTC collapse (skip repeats and blank).
            prev, ids, frame_of = -1, [], []
            for t, row in enumerate(logits.argmax(-1)):
                i = int(row)
                if i != prev and i != blank and i != _BLANK_FALLBACK:
                    ids.append(i)
                    frame_of.append(t)
                prev = i
            tokens = [vocab.get(i, "") for i in ids]

            text_parts, emotion, events, lang = [], "neutral", [], ""
            for tok in tokens:
                if tok.startswith("<|") and tok.endswith("|>"):
                    name = tok[2:-2]
                    if name in _EMOTIONS:
                        emotion = name.lower()
                    elif name in _AUDIO_EVENTS:
                        if name.lower() not in {e.lower() for e in events}:
                            events.append(name)
                    elif len(name) <= 6 and name.isalpha():
                        # language tags: <|en|>, <|zh|>, <|ja|> ... and the
                        # <|woitn|>-style normalisation hints, which are all
                        # short alpha runs. Keep the first plausible language.
                        if not lang and name not in {"woitn", "withitn"}:
                            lang = name
                    continue
                text_parts.append(tok)

            text = "".join(text_parts).replace("▁", " ").strip()

            intensity = 0.0
            if emotion not in ("neutral", "unknown"):
                idx = {v: k for k, v in vocab.items()}.get(f"<|{emotion.upper()}|>")
                if idx is not None:
                    probs = _row_softmax(logits, idx)
                    if probs:
                        # Calibrate against measured posteriors rather than
                        # guessing a divisor. On this host the winning emotion
                        # token's probability ran 0.49-0.93 across real voice
                        # samples, so a fixed /0.35 pinned everything at 1.0 and
                        # the dial was useless. Map 0.45..0.95 onto 0..1 instead:
                        # below 0.45 is a marginal call, at/above 0.95 is a
                        # decisive one.
                        raw = probs[0]
                        intensity = round(min(1.0, max(0.0,
                                                       (raw - 0.45) / (0.95 - 0.45))), 2)

            return EmotionResult(
                text=text, emotion=emotion, intensity=intensity, events=events,
                language=lang, raw_tokens=tokens)
        except Exception as exc:  # noqa: BLE001 - never break the turn
            self.last_error = str(exc)
            logger.warning("sensevoice: recognition failed: %s", exc)
            return EmotionResult(error=str(exc))


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
