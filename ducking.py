"""Duck other applications' audio while Hermes speaks.

The complaint this solves is "the voice sounds quiet compared to other audio on my
system". Raising the file's loudness is the wrong lever once it is already louder
than the reference material: what actually helps is not competing with it.

Windows has a built-in version of this (Sound > Communications > "When Windows
detects communications activity") but it keys off *call* activity and only ducks to
a configurable fixed level. Measured on this host it was set to `0x3`
("Do nothing"), i.e. off. This module does it per-session on the actual
application volume, which is both more precise and reversible.

Trigger: watching Hermes itself, not a Python hook
--------------------------------------------------
The first version hooked `tools.voice_mode.mark_audio_output_active`. That was
wrong for the desktop app: the Electron **renderer** owns playback there (`new
Audio(...)` and a WebSocket `AudioContext`), so the Python playback helpers never
run and the hook never fired — `duck_count` stayed 0 through many utterances.

The reliable signal is WASAPI itself: a session's `State` is `Active` (1) while
that process is producing audio and `Inactive` (0) when it stops. Watching
Hermes' own sessions therefore triggers on *any* playback path — renderer, CLI
speaker, streamed PCM — with no host patching. A small poller (default 200 ms)
opens and closes the duck around those transitions.

Mechanism
---------
`pycaw` enumerates WASAPI sessions, so each app's master volume can be read and
written individually. Hermes' own session is skipped when ducking — attenuating the
voice to hear the voice is the one outcome worse than doing nothing.

Safety
------
Ducking is a global side effect on other applications, so the rules are:

* every ducked session records its exact prior volume and is restored to that
  value, never to an assumed 100%;
* a watchdog restores everything after a hard ceiling even if the stop transition
  is never observed (a crashed renderer, a killed player);
* a short hold-off after speech ends avoids flapping between sentences of one
  reply;
* the whole thing fails open — if pycaw or COM is unavailable, speech plays at
  full volume and nothing else changes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# How far to drop other apps. -14 dB is a clear step back without making the
# system feel muted; expressed as a linear multiplier for WASAPI.
DEFAULT_DUCK_VOLUME = 0.20
# Never duck lower than this, even if another app was already quiet.
_MIN_FLOOR = 0.02
# Hard ceiling on how long a duck may persist without an observed stop.
_WATCHDOG_SECONDS = 300.0
# Keep ducking this long after speech ends. Replies are synthesised sentence by
# sentence, so unducking in the gaps makes the system audio pump.
_HOLD_OFF_SECONDS = 1.2
# How often to poll session state when idle / while speaking.
_IDLE_POLL_SECONDS = 0.20
_ACTIVE_POLL_SECONDS = 0.12
# Session states (pycaw.constants.AudioSessionState).
_STATE_ACTIVE = 1
# Process names that count as "Hermes is speaking".
_SELF_NAMES = ("hermes.exe",)
# Processes that must NEVER be ducked, regardless of the watch list. Hermes is
# here so the voice is never attenuated by its own duck; the watch list and the
# skip list are deliberately separate, because a caller may watch one app while
# still refusing to duck another.
_NEVER_DUCK = ("hermes.exe", "hermes")


class AudioDucker:
    """Ducks other apps while a watched process is producing audio.

    Thread-safe and idempotent: the guard is a boolean transition, not a counter,
    because the trigger is observed state rather than paired calls.
    """

    def __init__(self, *, factor: float = DEFAULT_DUCK_VOLUME,
                 watch: tuple[str, ...] = _SELF_NAMES,
                 never_duck: tuple[str, ...] = _NEVER_DUCK,
                 enabled: bool = True, hold_off: float = _HOLD_OFF_SECONDS):
        self.factor = max(0.0, min(1.0, float(factor)))
        self.watch = tuple(w.lower() for w in watch)
        # Always keep the hard-coded skips even if a caller passes its own list:
        # ducking the voice to hear the voice is never intended.
        self.never_duck = tuple(
            dict.fromkeys([w.lower() for w in never_duck] + list(_NEVER_DUCK)))
        self.enabled = bool(enabled)
        self.hold_off = max(0.0, float(hold_off))
        self._lock = threading.RLock()
        self._ducked = False
        self._saved: dict[str, tuple[object, float]] = {}
        self._watchdog: Optional[threading.Timer] = None
        self._release_at = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._available: Optional[bool] = None
        self.last_error = ""
        self.duck_count = 0
        self.speaking = False

    # -- capability -------------------------------------------------------
    def available(self) -> bool:
        """True when pycaw + a WASAPI session list are usable (cached)."""
        if self._available is not None:
            return self._available
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401
            AudioUtilities.GetAllSessions()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - ducking is optional
            self.last_error = str(exc)
            self._available = False
            logger.info("audio ducking unavailable (%s); speech plays at normal level", exc)
        return self._available

    # -- internals --------------------------------------------------------
    @staticmethod
    def _label(session) -> str:
        try:
            return session.Process.name() if session.Process else "(system)"
        except Exception:  # noqa: BLE001 - a dead process has no name
            return f"pid{session.ProcessId}"

    def _sessions(self):
        from pycaw.pycaw import AudioUtilities
        return AudioUtilities.GetAllSessions()

    def own_is_speaking(self) -> bool:
        """True when any watched (Hermes) session is in the Active state."""
        try:
            for s in self._sessions():
                label = self._label(s).lower()
                if any(w in label for w in self.watch):
                    if int(s.State) == _STATE_ACTIVE:
                        return True
            return False
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            self.last_error = str(exc)
            return False

    # -- duck / restore ---------------------------------------------------
    def duck(self) -> int:
        """Lower every other app's volume. Returns how many sessions changed."""
        if not self.enabled or not self.available():
            return 0
        with self._lock:
            if self._ducked:
                return 0
            changed = 0
            try:
                for s in self._sessions():
                    label = self._label(s)
                    low = label.lower()
                    # Skip the process we are watching (it is the one speaking —
                    # attenuating it defeats the purpose) and anything on the
                    # hard never-duck list (Hermes itself).
                    if any(w in low for w in self.never_duck) or any(w in low for w in self.watch):
                        continue
                    key = f"{label}:{s.ProcessId}"
                    if key in self._saved:
                        continue
                    vol = s.SimpleAudioVolume
                    current = float(vol.GetMasterVolume())
                    if current <= _MIN_FLOOR:
                        continue          # already inaudible; leave it alone
                    target = max(_MIN_FLOOR, current * self.factor)
                    if abs(target - current) < 0.01:
                        continue
                    vol.SetMasterVolume(target, None)
                    self._saved[key] = (s, current)
                    changed += 1
                self._ducked = True
                if changed:
                    self.duck_count += 1
                    logger.debug("audio ducked: %d session(s) to %d%%",
                                 changed, int(self.factor * 100))
            except Exception as exc:  # noqa: BLE001 - never break playback
                self.last_error = str(exc)
                logger.debug("audio duck failed: %s", exc)
            self._arm_watchdog_locked()
            return changed

    def restore(self) -> int:
        """Put every ducked session back exactly where it was."""
        if not self.available():
            return 0
        with self._lock:
            return self._restore_locked()

    def _restore_locked(self) -> int:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        self._ducked = False
        self._release_at = 0.0
        restored = 0
        saved, self._saved = self._saved, {}
        for key, (session, volume) in saved.items():
            try:
                session.SimpleAudioVolume.SetMasterVolume(volume, None)
                restored += 1
            except Exception as exc:  # noqa: BLE001 - the app may have exited
                logger.debug("audio restore failed for %s: %s", key, exc)
        if restored:
            logger.debug("audio restored: %d session(s)", restored)
        return restored

    def _arm_watchdog_locked(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        t = threading.Timer(_WATCHDOG_SECONDS, self._watchdog_fire)
        t.daemon = True
        self._watchdog = t
        t.start()

    def _watchdog_fire(self) -> None:
        with self._lock:
            if not self._saved:
                return
            logger.warning(
                "audio ducking exceeded %.0fs without a stop; restoring other apps",
                _WATCHDOG_SECONDS)
            self._restore_locked()

    def force_restore(self) -> int:
        """Unconditional restore, for shutdown."""
        with self._lock:
            return self._restore_locked()

    # -- poller -----------------------------------------------------------
    def start(self) -> bool:
        """Begin watching for Hermes' own audio activity."""
        if not self.enabled or not self.available():
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="zonos3-ducker", daemon=True)
            self._thread.start()
        logger.info("audio ducking watching %s (factor %.2f)",
                    ", ".join(self.watch), self.factor)
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self.force_restore()

    def _loop(self) -> None:
        """Poll Hermes' session state; duck on start, restore after hold-off."""
        while not self._stop.is_set():
            try:
                speaking = self.own_is_speaking()
            except Exception:  # noqa: BLE001 - keep the poller alive
                speaking = False
            with self._lock:
                self.speaking = speaking
                now = time.monotonic()
                if speaking:
                    if not self._ducked:
                        self.duck()
                    self._release_at = 0.0
                elif self._ducked:
                    # Hold off briefly so inter-sentence gaps do not flap.
                    if self._release_at == 0.0:
                        self._release_at = now + self.hold_off
                    elif now >= self._release_at:
                        self._restore_locked()
            self._stop.wait(_ACTIVE_POLL_SECONDS if speaking else _IDLE_POLL_SECONDS)

    # -- introspection ----------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "available": self.available(),
                "factor": self.factor,
                "watch": list(self.watch),
                "never_duck": list(self.never_duck),
                "ducked": self._ducked,
                "speaking": self.speaking,
                "ducked_sessions": len(self._saved),
                "duck_count": self.duck_count,
                "polling": bool(self._thread is not None and self._thread.is_alive()),
                "last_error": self.last_error,
            }
