"""ZONOS3-Mini — dashboard backend for the residency chip.

Mounted at ``/api/plugins/zonos3-tts/*`` (manifest.json -> ``"api"``).

The agent process owns the engine and the leases; this half only *reads* the
state it publishes. That separation is deliberate: an unload policy that the UI
could silently contradict would be worse than no chip at all.

Routes
------
- ``GET  /status``   engine residency, leases, windows, retry count, voice list
- ``POST /settings`` idle-unload / process-stop windows and retry attempts

Every setting here is re-read live by the agent half (``_live_setting_fn``), so
a change takes effect on the next release without restarting Hermes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

_PLUGIN_ID = "zonos3-tts"
_STATE_FILE = "_engine_state.json"
# The agent republishes every ~20s (heartbeat). Anything older means the writer
# is gone and the file is a fossil, not live state.
_STATE_MAX_AGE_SECONDS = 90.0


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _state_path() -> Path:
    path = _hermes_home() / "cache" / _PLUGIN_ID
    path.mkdir(parents=True, exist_ok=True)
    return path / _STATE_FILE


def _read_state() -> Dict[str, Any]:
    """The most trustworthy snapshot among the per-writer state files.

    More than one process can host this plugin (the gateway serves read-aloud, the
    dashboard serves the chip's routes, the relay serves voice), and each
    constructs its own supervisor and publishes its own snapshot. A single shared
    file then flickers: whichever process wrote last wins, so a stale instance can
    overwrite a fresh one and the chip reports "no streaming" for a provider that
    is streaming. So each writer publishes to `_engine_state.<pid>.json` and this
    reader picks between them.

    Picking the NEWEST is not enough, and the bug that taught us is worth naming:
    a process whose supervisor never loaded anything still publishes
    `{"tts": "idle", "pid": 0}` forever, and when its tick lands between two
    others' the chip flaps warm → idle → unloaded while the engine never moved.
    So writers that have actual evidence (an engine pid, or `warm`) outrank writers
    that only have an opinion; recency breaks ties within that. A dead writer is
    ignored outright. The legacy unsuffixed file is still read so an older writer
    keeps working.
    """
    folder = _state_path().parent
    best: Dict[str, Any] = {}
    best_key = (-1.0, -1.0)
    for path in sorted(folder.glob("_engine_state*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            at = float(state.get("updated_at") or 0)
        except (TypeError, ValueError):
            continue
        owner = state.get("agent_pid")
        if owner is not None and not _pid_alive(owner):
            continue  # a fossil from a process that exited
        try:
            engine_pid = int(state.get("pid") or 0)
        except (TypeError, ValueError):
            engine_pid = 0
        # Evidence outranks opinion, in tiers: a live engine pid is proof; "warm"
        # with pid 0 is an adopted engine (real, but unverifiable here); anything
        # else is a writer with nothing loaded. Recency breaks ties per tier.
        if engine_pid > 0 and _pid_alive(engine_pid):
            evidence = 2.0
        elif state.get("tts") == "warm":
            evidence = 1.0
        else:
            evidence = 0.0
        key = (evidence, at)
        if key > best_key:
            best, best_key = state, key
    return best


def _pid_alive(pid: Any) -> bool:
    """True when *pid* is running.

    On Windows ``os.kill(pid, 0)`` is not a probe — it calls TerminateProcess —
    so liveness comes from OpenProcess/GetExitCodeProcess.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _staleness(state: Dict[str, Any]) -> str:
    if not state:
        return "plugin state not published yet"
    try:
        age = time.time() - float(state.get("updated_at") or 0)
    except (TypeError, ValueError):
        return "plugin state has no timestamp"
    if age > _STATE_MAX_AGE_SECONDS:
        return f"agent half went quiet {age / 60:.1f} min ago"
    if "agent_pid" in state and not _pid_alive(state["agent_pid"]):
        return "agent process is no longer running"
    return ""


@router.get("/status")
def status() -> Dict[str, Any]:
    state = _read_state()
    why = _staleness(state)
    if why:
        return {"ok": False, "reason": why}
    return {"ok": True, **state}


@router.post("/emotion")
def emotion_probe(payload: Dict[str, Any]) -> Any:
    """Analyse an audio file with SenseVoice and report the detected affect.

    Lets the dashboard test the emotion path without a live voice call. The model
    runs in the *backend* process here, which is cheap (CPU, int8) but does not
    share the agent's pending-emotion state — this is a probe, not the live
    pipeline, and the response says so.
    """
    audio_path = str(payload.get("audio_path") or "").strip()
    if not audio_path:
        return JSONResponse({"ok": False, "error": "audio_path is required"},
                            status_code=400)
    try:
        import importlib.util
        import sys

        plugin_init = Path(__file__).resolve().parent.parent / "__init__.py"
        name = "zonos3_tts_dashboard_probe"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, plugin_init)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load the plugin module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        module = sys.modules[name]
        detector = module.SenseVoiceEmotion()
        if not detector.is_available():
            return JSONResponse(
                {"ok": False, "error": "SenseVoice ONNX model or onnxruntime unavailable"},
                status_code=503)
        result = detector.recognise(Path(audio_path).expanduser())
        return {"ok": result.ok, "emotion": result.emotion,
                "intensity": result.intensity, "events": result.events,
                "language": result.language, "text": result.text,
                "error": result.error, "source": "probe (does not affect the live pipeline)"}
    except Exception as exc:  # noqa: BLE001 - a probe must not 500 opaquely
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.post("/unload")
def unload() -> Any:
    """Ask the agent process to stop the engine and release its VRAM.

    This route runs in the dashboard/web-server process, which does **not** own
    the engine — the supervisor lives in whichever process registered the plugin.
    Calling ``stop_process()`` here would fork a second supervisor instead of
    stopping the real one, so the request is handed over as a file the owning
    process consumes on its poll loop (up to ~20 s, and the next status read
    shows the result).

    Refused while a surface holds a lease: unloading then would leave a live
    lease pointing at a dead engine, and the next utterance would pay a cold
    start mid-sentence.
    """
    state = _read_state()
    why = _staleness(state)
    if why:
        return JSONResponse({"ok": False, "error": why}, status_code=503)

    if state.get("tts_leased"):
        return JSONResponse({
            "ok": False,
            "state": state.get("tts"),
            "error": "engine is leased by a speaking surface; release it first",
        }, status_code=409)

    tts = state.get("tts")
    if tts in ("unloaded", "idle") and not state.get("pid"):
        # Nothing resident: report the honest state rather than queueing a
        # request that would have nothing to do.
        return {"ok": True, "state": tts, "already": True,
                "message": "engine is not loaded"}

    try:
        path = _state_path().parent
        path.mkdir(parents=True, exist_ok=True)
        target = path / "_unload_request.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"reason": "manual", "requested_at": time.time()}),
                       encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:  # noqa: BLE001 - a UI request must not 500 opaquely
        return JSONResponse({"ok": False, "error": f"could not queue unload: {exc}"},
                            status_code=500)

    return {"ok": True, "state": tts, "queued": True,
            "message": "unload requested; the engine stops within a couple of seconds"}


@router.post("/stt/unload")
def stt_unload() -> Any:
    """Ask the agent process to release the SenseVoice backend.

    Separate from ``/unload`` on purpose: the recogniser is CPU/RAM (~323 MB,
    0 VRAM) while the voice engine is GPU/VRAM, and wanting one freed says
    nothing about the other.

    Like ``/unload``, this runs in the dashboard process which does not own the
    detector, so it hands the request over as a file the owning process consumes.
    Refused while an STT lease is held: dropping the backend mid-recording would
    force a ~1.5 s rebuild inside the user's next utterance.
    """
    state = _read_state()
    why = _staleness(state)
    if why:
        return JSONResponse({"ok": False, "error": why}, status_code=503)

    stt = state.get("stt") or {}
    if not stt:
        return JSONResponse({
            "ok": False,
            "error": "the sensevoice STT provider is not registered "
                     "(set stt_enabled: true and stt.provider: sensevoice)",
        }, status_code=409)

    if state.get("stt_leased"):
        return JSONResponse({
            "ok": False,
            "error": "a recording surface holds the recogniser; release it first",
        }, status_code=409)

    if stt.get("loaded") is False:
        return {"ok": True, "already": True, "message": "the recogniser is not loaded"}

    try:
        folder = _state_path().parent
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "_stt_unload_request.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"reason": "manual", "requested_at": time.time()}),
                       encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:  # noqa: BLE001 - a UI request must not 500 opaquely
        return JSONResponse({"ok": False, "error": f"could not queue unload: {exc}"},
                            status_code=500)

    return {"ok": True, "queued": True,
            "message": "STT unload requested; the recogniser is released within a "
                       "couple of seconds"}


@router.post("/settings")
def settings(payload: Dict[str, Any]) -> Any:
    """Persist residency policy into this plugin's settings subtree."""
    from hermes_cli.plugins import save_plugin_setting

    numeric = {"unload_after_seconds": 0.0, "start_timeout_seconds": 1.0,
               "request_timeout_seconds": 1.0}
    written: Dict[str, Any] = {}
    for key, floor in numeric.items():
        if key not in payload:
            continue
        try:
            value = max(floor, float(payload[key]))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": f"{key} is not a number"},
                                status_code=400)
        save_plugin_setting(_PLUGIN_ID, (key,), value)
        written[key] = value

    if "retry_attempts" in payload:
        try:
            value = max(1, int(payload["retry_attempts"]))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "retry_attempts is not an integer"},
                                status_code=400)
        save_plugin_setting(_PLUGIN_ID, ("retry_attempts",), value)
        written["retry_attempts"] = value

    if "voice" in payload:
        value = str(payload["voice"] or "")
        save_plugin_setting(_PLUGIN_ID, ("voice",), value)
        written["voice"] = value

    if "emotion_aware" in payload:
        value = payload["emotion_aware"]
        if not isinstance(value, bool):
            value = str(value).strip().lower() in {"1", "true", "yes", "on"}
        save_plugin_setting(_PLUGIN_ID, ("emotion_aware",), value)
        written["emotion_aware"] = value

    return {"ok": True, "written": written}
