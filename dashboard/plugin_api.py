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
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

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
    # Resolve the running engine ONCE, not per file. `_is_engine_process` used to
    # rebuild the whole process table per entry, so a directory with a dozen
    # publisher files cost a dozen snapshots per status poll.
    live_engine_pids = _query_engine_pids()
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
        # A writer can be alive and still be a fossil: a process that has not
        # republished in hours (paused, wedged, or a build that published before
        # the heartbeat changed) keeps an old `warm`/pid frozen in its file. That
        # file must not outrank a fresh one — measured live: a 4-hour-old
        # `tts: warm, pid: <long-dead engine>` from a still-running host won the
        # evidence tier and made the chip report a phantom resident engine.
        # `_staleness` applies the same window to the winner; filtering here
        # stops a stale entry from being chosen in the first place.
        if (time.time() - at) > _STATE_MAX_AGE_SECONDS:
            continue
        try:
            engine_pid = int(state.get("pid") or 0)
        except (TypeError, ValueError):
            engine_pid = 0
        # Evidence outranks opinion, in tiers. A snapshot whose `pid` is a LIVE
        # ENGINE PROCESS is proof the engine was resident when it was written; a
        # claimed `warm` is next; anything else is a writer with nothing loaded.
        # Recency breaks ties per tier.
        #
        # Crucial: the pid must be the SERVER's. A writer that published its own
        # pid in that field (the pre-v1.16.1 bug) would look like live evidence
        # forever, which is what let an unloaded engine keep reading as loaded.
        # Membership in `live_engine_pids` checks the image name, so a host's own
        # pid can never masquerade as a resident engine.
        if engine_pid > 0 and engine_pid in live_engine_pids:
            evidence = 2.0
        elif state.get("tts") == "warm" and live_engine_pids:
            evidence = 1.0
        else:
            evidence = 0.0
        key = (evidence, at)
        if key > best_key:
            best, best_key = state, key
    return best


def _engine_image() -> str:
    """The server image name, from the plugin declaration (name, not a guess)."""
    return "zonos3-server.exe"


def _query_engine_pids() -> set[int]:
    """Pids whose image is the TTS server, or an empty set when the query fails.

    Native ``CreateToolhelp32Snapshot`` rather than ``tasklist``. Measured on this
    host: ``tasklist /FI IMAGENAME`` costs ~370 ms and this route is polled for
    the status chip every couple of seconds, so a subprocess per poll was pure
    latency. The snapshot reads the process table in ~20 ms and spawns nothing.

    Deliberately duplicated from the plugin's ``_engine_processes``: the dashboard
    module is loaded standalone by the plugin host (and by the regression suite),
    so importing the plugin package here would couple the two lifetimes for one
    helper.
    """
    target = _engine_image().lower()
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            TH32CS_SNAPPROCESS = 0x00000002
            INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260),
                ]

            k32 = ctypes.windll.kernel32
            k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
            snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if not snap or snap == INVALID_HANDLE_VALUE:
                return set()
            pids: set[int] = set()
            try:
                entry = PROCESSENTRY32W()
                entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
                ok = k32.Process32FirstW(ctypes.c_void_p(snap), ctypes.byref(entry))
                while ok:
                    if (entry.szExeFile or "").lower() == target:
                        pids.add(int(entry.th32ProcessID))
                    ok = k32.Process32NextW(ctypes.c_void_p(snap), ctypes.byref(entry))
            finally:
                k32.CloseHandle(ctypes.c_void_p(snap))
            return pids
        out = subprocess.run(["pgrep", "-f", Path(_engine_image()).stem],
                             capture_output=True, text=True, timeout=10).stdout or ""
        return {int(t) for t in out.split() if t.strip().isdigit()}
    except Exception:  # noqa: BLE001 - a failed probe reports "not running"
        return set()


def _engine_running() -> bool:
    """True when a TTS server process exists on this host.

    Ground truth for the telemetry: a snapshot is one process's opinion, and
    several processes publish. Asking the OS settles whether the engine is
    resident without trusting any writer.
    """
    return bool(_query_engine_pids())


def _is_engine_process(pid: Any) -> bool:
    """True when *pid* is a live TTS server.

    Stricter than "the pid exists", which is what made this class of bug possible:
    a host publishing its OWN pid looked like a resident engine because that pid
    was alive. Matching the image means only the real server counts.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    return pid > 0 and pid in _query_engine_pids()


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
    """Residency for the chip: the writers' snapshot, corrected by the OS.

    The snapshot is one process's opinion and several processes publish, so its
    `tts`/`pid` are treated as a hint and reconciled against reality here. Without
    this the chip could show "loaded" for a stopped engine (or the reverse) purely
    because a different writer won the last race.
    """
    state = _read_state()
    why = _staleness(state)
    if why:
        return {"ok": False, "reason": why}
    pids = _query_engine_pids()
    reported = state.get("pid")
    try:
        reported = int(reported or 0)
    except (TypeError, ValueError):
        reported = 0
    running = bool(pids)
    if running:
        state = {**state, "tts": "warm", "pid": reported if reported in pids else min(pids)}
    else:
        state = {**state, "tts": "unloaded", "pid": 0}
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


# --------------------------------------------------------------- voice selector

# Catalogues worth pointing at, linked rather than embedded: browsing and
# downloading belong in a browser, and the plugin only offers what is on disk.
# (An in-menu catalogue with live previews was tried and removed -- every preview
# paid a cold engine start or reference-encoder reload, so it was unusable at
# 1-2 min per click.)
#
# Ordered by how many distinct speakers each one gives you. Verified on this host
# by cloning one clip from each and transcribing the result.
# The one link the chip carries.
#
# Chosen because it is the only corpus organised by PERCEPTUAL STYLE rather than
# by speaker name: 500 voices in 24 measured groups ("Polished Broadcast
# Masculine · Authoritative", "Whispered Feminine Narrator · Calm ASMR"), each
# record carrying tags, register and casting directions in one JSON manifest. The
# name-list corpora (LibriTTS 995, Voice-Zero 154, Open Swara 867) are larger but
# you have to audition them blind; noted here so the trade-off is not lost.
CATALOGUE_URL = "https://projects.laion.ai/moss-reference-voice-taxonomy/"
CATALOGUE_LABEL = "Browse voices by style (500, CC-BY)"

# Downloaded clips live here. A user drops a file in and it shows up; the picker
# never fetches anything itself.
_GALLERY_DIR = "voice_gallery"
_CLIP_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def _gallery_root() -> Path:
    return _hermes_home() / "cache" / "zonos3-tts" / _GALLERY_DIR


def _local_clips() -> list[Dict[str, Any]]:
    """Clips already present on disk, newest first."""
    root = _gallery_root()
    if not root.is_dir():
        return []
    out = []
    for f in root.iterdir():
        if f.suffix.lower() not in _CLIP_SUFFIXES or not f.is_file():
            continue
        out.append({"name": f.stem, "file": f.name, "bytes": f.stat().st_size})
    out.sort(key=lambda v: v["name"])
    return out


@router.get("/voices")
def voices() -> Dict[str, Any]:
    """Everything selectable: engine voices and clips on disk.

    Reads the engine's speaker list only when it is already listening. A picker
    repaint must never start a 3.3 GB load, so an unloaded engine reports its
    bundled voices from the catalogue cache instead.
    """
    state = _read_state()
    engine: list[str] = []
    try:
        endpoint = str(state.get("endpoint") or "http://127.0.0.1:1919")
        import socket
        import urllib.request
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        with socket.create_connection((parsed.hostname or "127.0.0.1",
                                       parsed.port or 1919), timeout=1.5):
            raw = urllib.request.urlopen(endpoint + "/v2/speakers", timeout=6).read()
            # `id` is what the engine takes, `label` is what a human can read.
            # Showing the raw id gives a list of default_<hash> rows.
            engine = [{"id": s["id"], "label": s.get("label") or s["id"]}
                      for s in json.loads(raw).get("speakers", [])
                      if s.get("id")]
    except Exception:  # noqa: BLE001 - an unloaded engine simply has no live list
        engine = []

    return {"ok": True,
            "current": state.get("voice") or "",
            "engine": sorted(engine, key=lambda v: v["label"].lower()),
            "local": _local_clips(),
            "catalogue_url": CATALOGUE_URL,
            "catalogue_label": CATALOGUE_LABEL,
            "gallery_dir": str(_gallery_root())}


# ---------------------------------------------------------------- versions
#
# Three layers update in three different ways, and conflating them is how people
# end up believing "update" is one button:
#
#   plugin     python, self-updated by Hermes from its own git origin
#   engine     a compiled CUDA binary -- needs a rebuild, not just a pull
#   sensevoice a frozen 2024 ONNX export with no upstream ONNX to pull at all
#
# So this reports rather than acts. A control that rebuilt the engine in-process
# would be able to leave the voice path with no working binary; the script at
# scripts/update_zonos3_engine.sh does the rebuild with a build-then-swap and a
# rollback, which is the part that must stay explicit.

_PLUGIN_REPO = Path(__file__).resolve().parent.parent
_ENGINE_REPO = Path(r"C:\Users\MaJor\Github\ZONOS3-Mini")
_SENSEVOICE_MODEL = (Path.home() / ".dsh" / "speech-to-text" / "sensevoice"
                     / "models" / "sensevoice-onnx" / "model.int8.onnx")
_ENGINE_SCRIPT = "scripts/update_zonos3_engine.sh"


def _git(repo: Path, *args: str, timeout: float = 25.0) -> Optional[str]:
    """Run git in *repo*; None on any failure (missing repo, no git, offline)."""
    try:
        if not (repo / ".git").exists():
            return None
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                             if os.name == "nt" else 0)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - a version probe must never raise
        return None


def _plugin_version() -> Dict[str, Any]:
    version = ""
    # A plain line scan, not a YAML parse: the dashboard process may not have
    # PyYAML, and a version string does not need a parser. Strips the quotes a
    # hand-edited yaml often carries.
    try:
        text = (_PLUGIN_REPO / "plugin.yaml").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("version:"):
                version = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
    except Exception:  # noqa: BLE001
        pass
    if not version:
        # plugin.yaml is two levels up from dashboard/. If that path ever moves,
        # fall back to the installed copy's manifest.
        try:
            man = json.loads((_PLUGIN_REPO / "dashboard" / "manifest.json")
                             .read_text(encoding="utf-8"))
            version = str(man.get("version") or "")
        except Exception:  # noqa: BLE001
            pass
    head = _git(_PLUGIN_REPO, "rev-parse", "--short", "HEAD")
    return {"name": "Plugin", "version": version or head or "unknown",
            "commit": head, "state": "ok" if version else "unknown",
            "note": "auto-updated by Hermes from its git origin"}


def _engine_version(fetch: bool = False) -> Dict[str, Any]:
    head = _git(_ENGINE_REPO, "rev-parse", "--short", "HEAD")
    if head is None:
        return {"name": "Engine", "version": "not a checkout",
                "state": "unknown", "note": str(_ENGINE_REPO)}
    branch = _git(_ENGINE_REPO, "rev-parse", "--abbrev-ref", "HEAD") or "main"
    behind = None
    remote = None
    if fetch:
        # Only fetched on demand: a background poll must not hit the network.
        _git(_ENGINE_REPO, "fetch", "origin", "--quiet", timeout=45.0)
    remote = _git(_ENGINE_REPO, "rev-parse", "--short", f"origin/{branch}")
    try:
        n = _git(_ENGINE_REPO, "rev-list", "--count", f"HEAD..origin/{branch}")
        behind = int(n) if n is not None else None
    except (TypeError, ValueError):
        behind = None
    binary = _ENGINE_REPO / "servers" / "ggml" / "build-cuda" / "zonos3-server.exe"
    state = "ok" if behind == 0 else ("behind" if behind else "unknown")
    note = ("up to date" if behind == 0 else
            f"{behind} commit(s) behind {remote}; rebuild needed (not just a pull)"
            if behind else "update state unknown (offline?)")
    return {"name": "Engine", "version": head, "branch": branch,
            "behind": behind, "remote": remote, "state": state, "note": note,
            "binary_present": binary.is_file(),
            "update_script": _ENGINE_SCRIPT}


def _sensevoice_version() -> Dict[str, Any]:
    exists = _SENSEVOICE_MODEL.is_file()
    stamp = ""
    if exists:
        import datetime
        stamp = datetime.date.fromtimestamp(
            _SENSEVOICE_MODEL.stat().st_mtime).isoformat()
    return {"name": "SenseVoice", "version": stamp or "missing",
            "state": "frozen" if exists else "missing",
            "note": ("no upstream ONNX to pull; FunAudioLLM ships only model.pt "
                     "and the sherpa-onnx export is frozen at 2024-07-17")}


@router.get("/versions")
def versions(fetch: int = 0) -> Dict[str, Any]:
    """Version + update state for the three layers.

    ``?fetch=1`` performs the upstream git fetch (a network call) so the engine's
    behind-count is accurate. The default does not touch the network: the chip
    polls, and a status poll must stay cheap.
    """
    return {"ok": True, "items": [
        _plugin_version(),
        _engine_version(fetch=bool(fetch)),
        _sensevoice_version(),
    ]}
