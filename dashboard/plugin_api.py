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


# --------------------------------------------------------------- voice catalogue

_GALLERY_DIR = "voice_gallery"
_GALLERY_RAW = "https://raw.githubusercontent.com/OwenTyme/voice-zero/main/voices/"
# The catalogue is a fixed open corpus, not a live search: 154 CC0-licensed
# reference clips built for zero-shot TTS. Kept as a cached list so the picker
# opens instantly and works offline once fetched.
_GALLERY_CACHE = "gallery_index.json"
_GALLERY_TTL = 86400.0


def _gallery_root() -> Path:
    return _hermes_home() / "cache" / "zonos3-tts" / _GALLERY_DIR


def _gallery_index() -> Dict[str, Any]:
    """Voice-Zero clip list, cached for a day.

    Raises nothing: a failed fetch reports the error so the picker can say "the
    catalogue is unavailable" instead of showing an empty list that looks like
    the corpus disappeared.
    """
    cache = _hermes_home() / "cache" / "zonos3-tts" / _GALLERY_CACHE
    try:
        if cache.is_file() and (time.time() - cache.stat().st_mtime) < _GALLERY_TTL:
            return json.loads(cache.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a bad cache is a miss, not a failure
        pass

    import urllib.request

    url = ("https://api.github.com/repos/OwenTyme/voice-zero/"
           "git/trees/main?recursive=1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-plugin"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        tree = json.loads(raw).get("tree", [])
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"ok": False, "error": f"catalogue unavailable: {exc}", "voices": []}

    voices = []
    for entry in tree:
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        # Only the flat voices/ directory: voices-emotion/ holds Chatterbox
        # SYNTHETIC variations, which are other models' acting and would fight
        # this engine's own emotion steering if used as an embedding.
        if not path.startswith("voices/") or not path.endswith(".flac"):
            continue
        name = Path(path).stem
        voices.append({"name": name, "file": Path(path).name,
                       "url": _GALLERY_RAW + Path(path).name})
    voices.sort(key=lambda v: v["name"])
    payload = {"ok": True, "license": "CC0", "count": len(voices), "voices": voices}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:  # noqa: BLE001 - cache write is best-effort
        pass
    return payload


def _local_voices() -> list[Dict[str, Any]]:
    """Clips already downloaded into the local gallery."""
    root = _gallery_root()
    if not root.is_dir():
        return []
    out = []
    for f in sorted(root.iterdir()):
        if f.suffix.lower() not in {".flac", ".wav", ".mp3", ".ogg", ".m4a"}:
            continue
        out.append({"name": f.stem, "file": f.name,
                    "bytes": f.stat().st_size,
                    "downloaded_at": f.stat().st_mtime})
    return out


@router.get("/voices")
def voices() -> Dict[str, Any]:
    """One list for the picker: engine voices, the open catalogue, and local clips."""
    state = _read_state()
    engine: list[str] = []
    try:
        endpoint = str(state.get("endpoint") or "http://127.0.0.1:1919")
        import urllib.request

        raw = urllib.request.urlopen(endpoint + "/v2/speakers", timeout=6).read()
        engine = [s.get("id") or s.get("name") for s in
                  json.loads(raw).get("speakers", []) if (s.get("id") or s.get("name"))]
    except Exception:  # noqa: BLE001 - an unloaded engine simply has no list
        engine = []

    return {"ok": True,
            "current": state.get("voice") or "",
            "engine": sorted(engine),
            "local": _local_voices(),
            "gallery": _gallery_index()}


@router.post("/voices/preview")
def voice_preview(payload: Dict[str, Any]) -> Any:
    """Synthesise a short line so a catalogue entry can be auditioned.

    Returns base64 WAV inline because the desktop plugin bridge is JSON-only --
    there is no binary response channel for a plugin route. The clip is short and
    the engine is local, so this stays a few hundred KB.
    """
    import base64

    name = str(payload.get("name") or "").strip()
    url = str(payload.get("url") or "").strip()
    # One short line, not a pangram. Rendering time scales with length (~20 s for
    # a 14-word sentence on this host), and an audition only needs enough to judge
    # timbre -- the same reason a voice picker previews a phrase rather than a
    # paragraph.
    text = str(payload.get("text") or "Hello, this is my voice.").strip()
    if not (name or url):
        return JSONResponse({"ok": False, "error": "name or url required"}, status_code=400)
    text = text[:120]        # a long preview is slow and adds nothing to a judgement

    # Preview must work for a catalogue entry that has NOT been downloaded yet --
    # that is the whole point, auditioning before committing. So fall back to the
    # catalogue URL rather than requiring a local file. The fetched clip is cached
    # in the gallery so a second preview is instant.
    stem = Path(name).stem if name else (Path(url).stem if url else "")
    if not stem:
        return JSONResponse({"ok": False, "error": "name or url required"}, status_code=400)
    clip = _gallery_root() / f"{stem}.flac"
    if not clip.is_file():
        src = url or (_GALLERY_RAW + stem + ".flac")
        try:
            import urllib.request

            req = urllib.request.Request(src, headers={"User-Agent": "hermes-plugin"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if len(data) < 10000:
                return JSONResponse(
                    {"ok": False, "error": "clip looks truncated"}, status_code=502)
            _gallery_root().mkdir(parents=True, exist_ok=True)
            clip.write_bytes(data)
        except Exception as exc:  # noqa: BLE001 - reported
            return JSONResponse({"ok": False, "error": f"download failed: {exc}"},
                                status_code=502)

    # The engine is usually IDLE-UNLOADED (that is the point of the idle policy),
    # so posting straight to it is a guaranteed ECONNREFUSED 10061. Auditioning a
    # voice is an explicit request for speech, so starting the engine here is
    # correct -- what must NOT happen is taking a TTS lease, because that would
    # block the idle unload these previews rely on.
    endpoint = str(_read_state().get("endpoint") or "http://127.0.0.1:1919")
    try:
        _ensure_engine_for_preview(endpoint)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return JSONResponse(
            {"ok": False, "error": f"the speech engine could not be started: {exc}"},
            status_code=502)

    try:
        # `format: wav` is rejected with "format=wav needs stream=false" unless
        # stream is explicit, so send it rather than relying on the default.
        payload_body = {"text": text, "format": "wav", "stream": False,
                        "speaker_audio_base64": _b64_clip(clip),
                        "speaker_mode": "reference"}
        wav = _engine_post(endpoint + "/v2/generate", payload_body, timeout=240)
        if not wav[:4] == b"RIFF":
            return JSONResponse({"ok": False, "error": "engine returned no audio"},
                                status_code=502)
        return {"ok": True, "name": clip.stem,
                "audio": base64.b64encode(wav).decode("ascii"),
                "bytes": len(wav)}
    except Exception as exc:  # noqa: BLE001 - a preview must never 500 the chip
        # 10061 is a WinError number with no useful text; translate it, or the
        # chip shows "<urlopen error [Errno 10061] Unknown error>".
        return JSONResponse({"ok": False, "error": _describe_engine_error(exc)},
                            status_code=502)


def _engine_post(url: str, payload: Dict[str, Any], timeout: float = 240) -> bytes:
    """POST JSON to the engine, retrying once through a fresh start.

    The engine can also die between the health probe and the request (idle timer,
    OOM kill), so one retry after re-starting turns a spurious 10061 into a
    working preview instead of an error the user has to understand.
    """
    import time
    import urllib.request

    body = json.dumps(payload).encode()
    last: Optional[Exception] = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == 0:
                _ensure_engine_for_preview(url.split("/v2/")[0], force_restart=True)
                time.sleep(0.5)
                continue
            raise
    raise last if last else RuntimeError("engine request failed")


def _describe_engine_error(exc: Exception) -> str:
    """A human sentence for an engine failure, never a bare errno."""
    text = str(exc)
    if "10061" in text or "refused" in text.lower():
        return ("the speech engine is not running and did not start in time; "
                "it will load on the next request")
    if "10054" in text or "forcibly closed" in text.lower():
        return "the speech engine dropped the connection; try again"
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "the speech engine took too long to render the preview"
    return text


def _ensure_engine_for_preview(endpoint: str, *, force_restart: bool = False) -> None:
    """Start the engine if it is not listening.

    Deliberately does NOT go through the agent or acquire a lease: a lease is a
    claim on the engine the idle policy must respect, and a UI preview has no
    business holding one. The engine's own idle timer unloads it again.

    The argv mirrors EngineSupervisor._argv() exactly -- the backbone GGUF is a
    POSITIONAL argument and the encoders are flags, so guessing `--model-dir`
    produces a server that exits immediately.
    """
    import time

    if not force_restart and _engine_listening(endpoint):
        return

    exe = _engine_executable()
    if exe is None or not exe.is_file():
        raise RuntimeError("zonos3-server.exe was not found; set settings.server_exe")

    gguf = _hermes_home() / "models" / "zonos3-mini"
    backbone = gguf / "backbone-ptq.gguf"
    if not backbone.is_file():
        raise RuntimeError(f"the backbone weights are missing at {backbone}")

    port = endpoint.rsplit(":", 1)[-1].rstrip("/") or "1919"
    args = [str(exe), str(backbone), "--port", port, "--host", "127.0.0.1",
            "--spk-encoder", str(gguf / "spk-encoder.gguf"),
            "--tts-default-voices-dir", str(gguf / "voices"), "--gpu"]
    if (gguf / "steer.gguf").is_file():
        args += ["--steer", str(gguf / "steer.gguf")]
    if (gguf / "zac-encoder.gguf").is_file() and (gguf / "ssl-teacher.gguf").is_file():
        args += ["--zac-encoder", str(gguf / "zac-encoder.gguf"),
                 "--ssl-teacher", str(gguf / "ssl-teacher.gguf")]
    # 60 s is the plugin's normal choice, but a voice BROWSE is a burst of
    # previews: letting the ~3.4 GiB raw-reference encoder decay between clicks
    # makes each one pay a reload (measured 108 s cold vs 16 s warm). The preview
    # spawn therefore holds it longer and lets the engine's own idle policy reclaim
    # the whole process afterwards.
    args += ["--reference-idle-unload", "600"]

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=creationflags,
                     close_fds=True)

    # First start touches ~3.3 GB of weights plus the CUDA context, so this is
    # slow by nature. Poll the port rather than sleeping a fixed guess.
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if _engine_listening(endpoint):
            return
        time.sleep(0.5)
    raise RuntimeError("the engine did not open its port within 120 s")


def _engine_listening(endpoint: str) -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1",
                                       parsed.port or 1919), timeout=1.5):
            return True
    except OSError:
        return False


def _engine_executable() -> Optional[Path]:
    """Locate zonos3-server.exe from the plugin settings, then the usual homes."""
    candidates: list[Path] = []
    try:
        # Same import the /settings route uses, so it is known to resolve here.
        from hermes_cli.plugins import save_plugin_setting  # noqa: F401
        from hermes_cli import config as _config
        raw = _config.load_config() or {}
        entries = ((raw.get("plugins") or {}).get("entries") or {})
        settings = (entries.get(_PLUGIN_ID) or {}).get("settings") or {}
        if settings.get("server_exe"):
            candidates.append(Path(str(settings["server_exe"])))
    except Exception:  # noqa: BLE001 - settings lookup is optional
        pass
    candidates.append(Path(r"C:\Users\MaJor\Github\ZONOS3-Mini\servers\ggml\build-cuda\zonos3-server.exe"))
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0] if candidates else None


def _b64_clip(path: Path) -> str:
    """base64 of a 16 kHz mono PCM WAV, the form the engine accepts."""
    import base64
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    try:
        from tools.transcription_audio import _find_ffmpeg_binary
        ffmpeg = _find_ffmpeg_binary() or ffmpeg
    except Exception:  # noqa: BLE001 - host helper is a convenience
        pass
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare a voice clip")

    # Cache the converted clip beside the original. Re-running ffmpeg on every
    # preview added seconds of variance per click (measured 11 s to 72 s for the
    # same voice), because a repeated audition re-encoded an already-encoded file.
    cached = path.with_suffix(path.suffix + ".16k.b64")
    try:
        if cached.is_file() and cached.stat().st_mtime >= path.stat().st_mtime:
            return cached.read_text(encoding="ascii")
    except Exception:  # noqa: BLE001 - a bad cache is a miss, not a failure
        pass

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ref.wav"
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path), "-t", "30",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"ffmpeg could not prepare the clip: {proc.stderr[:200]}")
        encoded = base64.b64encode(out.read_bytes()).decode("ascii")
    try:
        cached.write_text(encoded, encoding="ascii")
    except Exception:  # noqa: BLE001 - cache write is best-effort
        pass
    return encoded


@router.post("/voices/add")
def voice_add(payload: Dict[str, Any]) -> Any:
    """Download a catalogue clip into the local gallery so it can be selected."""
    name = str(payload.get("name") or "").strip()
    url = str(payload.get("url") or "").strip()
    if not name and not url:
        return JSONResponse({"ok": False, "error": "name or url required"}, status_code=400)
    if not url:
        url = _GALLERY_RAW + Path(name).stem + ".flac"
    dest = _gallery_root() / (Path(name).stem if name else Path(url).stem)
    dest = dest.with_suffix(".flac")
    if dest.is_file():
        return {"ok": True, "name": dest.stem, "already": True}
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "hermes-plugin"})
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(urllib.request.urlopen(req, timeout=90).read())
    except Exception as exc:  # noqa: BLE001 - reported
        return JSONResponse({"ok": False, "error": f"download failed: {exc}"},
                            status_code=502)
    return {"ok": True, "name": dest.stem, "bytes": dest.stat().st_size}


@router.post("/voices/remove")
def voice_remove(payload: Dict[str, Any]) -> Any:
    """Delete a downloaded clip from the local gallery."""
    name = str(payload.get("name") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
    target = _gallery_root() / (Path(name).stem + ".flac")
    try:
        if target.is_file():
            target.unlink()
    except Exception as exc:  # noqa: BLE001 - reported
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return {"ok": True, "name": target.stem}
