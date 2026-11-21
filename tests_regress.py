"""Regression suite for the zonos3 TTS provider.

Run:  python tests_regress.py   (from the plugin directory, HERMES_HOME set)

Kept inside the plugin rather than in $TMPDIR: the scratch cleaner prunes idle
files after 24 h, and this suite is the only thing that catches the lifecycle
regressions it asserts (lease gating, engine-pid reporting, level safety).
"""
# pyright: reportOptionalMemberAccess=false, reportArgumentType=false, reportAttributeAccessIssue=false
# ^ module_from_spec()/spec.loader are Optional in the stubs; this is a script that
#   deliberately swaps module internals (api._state_path) to isolate the reader.
import importlib.util
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HOME = Path(os.environ["HERMES_HOME"])
sys.path.insert(0, str(HOME / "hermes-agent"))
PLUGIN = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("zreg", PLUGIN / "__init__.py")
m = importlib.util.module_from_spec(spec)
sys.modules["zreg"] = m
spec.loader.exec_module(m)

aspec = importlib.util.spec_from_file_location("zreg_api", PLUGIN / "dashboard" / "plugin_api.py")
api = importlib.util.module_from_spec(aspec)
aspec.loader.exec_module(api)

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

RUNTIME = Path(r"C:\Users\MaJor\Github\ZONOS3-Mini\servers\ggml\build-cuda")

def supervisor(**kw):
    return m.EngineSupervisor(runtime_dir=RUNTIME,
                              model_dir=HOME / "models" / "zonos3-mini",
                              default_voice="Savannah", **kw)

sup = supervisor()
prov = m.Zonos3TTS(sup, dynamic_range=m._DEFAULT_DYNAMIC_RANGE,
                   expressive=m._DEFAULT_EXPRESSIVE,
                   volume_boost=m._DEFAULT_VOLUME_BOOST)
sup.ensure_up()

print("== voice resolution ==")
for name in ("Savannah", "Imogen", "Wyatt"):
    check(f"resolve {name}", bool(sup.resolve_voice(name)))
check("unknown voice -> None", sup.resolve_voice("NotARealVoice") is None)

print("\n== buffered synthesis ==")
out = Path(os.environ["TMPDIR"]) / "zreg_out.wav"
prov.synthesize("The quick brown fox jumps over the lazy dog.", str(out), voice="Savannah")
raw = out.read_bytes()
check("WAV written", len(raw) > 1000, f"{len(raw)} bytes")
with wave.open(str(out)) as w:
    rate = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
check("44.1 kHz header", rate == 44100, str(rate))
check("not silent", int(np.abs(a).max()) > 1000, f"peak {int(np.abs(a).max())}")
nz = np.nonzero(np.abs(a) > 32)[0]
check("front pad trimmed", nz.size > 0 and nz[0] / rate * 1000 < 80,
      f"{nz[0]/rate*1000:.0f} ms lead")

print("\n== streaming generator contract ==")
gen = prov.stream("Streaming works when the engine is warm.", voice="Savannah")
check("is a generator", hasattr(gen, "__next__"))
total = b"".join(gen)
check("int16-aligned", len(total) > 0 and len(total) % 2 == 0, f"{len(total)} bytes")
check("rate set to engine's", prov.stream_sample_rate == 44100, str(prov.stream_sample_rate))
check("audio real", len(total) > 20000, f"{len(total)//2} samples")
check("empty -> no chunks", list(prov.stream("", voice="Savannah")) == [])

print("\n== prosody + emotion are level-safe ==")
ctrl = m.emotion_controls("happy", 2.0)
check("emotion cap applied",
      all(abs(v) <= 0.9 for k, v in ctrl.items() if k.startswith("emotions.")), str(ctrl))
check("no mood coords in presets",
      not any(k.startswith("mood.") for c in m.EMOTION_CONTROLS.values() for k in c))
check("expressive off by default", m._DEFAULT_EXPRESSIVE == 0.0)
check("neutral sends no controls", m.emotion_controls("neutral", 1.0) == {})

print("\n== engine pid reporting ==")
engine_pid = sup._engine_pid()
check("engine pid is the SERVER's, not this host's",
      engine_pid > 0 and engine_pid != os.getpid(),
      f"engine={engine_pid} host={os.getpid()}")
rec = json.loads(sup._owner_file().read_text(encoding="utf-8"))
check("owner record carries both pids",
      rec.get("owner_pid") == os.getpid() and rec.get("engine_pid") == engine_pid, str(rec))
st = api.status()
check("/status reports loaded with the engine pid",
      st.get("tts") == "warm" and st.get("pid") == engine_pid,
      f"{st.get('tts')}/{st.get('pid')}")

print("\n== a live non-engine pid must not read as a resident engine ==")
fake = api._state_path().parent / f"_zreg_fake_{os.getpid()}.json"
fake.write_text(json.dumps({"agent_pid": os.getpid(), "tts": "warm",
                            "pid": os.getpid(), "updated_at": time.time(),
                            "tts_leased": 0, "unload_after": 600}), encoding="utf-8")
try:
    # Point the reader at only this file so the live writers cannot mask it.
    real = api._state_path
    api._state_path = lambda: fake
    seen = api.status()
    check("host pid is not treated as an engine", seen.get("pid") in (0, engine_pid),
          f"pid={seen.get('pid')}")
finally:
    api._state_path = real
    fake.unlink(missing_ok=True)

print("\n== lease + unload lifecycle ==")
check("spawner may stop its own engine", sup._can_stop_engine())
sup._held = False
sup.release()
check("release() arms an idle timer with no lease", sup._idle_timer is not None)
sup._cancel_timer_locked()

print("\n== dead owner must not block unloading ==")
rec = json.loads(sup._owner_file().read_text(encoding="utf-8"))
rec["owner_pid"] = 999999
sup._owner_file().write_text(json.dumps(rec), encoding="utf-8")
saved_proc = sup._proc
sup._proc = None
try:
    check("a dead owner does not wedge unload", sup._can_stop_engine())
finally:
    sup._proc = saved_proc

print("\n== idle unload actually frees the engine ==")
sup.unload_after = 2.0
sup._held = False
sup._cancel_timer_locked()
sup.arm_if_orphaned()
for _ in range(40):
    if not api._engine_running():
        break
    time.sleep(0.5)
check("engine unloaded on its own", not api._engine_running())
st = api.status()
check("/status reports unloaded", st.get("tts") == "unloaded" and not st.get("pid"),
      f"{st.get('tts')}/{st.get('pid')}")

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("PASS no regressions")
