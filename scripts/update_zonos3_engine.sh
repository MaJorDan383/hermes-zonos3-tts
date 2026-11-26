#!/usr/bin/env bash
# Update + rebuild the ZONOS3-Mini engine, safely.
#
# WHY A SCRIPT AND NOT A BUTTON: the engine is a compiled CUDA binary. An update
# is `git pull` + a multi-minute rebuild, and if the rebuild fails you are left
# with no working zonos3-server.exe -- which is the whole voice path. So this
# builds into a SEPARATE directory and only swaps the binary in after the build
# succeeds AND the new binary answers a health probe. A failure leaves the old
# binary untouched.
#
# Usage:
#   bash update_zonos3_engine.sh            # check + update + rebuild + swap
#   bash update_zonos3_engine.sh --check    # report only, change nothing
#   bash update_zonos3_engine.sh --rollback # restore the previous binary
#
# Exit codes: 0 ok/no-op, 1 failed (old binary kept), 2 nothing to do.
set -uo pipefail

REPO="C:/Users/MaJor/Github/ZONOS3-Mini"
SRC="$REPO"
BUILD="$SRC/servers/ggml/build-cuda"
NEWBUILD="$SRC/servers/ggml/build-cuda-next"
BIN="$BUILD/zonos3-server.exe"
BAK="$BUILD/zonos3-server.exe.prev"
MODEL="C:/Users/MaJor/AppData/Local/hermes/models/zonos3-mini"
PORT=19191           # probe on an unused port, never 1919 (the live engine)
# Per-run log: a single shared file meant an innocuous --check truncated a
# running build's log, which is how a 20-minute build was misread as stalled.
LOGDIR="C:/Users/MaJor/AppData/Local/hermes/cache/scratch"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/engine_update_$(date +%Y%m%d-%H%M%S).log"

MODE="update"
case "${1:-}" in
  --check) MODE="check" ;;
  --rollback) MODE="rollback" ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
esac

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

say "=== zonos3 engine update  $(date -Iseconds)  mode=$MODE ==="

if [ "$MODE" = "rollback" ]; then
  if [ ! -f "$BAK" ]; then say "no backup at $BAK"; exit 1; fi
  cp -f "$BAK" "$BIN" && say "restored previous binary"
  exit 0
fi

# ---------------------------------------------------------------- prerequisites
for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || { say "MISSING: $tool"; exit 1; }
done
[ -d "$SRC/.git" ] || { say "not a git checkout: $SRC"; exit 1; }

BEFORE=$(git -C "$SRC" rev-parse --short HEAD)
BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)
say "current: $BEFORE on $BRANCH"

git -C "$SRC" fetch origin --quiet || { say "fetch failed (offline?)"; exit 1; }
REMOTE=$(git -C "$SRC" rev-parse --short "origin/$BRANCH")
BEHIND=$(git -C "$SRC" rev-list --count "HEAD..origin/$BRANCH")
say "remote : $REMOTE   behind: $BEHIND commit(s)"

if [ "${FORCE_REBUILD:-0}" = "0" ] && [ "$BEHIND" = "0" ]; then
  say "already up to date -- nothing to do"
  exit 2
fi

say ""
say "incoming:"
git -C "$SRC" log --oneline "HEAD..origin/$BRANCH" | sed 's/^/  /' | tee -a "$LOG"

if [ "$MODE" = "check" ]; then
  say ""
  say "--check: not changing anything"
  exit 0
fi

# Refuse while the engine is actually running: swapping the binary under a live
# process is fine on Linux, not on Windows, where the file is locked.
if netstat -ano 2>/dev/null | grep -qE ":1919[[:space:]].*LISTENING"; then
  say ""
  say "The engine is running. Unload it first (chip -> Unload model now) and re-run."
  exit 1
fi

# ---------------------------------------------------------------- pull
say ""
say "--- pulling ---"
git -C "$SRC" stash list | grep -q . && say "WARNING: local stashes exist; not touching them"
if ! git -C "$SRC" merge --ff-only "origin/$BRANCH" >>"$LOG" 2>&1; then
  say "fast-forward failed. Local changes? Resolve manually -- nothing was rebuilt."
  git -C "$SRC" merge --abort >>"$LOG" 2>&1
  exit 1
fi
AFTER=$(git -C "$SRC" rev-parse --short HEAD)
say "now at: $AFTER"

# ---------------------------------------------------------------- build
# Reuse the flags that produced the working binary (read from the live cache) so
# a rebuild cannot silently drop CUDA or change the arch.
ARCH=$(grep -E "^CMAKE_CUDA_ARCHITECTURES:" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
ARCH=${ARCH:-89}
# BUILD_SHARED_LIBS defaults to ON, which produces a 17 MB exe plus four DLLs
# (ggml.dll, ggml-cuda.dll, ...). The plugin spawns the exe from build-cuda/ with
# no DLL search path, so a shared build cannot run there at all -- and it fails
# with a bogus "api-ms-win-crt-locale-l1-1-0.dll missing" message instead of
# naming the real problem. The shipped build is fully static (75 MB, no DLLs),
# so mirror whatever the working cache uses.
SHARED=$(grep -E "^BUILD_SHARED_LIBS:" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
SHARED=${SHARED:-OFF}
NVCC=$(grep -E "^CMAKE_CUDA_COMPILER:" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
say ""
say "--- configuring (CUDA arch $ARCH, Ninja, into a fresh dir) ---"
rm -rf "$NEWBUILD"

# MSVC is REQUIRED and git-bash cannot load vcvars64.bat into its own
# environment, so the configure+build runs inside cmd.exe with vcvars applied.
# Without this, cmake reports "No CMAKE_C_COMPILER could be found" -- the exact
# failure this script hit on its first real run.
CC_PATH=$(grep -E "^CMAKE_C_COMPILER:" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
VCVARS="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
if [ ! -f "$VCVARS" ]; then
  say "vcvars64.bat not found at $VCVARS"
  say "Set VCVARS in this script to your Visual Studio Build Tools location."
  exit 1
fi
say "toolchain: MSVC ($(basename "${CC_PATH:-cl.exe}")) + Ninja"

WIN_SRC=$(cygpath -w "$SRC/servers/ggml" 2>/dev/null || echo "$SRC/servers/ggml")
WIN_BUILD=$(cygpath -w "$NEWBUILD" 2>/dev/null || echo "$NEWBUILD")
WIN_NVCC=$(cygpath -w "$NVCC" 2>/dev/null || echo "$NVCC")

BUILD_CMD="call \"$VCVARS\" >nul && cmake -S \"$WIN_SRC\" -B \"$WIN_BUILD\" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH"
[ -n "$NVCC" ] && BUILD_CMD="$BUILD_CMD -DCMAKE_CUDA_COMPILER=\"$WIN_NVCC\""
BUILD_CMD="$BUILD_CMD && cmake --build \"$WIN_BUILD\" --target zonos3-server -j"

say "--- configuring + building (this takes minutes) ---"
# Write a .bat and run it, rather than `cmd //c "<long string>"`: MSYS rewrites
# the argument list of a direct cmd invocation, which silently produced an
# interactive shell that ran nothing and reported success.
BAT="$BUILD/NOPE"; BAT="$(dirname "$LOG")/build_zonos3.bat"
{
  echo "@echo off"
  echo "call \"$(cygpath -w "$VCVARS" 2>/dev/null || echo "$VCVARS")\""
  echo "if errorlevel 1 exit /b 1"
  echo "cmake -S \"$WIN_SRC\" -B \"$WIN_BUILD\" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=$SHARED -DCMAKE_CUDA_ARCHITECTURES=$ARCH$([ -n "$NVCC" ] && echo " -DCMAKE_CUDA_COMPILER=\"$WIN_NVCC\"")"
  echo "if errorlevel 1 exit /b 1"
  echo "cmake --build \"$WIN_BUILD\" --target zonos3-server -j"
  echo "exit /b %errorlevel%"
} > "$BAT"

if ! MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$BAT" 2>/dev/null || echo "$BAT")" >>"$LOG" 2>&1; then
  say "BUILD FAILED -- old binary untouched. Tail:"
  grep -iE "error|CMake Error" "$LOG" | tail -15 | sed 's/^/  /'
  exit 1
fi
NEWBIN="$NEWBUILD/zonos3-server.exe"
[ -f "$NEWBIN" ] || { say "build reported success but no binary at $NEWBIN"; exit 1; }
NEWSIZE=$(stat -c %s "$NEWBIN" 2>/dev/null || wc -c <"$NEWBIN")
say "built: $NEWSIZE bytes"

# A shared build links against sibling DLLs the plugin cannot find, and reports it
# as a missing CRT DLL -- a misleading error. Detect the shape directly instead of
# letting the probe fail opaquely.
DLLS=$(find "$NEWBUILD" -maxdepth 2 -name "*.dll" 2>/dev/null | wc -l)
OLDSIZE=$(stat -c %s "$BIN" 2>/dev/null || echo 0)
if [ "$DLLS" -gt 0 ]; then
  say "REFUSING: the build produced $DLLS DLL(s); this needs a STATIC build"
  say "  (put -DBUILD_SHARED_LIBS=OFF in the cmake line, or the exe cannot find them)"
  exit 1
fi
if [ "$OLDSIZE" -gt 0 ] && [ "$NEWSIZE" -lt $((OLDSIZE / 2)) ]; then
  say "REFUSING: new binary is $NEWSIZE bytes vs $OLDSIZE -- far too small for a static build"
  exit 1
fi

# ---------------------------------------------------------------- smoke test
# A binary that builds but cannot load the weights is worse than the old one.
say "--- probing the new binary on :$PORT ---"
( cd "$NEWBUILD" && "$NEWBIN" "$MODEL/backbone-ptq.gguf" --port "$PORT" --host 127.0.0.1 \
    --spk-encoder "$MODEL/spk-encoder.gguf" --tts-default-voices-dir "$MODEL/voices" \
    --gpu ) >>"$LOG" 2>&1 &
PROBE_PID=$!
OK=0
for _ in $(seq 1 60); do
  sleep 1
  if netstat -ano 2>/dev/null | grep -qE ":$PORT[[:space:]].*LISTENING"; then OK=1; break; fi
  kill -0 "$PROBE_PID" 2>/dev/null || break      # died early
done
if [ "$OK" = "1" ]; then
  curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v2/health" >>"$LOG" 2>&1 \
    && say "health probe ok"
fi
kill "$PROBE_PID" 2>/dev/null
# On Windows the process may need a harder kill.
powershell -NoProfile -Command "Get-Process -Id $PROBE_PID -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1
sleep 1

if [ "$OK" != "1" ]; then
  say "NEW BINARY DID NOT START -- old binary untouched. Tail:"
  tail -25 "$LOG" | sed 's/^/  /'
  exit 1
fi

# ---------------------------------------------------------------- swap
say "--- swapping binary ---"
cp -f "$BIN" "$BAK" 2>/dev/null && say "backup kept at $BAK"
if cp -f "$NEWBIN" "$BIN"; then
  say "swapped in $AFTER"
  say "rollback with: bash $0 --rollback"
else
  say "COPY FAILED -- restoring backup"
  cp -f "$BAK" "$BIN"
  exit 1
fi

say ""
say "done. The engine will spawn the new binary on next use."
exit 0
