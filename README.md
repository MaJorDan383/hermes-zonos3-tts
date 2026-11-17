# zonos3-tts — ZONOS3-Mini for Hermes

Local text-to-speech from Zyphra's **ZONOS3-Mini** (886M total / 100M active
parameters, Apache-2.0), served by the native ggml/CUDA runtime and registered
as a Hermes TTS provider named `zonos3`.

- 44.1 kHz mono, 43 languages, zero-shot voice cloning
- **~0.8 s** per spoken sentence on an RTX 4090 (RTF ≈ 0.19); backbone at 376 frames/s
- Lease-driven: warms on the first Read-aloud/mic lease, **unloads after an idle window**
- **Emotion steering** wired for SenseVoice: `set_emotion("happy")` applies real prosody

## Why this engine

Two ZONOS3 runtimes exist. Both were built and measured on this host
(Windows 11, RTX 4090, CUDA 13.5):

| | Zyphra `servers/ggml` | `audio.cpp` port |
|---|---|---|
| Silent-output rate | **0/12** | 4/10 |
| Backbone throughput | **376 frames/s** | ~120 frames/s |
| Emotion / steering bundle | **yes** (`emotions.*`, `mood.*`, `delivery.*`) | not implemented |
| Named default voices | **yes** (`/v2/speakers`) | request options only |
| Model unload | none — process only | `--idle-unload-ms` |

Zyphra's runtime wins on correctness and capability. The only thing it lacks is an
idle unload, which this plugin supplies by owning the server as a child process:
`stop_process()` terminates it, returning **all** VRAM (weights + KV cache +
decoder + CUDA context). That is *more* than audio.cpp's model-only unload.

`servers/sgl` (Mini-SGLang) and `servers/mlx` are not options here — SGL is
Linux-only (TileLang kernels) and MLX is Apple Silicon only.

## Install

**1. Engine binary.** Build once from Zyphra's repo (needs VS 2022 C++ tools,
CUDA toolkit, CMake, and a working Rust toolchain for the text normalizer):

```bash
git clone --depth 1 https://github.com/Zyphra/ZONOS3-Mini
cd ZONOS3-Mini
git submodule update --init servers/ggml/ggml third_party/zyphra-textnorm \
    third_party/fusealign-multilingual
cd servers/ggml
cmake -B build-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-cuda -j 16
```

`CMAKE_CUDA_ARCHITECTURES=89` is sm_89 (RTX 4090). Use `86` for Ampere, `90` for
Hopper, or omit it to inherit the default (which targets sm_90 only — wrong for a
consumer card).

If the build fails with
`the 'cargo.exe' binary ... is not applicable to the 'stable-...' toolchain`, the
Rust toolchain is corrupt (its directory holds only clippy). Repair it:

```bash
rustup toolchain uninstall stable-x86_64-pc-windows-msvc
rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal \
       --component cargo,rustc
```

**2. Weights** (~1.2 GB, into `$HERMES_HOME/models/zonos3-mini/`):

```bash
hf download Zyphra/ZONOS3-Mini \
   gguf/backbone-ptq.gguf gguf/zac-keep2.gguf gguf/spk-encoder.gguf gguf/steer.gguf \
   --local-dir "$HERMES_HOME/models/zonos3-mini"
hf download Zyphra/ZONOS3-Mini --include 'gguf/voices/*' \
   --local-dir "$HERMES_HOME/models/zonos3-mini/tmp" \
   && mv "$HERMES_HOME/models/zonos3-mini/tmp/gguf/voices" \
         "$HERMES_HOME/models/zonos3-mini/" && rm -rf "$HERMES_HOME/models/zonos3-mini/tmp"
```

`hf download` mirrors the repo layout, so files land under a nested `gguf/`
directory — flatten them. `steer.gguf` is optional but **required for emotion
steering**; without it the server returns
`400 ... this server has no steering bundle` for any `controls` request.

The default voices are precomputed embeddings, so cloning them needs **no
ffmpeg and no speaker encoder** at runtime.

**3. Register.**

```bash
hermes plugins enable zonos3-tts
hermes config set tts.provider zonos3
hermes config set plugins.entries.zonos3-tts.settings.voice Savannah
hermes config set plugins.entries.zonos3-tts.settings.unload_after_seconds 300
```

## Settings

| Key | Default | Meaning |
|---|---|---|
| `voice` | — | Default voice: a name (`Savannah`, matched against `label`/filename), or a `default_…` id |
| `unload_after_seconds` | `300` | Idle window before the server process is stopped. `0` = stop as soon as the last lease releases |
| `emotion_aware` | `true` | Apply a `set_emotion()` value to requests |
| `language` | `en_us` | Text-normalization grammar (`en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko`) |
| `retry_attempts` | `3` | Attempts before giving up on silent audio |
| `runtime_dir` | auto | Directory holding `zonos3-server`; auto-detects the plugin's `runtime/` then the build checkout |
| `model_dir` | `$HERMES_HOME/models/zonos3-mini` | GGUF directory |
| `server_exe` | — | Explicit path to the server binary |
| `host` / `port` / `base_url` | `127.0.0.1` / `1919` | Where the server listens |
| `gpu` | `true` | Pass `--gpu` (CUDA). Set `false` for the CPU backend |
| `start_timeout_seconds` | `120` | Startup ceiling |
| `request_timeout_seconds` | `300` | Per-request ceiling |
| `dynamic_range` | `1.5` | `pitch.pitch_range` — F0 spread width. Safe band ±2; +2 nearly doubles it |
| `expressive` | `4.0` | `expressive.expressive` — liveliness. Safe band ±8 |
| `volume_boost` | `2.0` | `delivery.loud` — the engine's only real gain. Safe band 0–4; 4 is clearly louder |
| `streaming` | `true` | Opt into Hermes' raw-PCM streaming path (native 44100 Hz, float32→int16 in the client) |
| `steer_hold` | `0.35` | Steer-then-hold: dose multiplier after the onset. `1` = constant push (old behaviour). **Only sent when the live bundle advertises `steer_release`** |
| `steer_onset_seconds` | `4.0` | Seconds at full dose before the ramp begins |
| `steer_ramp_seconds` | `1.0` | Linear ramp length from full dose to `steer_hold` |
| `headroom_db` | unset | Engine output gain. Unset keeps the server default (3 dB attenuation); `0` recovers that 3 dB for free, measured +3.0 dB peak and RMS. Below 0 is a 422 |

**Every request object is CLOSED.** An unknown field is a **422**, not a no-op —
measured, even `speed: 1.0` and `tag_cfg_scale: 1.0` are rejected. The plugin
therefore drops any field the engine schema does not list (`_ENGINE_FIELDS`)
instead of passing caller kwargs straight through.

Every setting except the ones read once at `register()` is re-read live, so a
status-chip change applies on the next release without restarting Hermes.
`dynamic_range`, `expressive`, `volume_boost` and `streaming` **are** read once at
`register()` — change those with `hermes config set` and restart Hermes.

## Residency model

Hermes holds TTS *leases* (`desktop:read-aloud`, `cli:voice`, the desktop mic).
This plugin reacts to them; it does not invent its own policy:

| Event | Effect |
|---|---|
| first `warm()` (a surface turns speech on) | start the server, wait for `/v2/health` (~2 s) |
| `release()` from the last holder | arm a timer for `unload_after_seconds` |
| timer fires with no holder | `stop_process()` — terminate, verify down, report `unloaded` |
| a lease arrives while stopped | start again; the next request just works |
| synthesis succeeds | `state = warm` (evidence, not intent) |

`warm()` is called on **every** lease acquire but `release()` only fires once, when
the last holder leaves — Hermes treats leases as a set. The plugin therefore keeps
a boolean, not a counter; a counter would leak one reference per re-acquire and the
unload would never fire.

States: `idle` → `starting` → `warm` → `unloading` → `unloaded`, plus `error`.

## Emotion: SenseVoice → stance → reply voice

Three stages, and they are deliberately different mechanisms because they solve
different problems.

```
 mic audio
    │
    ▼
 SenseVoice (int8 ONNX, CPU)  ──►  emotion + intensity + audio events
    │
    ├─►  pre_llm_call hook  ──►  "how to answer this person"  (text/prompt)
    │
    └─►  ZONOS3 controls    ──►  "how the answer sounds"      (prosody)
```

### 1. Detection — no new download

The plugin reuses the **int8 ONNX SenseVoice already on this machine**:

```
~/.dsh/speech-to-text/sensevoice/models/sensevoice-onnx/
    model.int8.onnx   (239 MB)
    tokens.txt
```

It is *referenced*, not copied — that is the whole point of using this build. The
893 MB PyTorch checkpoint in Voxta's model folder is **not** used: it needs
`torch` + `funasr` (`kaldi_native_fbank`, `torchaudio`) at runtime, and none of
those are installed. The ONNX export takes the raw 560-dim fbank features
directly, and `onnxruntime` is already a Hermes dependency, so
`pip_dependencies` stays empty and **nothing is downloaded**.

> **Do you need a separate SenseVoice download?** No. Verified working on the
> existing copy: 6/6 real voice clips transcribed correctly and produced
> *different* emotions (happy / angry / neutral). If that copy is ever removed,
> the model comes from `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`
> (int8 ONNX ≈ 234 MB) — but there is no reason to fetch it now.

#### Running it without Voxta

Set `stt_enabled: true` and `stt.provider: sensevoice` and this plugin is the
speech-recognition engine too: one SenseVoice pass returns the transcript *and*
the affect, so Voxta (and its 4–12 GB CUDA torch install) is not needed at all
for voice input. Measured on this host, same audio, 3 s clip:

| path | warm latency | notes |
|---|---|---|
| Voxta (`/api/transcription`, fp32 on CUDA) | ~350–510 ms | 11.8 s cold start, needs the server running |
| this plugin, `sensevoice_engine: onnx` (default) | ~100–180 ms | no extra dependency, reports intensity |
| this plugin, `sensevoice_engine: sherpa` | ~105 ms infer | needs `pip install sherpa-onnx` (28 MB) |

Accuracy is the same or better in places: Voxta's fp32 run transcribed a Chinese
sample as `开放9至5.` (truncated, non-deterministic across runs) where this path
returns `开饭时间早上9点至下午5点。`. Both engines miss the same word on one
English clip (`gold` → `code`), which is int8 quantization, not the frontend.

> **`text_norm` is a token id, not a flag.** The exported graph resolves
> `language` and `text_norm` through an embedding table. It must be `14`
> (`<|withitn|>`) for digits, punctuation and capitalisation; `1` selects
> `<|woitn|>` and silently returns `fifty pieces of code` instead of
> `50 pieces of code.`, and out-of-range values crash inside the graph's Gather.
> The old code passed `1`, which is why punctuation was missing.

Emotion is not a second model. SenseVoice emits **control tokens inside its own
CTC output** — `<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`,
`<|FEARFUL|>`, `<|DISGUSTED|>`, `<|SURPRISED|>` — plus audio events
(`<|Laughter|>`, `<|Cry|>`, `<|Applause|>`, `<|Cough|>` …). One forward pass gives
transcript *and* affect.

**Intensity** is the CTC posterior on the winning emotion token, mapped from the
measured 0.45–0.95 band onto 0–1. It is a heuristic, not a calibrated score: a
decisive token reads as a strongly expressed feeling. Measured posteriors across
real clips ran 0.49–0.93, and an earlier fixed divisor pinned everything at 1.0 —
which is why the band is explicit in the code.

### 2. Response stance — the model's turn

This is where *sympathetic, rational, productive* is actually implemented, and
it is a prompt problem, not an audio one. `pre_llm_call` appends one line to the
user turn:

```
[heard: angry 0.8 events=Cry] The user's speech sounded strongly angry. Name the
frustration plainly and briefly, without matching its heat; do not be defensive,
do not over-apologise, do not explain at length before fixing; lead with the
concrete fix or the one decision that unblocks it.
```

Every emotion has an `acknowledge` / `avoid` / `do` triple in
`emotion.RESPONSE_GUIDANCE` — sympathy without the action is hollow, action
without the acknowledgement reads as cold. The hook returns `{"context": ...}`,
which the host appends to the **user message**, never the system prompt, so it
is turn-local and cannot leak into later turns. Neutral emits nothing at all: an
instruction for a flat delivery just makes the model perform feelings the user
did not express.

The stance fires **once per utterance** — a second `pre_llm_call` in the same
turn returns `None`, so a multi-step turn is not re-primed from a stale reading.
A detection older than 120 s is ignored, so a typed message ten minutes later is
not coloured by an old voice turn.

### 3. Reply voice — ZONOS3's turn

The reply is *not* steered to mirror the user. Mirroring anger escalates and
mirroring sadness drags the conversation down, so the mapping is **meet, then
settle** — start near their affect, land on composure:

| user | reply steered toward | dose | tempo |
|---|---|---|---|
| angry | `emotions.serene` | 0.7 | 0.95 |
| frustrated | `emotions.content` | 0.5 | 1.00 |
| sad | `emotions.content` | 0.35 | 1.00 |
| fearful | `emotions.serene` | 0.8 | 0.95 |
| happy | `emotions.happy` | 0.4 | 1.05 |
| excited | `emotions.happy` | 0.5 | 1.08 |
| surprised | `emotions.interested` | 0.4 | 1.02 |
| disgusted | `emotions.serene` | 0.4 | 1.00 |
| neutral | — | — | — |

Doses are scaled by the user's intensity (`0.4 + 0.6 × intensity`), so mild
irritation gets a light touch. All four direction names were verified against the
live `GET /v2/controls` bundle; `serene`/`content`/`interested` are used because
they read as composed rather than blank.

### Automatic detection — the `pre_transcription` hook

Detection runs **by itself** on every recorded utterance. `pre_transcription`
fires with the recorded file *before any STT backend runs*, so it sees the exact
audio that is about to be transcribed:

```python
def on_pre_transcription(self, **kwargs):
    path = kwargs.get("file_path")      # the recorded utterance
    threading.Thread(target=lambda: self.observe_audio(path), daemon=True).start()
    return None                          # observe only — STT still produces the text
```

This is why the affect never needs a second process in the loop. The hook is a
Hermes-native extension point that fires on the desktop mic, CLI voice mode and
the gateway alike, so it covers every surface without patching a host call site.


Analysis runs on a daemon thread: the user is waiting for their transcript, and
blocking STT dispatch to detect a mood would be the wrong trade. Detection takes
~0.3 s for several seconds of audio, so in practice it lands before the reply is
requested. If it ever lost that race, the turn simply has no affect tag — never a
broken turn.

Set `emotion_auto: false` to go back to manual (`sensevoice_emotion` tool) or to
drive it yourself from an external front-end.

### API

```python
pipe = ctx._zonos3_emotion                 # exposed by register()

pipe.observe_audio("utterance.wav")        # detect; arms the reply voice
pipe.on_pre_llm_call()                     # what the hook returns for this turn
pipe.snapshot()                            # emotion, intensity, events, age

# or drive the voice directly, without STT:
prov = ctx._zonos3_provider
prov.set_emotion("sad", 0.6)               # steer toward sadness
prov.set_emotion_controls({"emotions.serene": 0.6})   # explicit prosody
prov.clear_emotion()
```

Two tools are registered so a voice surface can feed recordings in:
`sensevoice_emotion` (analyse a file, arm the reply voice, `clear=true` to reset)
and `voice_emotion_state` (read the pending affect).

### Settings

| Key | Default | Meaning |
|---|---|---|
| `emotion_stt` | `true` | Run SenseVoice detection at all |
| `emotion_annotate` | `true` | Inject the stance line into the user turn |
| `emotion_reply_prosody` | `true` | Steer the reply's voice from the detected affect |
| `emotion_auto` | `true` | Detect affect from the `pre_transcription` hook (no manual call needed) |
| `ducking_enabled` | `true` | Lower other apps' volume while Hermes speaks |
| `ducking_factor` | `0.20` | Relative volume for other apps during speech (−14 dB) |
| `ducking_watch` | `hermes.exe` | Process(es) whose audio activity triggers ducking |
| `ducking_hold_off_seconds` | `1.2` | Keep ducking this long after speech stops |
| `sensevoice_model_dir` | auto | Override the ONNX model directory |
| `sensevoice_engine` | `onnx` | `onnx` (no deps, reports intensity) or `sherpa` (needs `sherpa-onnx`, label only) |
| `sensevoice_threads` | `2` | Decoding threads; 2 measured fastest on this i7-12700K |
| `stt_enabled` | `false` | Register the `sensevoice` STT provider (set `stt.provider: sensevoice` to use it) |

### Audio ducking (Windows)

Raising the file's loudness stops helping once it is already louder than the
reference material on the machine — measured here, the user's own voice library sits
at −22 LUFS while the normalised output is −16. What actually helps is not
*competing* with the rest of the system.

`ducking.py` lowers every other app's volume to `ducking_factor` while Hermes is
playing audio, and restores it afterwards, using WASAPI per-session volume through
`pycaw`.

**The trigger is session state, not a Python hook.** The first version hooked
`tools.voice_mode.mark_audio_output_active`, which looked like the right seam —
every Python playback path brackets it. It never fired: the desktop app plays TTS
in its **Electron renderer** (`new Audio(...)` and a WebSocket `AudioContext`), so
those Python helpers never run. The symptom was `duck_count: 0` through many
utterances while everything "looked" wired.

The reliable signal is WASAPI's own `State`: `Active` (1) while a process produces
audio, `Inactive` (0) when it stops. A poller (200 ms idle / 120 ms active) watches
Hermes' session and opens/restores the duck around those transitions, which works
for every playback path — renderer, CLI speaker, streamed PCM — with no host
patching. Verified: `ffplay` 100% → 20% → back to exactly 100%, with the speaking
process never attenuated.

Why not Windows' built-in ducking? Sound ▸ Communications ducks on *call* activity
and only to a fixed level; on this host `UserDuckingPreference` was `0x3` ("do
nothing"), i.e. disabled. Per-session control is both more precise and reversible.

Safety rules, because this is a global side effect on other applications:

- each ducked session records its **exact** prior volume and is restored to that,
  never to an assumed 100%;
- **Hermes itself and the watched process are never ducked** — the watch list and
  the skip list are separate, because watching one app must not make another
  duckable;
- a short hold-off (1.2 s) after speech stops prevents flapping between the
  sentences of one reply;
- a 300 s watchdog restores everything even if the stop transition is never
  observed (crashed renderer, killed player);
- it fails open — no `pycaw`/COM means speech plays at normal level and nothing
  else changes.

Settings: `ducking_enabled` (`true`), `ducking_factor` (`0.20` ≈ −14 dB),
`ducking_watch` (`hermes.exe`, comma-separated), `ducking_hold_off_seconds` (`1.2`).
`pycaw`/`comtypes` are **not** declared in `pip_dependencies` — they are optional
and Windows-only; install with `pip install pycaw comtypes` if ducking is wanted.

**Measured on this host** (2026-11-17), with `pycaw 20260927` + `comtypes 1.5.0`
installed into the gateway's own interpreter
(`$HERMES_HOME/tools/python-*/python.exe`, not `python` on PATH — installing
elsewhere leaves the gateway unable to import it and ducking silently reports
`unavailable`): an `ffplay` session at **100%** went to **20%** during `duck()`
and back to exactly **100%** on `restore()`, with 2 sessions ducked/restored.

Install into the interpreter the gateway actually runs — a plain
`pip install pycaw` often targets a different Python. Verify with:

```
<gateway python> -c "from pycaw.pycaw import AudioUtilities; print(len(AudioUtilities.GetAllSessions()))"
```

The gateway must **restart** afterwards; `register()` constructs the ducker once,
so a fresh install is invisible to an already-running process.

### Audio level: the engine's own output, with steering-based lift

The provider writes the engine's WAV **byte-for-byte** — verified: a synthesis and
a direct `POST /v2/generate` for the same request produced identical bytes
(329,324 = 329,324). No gain stage, no re-encode, no ffmpeg.

There was briefly an EBU R128 normalisation step here. It worked (7.9 dB spread
between voices reduced to 0.1 dB, measured 0.04 dB against ffmpeg's `ebur128`),
but it was **removed** on request: the voice was not actually quiet relative to
anything else on the machine — measured, the user's own voice library sits at
−22 LUFS and Windows system sounds at −22.7, so the raw engine output was already
in the normal range. The perceived problem was competition with other audio, which
ducking solves directly and without touching the signal. `loudness.py` was deleted
rather than left dormant.

Amplification now comes from the engine's **own** steering instead, which keeps
the signal path untouched:

| Setting | Direction | Measured effect |
|---|---|---|
| `volume_boost` | `delivery.loud` | ×4 → **+4.7 dB peak / +6.8 dB RMS** (×1 and ×2 do little) |
| `dynamic_range` | `pitch.pitch_range` | +2 → F0 spread **3.17 → 5.93 st** |
| `expressive` | `expressive.expressive` | +16 → F0 spread 3.17 → 4.25 st |

`mood.loudness` is **not** a gain — measured −2.2 dB peak / +0.6 dB RMS. It changes
character, not level.

With the defaults (`dynamic_range 1.5`, `expressive 4`, `volume_boost 2`) a real
A/B on one sentence measured **+10.4 dB peak, +13.8 dB RMS, +57% F0 spread**.

`pitch.pitch_range` re-projects the speaker embedding, so the server refuses it
with a 400 unless `speaker_mode=embedding`. The plugin sets that mode
automatically for bundled voices; a reference-clip voice skips the range control
(its codes cannot be re-projected) while keeping expressive/loud.

Still useful if you ever want it back: the engine attenuates by
`headroom_db: 3` on every request, so `request_options: {"headroom_db": 0}`
recovers that 3 dB for free with no post-processing.

### Long text: chunked, because the engine truncates and then crashes

The engine caps generation at `max_tokens` frames — 2000 on this release at 15 fps,
so **~133 s per request** — and it does **not** report hitting the cap. Measured on
this host:

| Words | Result |
|---|---|
| 116–464 | under the cap, spoken in full |
| 522–638 | **132.7 s returned** — silently truncated at the cap |
| 754+ | **connection reset, engine process died** |

So `synthesize()` splits text at sentence boundaries (then `,;:`, then whitespace)
against a character budget derived from the live `max_tokens`, speaks each piece,
and splices the PCM into one WAV with rewritten sizes. Nothing is dropped and the
header stays honest. Before this, a long reply was either cut short or killed the
engine mid-session.

### Steer then hold: the bundle lacks it, so the plugin approximates it

Zyphra's guide documents `steer_release` (`{onset_seconds, ramp_seconds, hold,
exempt}`): request-level hidden `controls` run at full dose for `onset_seconds`,
ramp linearly to `hold × dose`, then stay. A constant push integrates over a long
line — the model copies the style of speech it already produced — so stacked
liveliness + valence + arousal climb towards shouting. Their measurement: a stacked
voice design held constant climbed in **19 of 24 takes** (+7.2 st, +7.0 dB), while
hold 0.35 gave +0.6 st / +5.2 dB with neutral's climb count.

**This bundle predates the block:** `GET /v2/capabilities` and `/v2/controls` both
report `steer_release: null`, and `steer_hold < 1` without it is a hard 400. So the
engine cannot ramp the push for us.

`steer_scale()` therefore applies the same curve **client-side, per request**: it
estimates the utterance's duration (~14 chars/second, measured), and once past
onset + ramp it scales every prosody control down to `steer_hold`. A short reply
finishes before the onset and is untouched; a long one stops accumulating pitch and
loudness. When a future bundle carries `steer_release`, the plugin sends the real
fields instead and the estimate stops mattering.

Two further doc features are **unavailable on this build**, both capability-gated:
reading-position span steering and `control_spans` emphasis need `laci_languages`
(reported as `[]`), and `short_text` (the 1–2-word filler trick) needs
`pause_tokens` (reported as `null`). Both were checked, not assumed.

### Streaming: native rate, no resampling

The plugin opts into Hermes' raw-PCM path (`streams_pcm`, setting
`streaming: true`), so a streamed reply starts speaking on the first sentence
instead of after the whole text is synthesised.

Two things the engine does that the streaming path must handle, both verified
against the live server:

- **It streams float32, not int16.** The chunked response carries
  `X-Audio-Format: float32` with `Content-Type: audio/pcm`. Handing those bytes
  to a consumer expecting int16 produces loud white noise, not silence. The
  client unpacks float32, clamps to [−1, 1], and emits int16 little-endian.
- **It runs at 44100 Hz, not Hermes' 24000 default.** The rate is read from
  `X-Audio-Sample-Rate` before the first chunk is yielded, so the consumer opens
  its device at the engine's real rate. No resampling happens anywhere; the
  buffered WAV path also carries 44100 Hz in its header.

Neither format nor rate is guessed: a stream without `X-Audio-Sample-Rate` raises
rather than playing at a default.

### The real ffmpeg steps (host-side, not the plugin)

For the record, since "one less ffmpeg step" is a reasonable goal:

| Step | Where | Needed? |
|---|---|---|
| WAV → Opus for the voice bubble | host (`tts_tool_delivery.py`), because `voice_compatible = True` | yes, for `.ogg` |
| Chunk concat (multi-chunk only) | host, and only when text splits | only on long text |
| Voice-clip prep (clone only) | plugin, `ffmpeg -i clip -ar 16000` | only when cloning |
| Loudness normalisation | plugin | **removed** |

The plugin adds **no** ffmpeg step to a normal single-chunk utterance. Setting
`voice_compatible = False` in the provider would skip the Opus encode and hand
back a plain WAV attachment instead of a voice bubble — one fewer ffmpeg call, at
the cost of the bubble presentation in chat clients.

Because Hermes synthesises streamed replies sentence by sentence, a whole-file
normaliser flattens exactly the prosody the emotion steering creates — a whispered
aside gets boosted to a shout. That is a second, independent reason the gain step
was the wrong tool here: ducking lowers the competition instead of flattening the
voice.

`SENSEVOICE_ONNX_DIR` in the environment is an alternative to the setting.

### What is measured, and what is not

- Emotion **labels** are reliable on real speech: 6/6 clips labelled, three
  distinct emotions across speakers.
- Emotion **labels on synthetic TTS** are much weaker (1/3 matched the steering
  used to generate the clip). SenseVoice was trained on human speech; do not
  judge detection quality with TTS fixtures.
- **Intensity** is a heuristic ordering. Treat it as a dial, not a measurement.
- The reply-steer directions were confirmed to change the rendered audio
  (duration, zero-crossing rate and RMS all shift between `serene`, `content`
  and `happy`), but "sounds sympathetic" is a taste judgement, not a metric.

## Verification

`verify_zonos3.py` (in the session scratch dir) drives the real host lifecycle
module and the real `text_to_speech_tool`. Current result: **24/24 checks pass**,
including:

- acquire ×2 → release ×1 leaves one held lease (no counter leak)
- real synthesis: honest RIFF sizes, audible samples, `state = warm`
- idle stop: timer armed → process gone → `unloaded`, VRAM 23595 → 21493 MiB
- recovery: synthesis after an unload restarts the server and works
- emotion: controls reach the request and measurably change the audio
- dispatch: `text_to_speech_tool` reports `provider: zonos3` and delivers a
  playable voice bubble (the host converts the WAV to Opus because
  `voice_compatible` is True)

## Voice cloning

ZONOS3 clones voices from a short clip. Whether it works here depends on the
checkpoint, and this one is a specific case worth stating plainly.

**The release backbone is raw-reference-conditioned.**
`GET /v2/capabilities` reports `speaker_conditioning_type: "raw_reference_prefix"`
and `speaker_mode_default: "reference"`. Consequences, all verified against the
live server:

| Attempt | Result |
|---|---|
| `speaker_audio_base64` + `speaker_mode: reference` | works **only with** `zac-encoder.gguf` + `ssl-teacher.gguf` |
| `speaker_embedding_base64` (a `.npy` file) | **400** — "This checkpoint conditions on raw reference audio and does not accept speaker embedding files" |
| a cached speaker id from `POST /v2/speakers` | **400** — the cached entry is an embedding, which this checkpoint rejects |

So the 27 bundled voices work through the server's own path, and the
`.embedding.npy` sidecars shipped beside them are **not** accepted by this
checkpoint — only the server's internal speaker handling uses them.

**Real cloning = add two GGUFs (+5.6 GB).**

```bash
hf download Zyphra/ZONOS3-Mini gguf/zac-encoder.gguf gguf/ssl-teacher.gguf \
   --local-dir "$HERMES_HOME/models/zonos3-mini"
```

The plugin detects them at spawn and passes `--zac-encoder` / `--ssl-teacher`
automatically (`cloning_supported` reports the state). Once present:

```python
prov.synthesize("Hello in my own voice.", out, voice="C:/path/to/clip.wav")
# or per request: {"speaker_audio_base64": <b64 wav>, "speaker_mode": "reference"}
```

Reference audio must be **WAV** (mixed to mono, resampled to 44.1 kHz, first 60 s
used), and the encoder is a ~3.4 GiB allocation that `--reference-idle-unload 60`
releases a minute after the last clone. Encoding is slow on CPU — the docs quote
~1,740 CPU-seconds for a 10-second clip — so clone once, then reuse the resulting
codes rather than re-encoding per request.

Without the companions the server still runs and answers with bundled voices; it
simply refuses reference-mode cloning with a clear 400 rather than failing
obscurely.

## Known behaviour

- **Startup is ~2 s** because the backbone loads before the listener binds. That
  is why the first lease warms synchronously: the model is genuinely ready when
  `warm()` returns.
- **`/v1/audio/speech` `speed` is not supported** by the engine (422). Rate is
  exposed as the `delivery.tempo` steering direction, which this plugin maps when
  a `speed` multiplier is requested and a steering bundle is loaded.
- The **reference encoder is a separate 3.4 GiB allocation**; `--reference-idle-unload 60`
  lets it decay a minute after use.
- **Silent-audio guard**: a WAV with no audible samples is a *successful* response
  with a valid header, so it is detected from the samples and retried with a fresh
  seed. The native engine measured 0/12 silent, but this catches the failure users
  report as "read-aloud does nothing".
