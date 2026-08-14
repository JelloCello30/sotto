# Sotto v0.3 — Fast-feel spec: verdict cache, streaming partials, split segments, audio+lip fusion

Measured reality this spec is built on (M-series Mac, warm, visible tab):
wasm q8 base.en inference is ~1.0s for 4.6s audio (threads do NOT help — 8
threads ≈ 1 thread, decoder is sequential). The slowness users feel is:
(a) ~23s cold load when webgpu is attempted, produces garbage, canary demotes;
(b) zero feedback during decode; (c) MAX_SEG_MS=6s discards long utterances;
(d) lips-only segmentation misses quiet articulation. v0.3 fixes all four.
COOP/COEP thread injection was evaluated and REJECTED (no win); serve.py's
local COI headers stay (harmless, allow future re-testing).

## 1. Device-verdict cache (js/asr.js only)

- localStorage key `sotto.asr.verdict.v1`: `{"model": "<MODEL_ID>", "device": "wasm", "at": <epoch ms>}`.
- Written ONLY when an auto load ends demoted to wasm (webgpu attempted and
  failed/garbage/stalled). Cleared/ignored when model differs or `at` older
  than 14 days (GPU drivers change; re-probe occasionally).
- On load with `device:'auto'`: if a valid verdict says wasm, pass 'wasm'
  through to the worker (skips webgpu + canary entirely). Expected effect:
  repeat-session load drops from ~23s to model-init+warmup (~4-6s).
- asr.js knows the model id: worker's ready/status payloads must include
  `model: MODEL_ID` (add to worker if absent); asr.js exposes `asr.model`.
- localStorage may be unavailable (private mode): degrade silently to no cache.

## 2. Streaming partials (js/asr-worker.js, js/asr.js, app integration)

- Worker: for each transcribe job, stream decoder text as it is generated
  using transformers.js streamers: prefer `WhisperTextStreamer` (exported by
  vendor/transformers/transformers.min.js v3.8) with its callback, else
  `TextStreamer(pipe.tokenizer, {skip_prompt: true, callback_function})`;
  wrap construction in try/catch — on any failure run without a streamer
  (partials are an enhancement, never a dependency). Post
  `{id, type: 'partial', payload: {text}}` with the ACCUMULATED text so far
  (worker accumulates; main thread just renders), throttled to at most one
  message per 120ms (trailing flush not required — the final result message
  supersedes).
- Do NOT emit partials for the canary/warm-up runs — only for real jobs.
- asr.js: `transcribe(audio, {onPartial})` — invoked with the accumulated
  string; exceptions in the callback are swallowed. Partials for stale
  (timed-out) jobs are dropped. Protocol doc in both file headers updated.
- Loop-guard/silence suppression still judge ONLY the final text.

## 3. Segment splitting — no more 6s ceiling (js/engine.js)

- New public method `setSegmentOptions({maxSegMs, splitOnMax})`, both fields
  optional; defaults preserve v0.1 behavior exactly (maxSegMs=MAX_SEG_MS=6000,
  splitOnMax=false). Documented in JSDoc; values clamped to sane ranges
  (maxSegMs 2000..30000 — the mic ring holds 30s).
- With splitOnMax=true, when an in-flight segment reaches maxSegMs: emit it
  as a completed segment (onSegment fires with recordingLabel handling
  unchanged — though in practice only whisper mode uses this), then IMMEDIATELY
  continue speaking state with a fresh segment seeded from the last PRE_ROLL
  ring frames, no refractory, no discard. energy/hysteresis state carries over.
  A later true still-tail ends the final piece normally.
- Splitting must not fire during calibration recording (phrase takes stay
  bounded): if a recording is pending, keep the old discard behavior.

## 4. Audio+lip fused activity (js/engine.js + app wiring)

- New public method `setAudioLevelProvider(fn|null)`: fn returns current mic
  RMS 0..1 (SottoMic.level()). Engine calls it at most once per frame inside
  _segmentStep; exceptions → treated as 0 and provider auto-cleared after 3
  consecutive throws. null resets to lips-only (v0.1 behavior; phrases mode).
- Fusion rules (only when provider set):
  - AUDIO_ACTIVE_RMS = 0.012 (whisper-level speech with AGC on).
  - Segment START: effective TAU_ON becomes TAU_ON * 0.55 while audio is
    active — quiet-lipped speech triggers; silent lip motion still needs the
    full threshold, so chewing/smiling alone does not fire more than before.
  - Segment END: while audio stays > AUDIO_ACTIVE_RMS, required still-frames
    double (OFF_FRAMES*2) — talking through a lip-still moment does not
    truncate the utterance. Bounded: audio alone can never hold a segment
    past maxSegMs (splitting handles that).
  - Segments still REQUIRE lip motion to start — audio alone never starts one
    (that is the whole privacy design: lips gate WHEN).
- app.js: on entering whisper mode, `engine.setAudioLevelProvider(() =>
  whisper.mic && whisper.mic.running ? whisper.mic.level() : 0)` and
  `engine.setSegmentOptions({splitOnMax: true})`; on leaving, provider null
  and options reset to defaults. Phrases mode untouched.

## 5. App UX for streaming (js/app.js, app.html, css/app.css)

- New live-transcription line ("the ticker") between the camera card and the
  pad in whisper mode only: a single-line, ellipsis-overflow, aria-live=polite
  element showing the current job's accumulated partial text in --ink-soft
  with a subtle accent left-border; empty+hidden when idle. Final text still
  commits to the pad exactly as now (the ticker clears when its job resolves
  or fails). Multiple queued jobs: ticker shows the currently-decoding job.
- The long-utterance toast (LONG_UTTER_MS machinery) is REMOVED — splitting
  makes it obsolete. Delete the dead code, not just the call.
- Whisper pending-queue cap rises from 2 to 3 (splitting produces more,
  shorter segments; each decodes in ~1s warm).
- Loading note copy: when the verdict cache made this a fast load, say
  "Loading speech model…" plainly; keep the '~75MB, one time' phrasing only
  for genuinely first loads if distinguishable, else use one honest line:
  "Loading the speech model. First ever load fetches ~75MB; after that it
  comes from cache."

## 6. Copy + docs (README.md, index.html)

- README performance paragraph: replace with measured truth — ~1s per
  utterance warm on an M-series MacBook (wasm, single-thread; threads were
  benchmarked and do not help this workload), a few seconds on the first
  utterance after load, ~4-6s model init on repeat visits (verdict cache),
  ~20s worst-case very first visit on machines whose GPU fails the canary.
  Mention live partial text streaming while decoding.
- README gets an Author section: Sotto is built by Nolan Woo
  ([@JelloCello30](https://github.com/JelloCello30)) — a cellist; one wry
  line about wanting to type while both hands are on the instrument. Factual,
  no contact info, no email.
- index.html: whisper-mode ledger entry gains the streaming mention ("words
  appear while you speak"); FAQ 'replace my keyboard' answer updates the
  latency claim to the measured ~1s.
- sw.js: bump CACHE to 'sotto-v3'.

## Honesty (binding, unchanged in spirit)

Numbers stated are the measured ones above; no 'instant', no 'perfect'.
Wispr-Flow comparisons stay out of the copy: Sotto is on-device and private;
that is its claim, not parity with cloud products.
