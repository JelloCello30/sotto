# Sotto v0.2 — Whisper mode (audio + lips fusion) specification

v0.1 (SPEC.md) stays intact: Phrase mode is unchanged, camera-only, mic-free.
v0.2 adds a second input mode. The two modes are exclusive; Phrase mode remains
the default. Nothing about v0.1's engine API changes.

## Modes

- **Phrases** (default): calibrated silent-phrase input. Camera only. The
  microphone is NEVER requested while in this mode.
- **Whisper**: open-vocabulary dictation. Camera + microphone. Speech — whispered
  or voiced — is transcribed by a vendored OpenAI Whisper model running fully
  on-device (transformers.js + ONNX Runtime WASM/WebGPU). The lip tracker gates
  the microphone: an utterance is transcribed only when the user's OWN mouth was
  moving, which rejects background speech, media audio, and other people.

## Vendored assets

- `vendor/transformers/transformers.min.js` — @huggingface/transformers 3.8.1 ESM
- `vendor/transformers/ort-wasm-simd-threaded.jsep.mjs` + `.wasm` — ORT backend
- `vendor/whisper/<MODEL_ID>/` — config.json, generation_config.json,
  preprocessor_config.json, tokenizer.json, tokenizer_config.json,
  onnx/encoder_model_quantized.onnx, onnx/decoder_model_merged_quantized.onnx

Verified init incantation (works against these exact files):
```js
const { pipeline, env } = await import('../vendor/transformers/transformers.min.js');
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('../vendor/whisper/', import.meta.url).href; // MUST be absolute-safe under a subpath deploy (GitHub Pages /sotto/)
env.backends.onnx.wasm.wasmPaths = new URL('../vendor/transformers/', import.meta.url).href;
const asr = await pipeline('automatic-speech-recognition', MODEL_ID, { dtype: 'q8', device });
const { text } = await asr(float32Audio16k); // mono Float32Array @ 16000 Hz
```
`device`: try 'webgpu' when `navigator.gpu` exists, fall back to 'wasm' on any
failure. IMPORTANT for workers: import.meta.url-based paths keep everything
relative to js/, so the site works at both localhost root and /sotto/ subpath.

## js/audio.js — `SottoMic` (ES module, no UI)

```js
const mic = new SottoMic();
await mic.start();      // getUserMedia({audio: {...}, video: false}); throws SottoMicError code 'mic-denied'|'mic-none'
mic.stop();             // release tracks + close AudioContext
mic.slice(t0Ms, t1Ms);  // -> Float32Array @16kHz for the wall-clock window [t0,t1], clamped to buffer
mic.level();            // -> 0..1 smoothed RMS for a live meter
mic.running             // boolean
```
- AudioContext({sampleRate: 16000}); if the hardware refuses 16k, capture at
  native rate and resample linearly in slice().
- AudioWorklet capture into a preallocated Float32 ring buffer holding 30s.
  Track wall-clock (performance.now()) alignment of the write head so slice()
  can map engine segment times (also performance.now()-based) onto samples.
  AudioWorklet requires a module file: js/audio-worklet.js (tiny; registerProcessor
  forwarding input frames via port.postMessage with transferable chunks is fine).
- echoCancellation: false, noiseSuppression: false, autoGainControl: true.
- No MediaRecorder (compressed + unalignable). Raw PCM only. Nothing persisted.

## js/asr.js — `SottoASR` (wrapper) + js/asr-worker.js (module worker)

All inference in a module Worker so the UI thread never blocks.
```js
const asr = new SottoASR({ onStatus });   // status: 'loading-model' (with pct if available) | 'warming' | 'ready' | 'error'
await asr.load();                          // idempotent; resolves when ready; rejects with message on failure
const text = await asr.transcribe(float32, {timeoutMs = 30000});  // queued FIFO; empty string for silence/no-speech
asr.dispose();
```
- Worker protocol: {id, type:'load'|'transcribe'|'dispose', payload}; transfer the
  Float32Array buffer (postMessage transferables) — do not copy.
- In the worker: device selection webgpu→wasm fallback (catch createFromOptions
  errors AND first-inference errors — some GPUs fail late; on late failure,
  rebuild the pipeline on wasm and retry the job once).
- Suppress Whisper hallucination noise: if the returned text, lowercased and
  stripped of punctuation/whitespace, is empty or one of the classic silence
  hallucinations ('thank you', 'thanks for watching', 'you', '.') AND the input
  RMS was near-silence (< 0.004), return ''. Compute input RMS in the worker.
- MODEL_ID is a single exported const at the top of asr-worker.js.

## Fusion + app integration (js/app.js, app.html, css/app.css)

- Mode switch UI in the camera column: segmented control "Phrases | Whisper"
  (radio group semantics, keyboard operable). Persist in localStorage
  'sotto.mode.v1'. Default 'phrases'.
- Entering Whisper mode: lazily `new SottoMic().start()` + `asr.load()` with a
  visible loading state on the camera card ("loading speech model — ~75MB, one
  time"); on 'mic-denied' show a mic-blocked empty-state (mirror the camera one)
  and drop back to Phrases mode cleanly. Leaving Whisper mode: mic.stop()
  immediately (the mic indicator must turn off), keep ASR loaded.
- Fusion: reuse the engine's existing utterance segmentation (onSegment fires
  with t0/t1 from lip motion). In Whisper mode, onSegment → mic.slice(t0 - 250,
  t1 + 250) → asr.transcribe → non-empty text lands in the pad via the existing
  appendToPad path (verbatim, no sentence-casing of dictated text beyond what
  appendToPad already does), with a toast showing the first words. Matching
  against the phrase library is BYPASSED in Whisper mode.
- While ASR is busy, queue at most 2 pending segments; if the queue is full,
  drop the oldest and toast "transcriber is behind — dropped a segment" once.
- Status pill states while in Whisper mode: 'watching' → 'speaking…' →
  'transcribing…' → back. Show ASR device ('gpu'/'cpu') subtly in settings.
- The engine's calibration/practice/phrasebook UI is Phrases-mode-only: hide
  those cards in Whisper mode (they return when switching back).
- Privacy copy in-app must update per mode (see DESIGN.md honesty rules):
  Whisper mode card notes "microphone on, processed on this device, nothing
  uploaded". The nav 'Research preview' badge stays.

## sw.js

Add to PRECACHE: vendor/transformers/* (3 files), vendor/whisper/<MODEL_ID>/
(5 json + 2 onnx). Bump CACHE to 'sotto-v2'.

## Honesty requirements for all copy (binding)

- Whisper mode is presented as: open vocabulary, needs audible or whispered
  speech, on-device. It is NOT lipreading; the lips gate WHEN to listen, the
  audio decides WHAT was said. Say this plainly.
- Phrase mode remains the only fully-silent mode, and remains vocabulary-bound.
- Words like 'perfect', 'flawless', 'revolutionary' remain banned. State real
  latency numbers once measured.
