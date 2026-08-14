# Sotto — Engine Specification (js/engine.js)

ES module exporting class `SottoEngine`. No UI code. No frameworks. Runs fully
on-device using vendored MediaPipe Tasks Vision.

## Vendored assets (relative to site root)

- `vendor/mediapipe/vision_bundle.js` — ESM bundle exporting `FaceLandmarker`, `FilesetResolver`
- `vendor/mediapipe/wasm/` — wasm fileset dir
- `vendor/mediapipe/face_landmarker.task` — float16 model

Init pattern:
```js
import { FaceLandmarker, FilesetResolver } from '../vendor/mediapipe/vision_bundle.js';
const fileset = await FilesetResolver.forVisionTasks('vendor/mediapipe/wasm');
const lm = await FaceLandmarker.createFromOptions(fileset, {
  baseOptions: { modelAssetPath: 'vendor/mediapipe/face_landmarker.task', delegate: 'GPU' },
  runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
});
```
On GPU-delegate failure, retry once with `delegate: 'CPU'` before reporting error.

## Public API

```js
const engine = new SottoEngine({
  onState,    // (state: string, detail?: string) => void
  onFrame,    // (frame: FrameInfo) => void          — every processed video frame
  onSegment,  // (seg: Segment) => void              — a completed utterance segment
  onMatch,    // (m: Match|null, seg: Segment) => void — recognition result (null = no match)
});
await engine.start(videoEl);        // getUserMedia + model load + loop. Throws on camera denial.
engine.stop();                       // stop camera + loop, release tracks
engine.setSensitivity(x);            // 0..1, default 0.5 — scales accept threshold
engine.setPaused(bool);              // keep camera, suspend segmentation

// Calibration
engine.beginRecording(label);        // next detected utterance becomes a template
engine.cancelRecording();
// -> fires onSegment with seg.recordingLabel set, and stores the template

// Library (persisted in localStorage key 'sotto.library.v1')
engine.getLibrary();                 // -> [{label, templates: number, createdAt}]
engine.deleteTemplate(label, idx);   // remove one take
engine.deletePhrase(label);
engine.renamePhrase(oldLabel, newLabel);
engine.exportLibrary();              // -> JSON string
engine.importLibrary(json);          // merge; throws on invalid shape
engine.matchStats(seg);              // -> ranked [{label, distance, confidence}] for practice mode

// Testing / simulation (labeled clearly in UI as simulated)
engine.debugInjectSegment(frames);   // frames: number[][] — runs the full segment->match path
SottoEngine.syntheticSegment(kind);  // static helper: returns a plausible frames[][] for testing
```

## States (via onState)

`loading` → `ready` (camera on, face not yet seen) → `idle` (face tracked, mouth still)
→ `speaking` (utterance in progress) → back to `idle`. Errors: `no-camera`,
`no-face` (face lost > 1s), `error` (model failure, detail message).

## FrameInfo

```ts
{ t: number,               // ms timestamp
  features: Float32Array,  // length F — see Features
  mouthOpen: number,       // 0..1 convenience scalar (jawOpen blend)
  faceOk: boolean,
  fps: number }            // smoothed processing fps
```

## Features (F = 22)

18 mouth-relevant blendshape scores by categoryName:
jawOpen, jawForward, mouthClose, mouthFunnel, mouthPucker, mouthLeft, mouthRight,
mouthSmileLeft, mouthSmileRight, mouthFrownLeft, mouthFrownRight, mouthStretchLeft,
mouthStretchRight, mouthRollLower, mouthRollUpper, mouthShrugLower, mouthShrugUpper,
cheekPuff
plus 4 geometric scalars from landmarks, each normalized by inter-ocular distance
(landmarks 33 and 263): lip aperture (13↔14), lip width (61↔291), upper-lip height
(0↔13), lower-lip height (14↔17). Geometric values clamped to sane range.

Landmark indices refer to MediaPipe FaceMesh canonical topology.

## Utterance segmentation

- Energy per frame: mean |Δfeatures| vs previous frame (EMA-smoothed, α≈0.4).
- `speaking` begins when energy > τ_on for ≥3 consecutive frames.
- Ends when energy < τ_off (hysteresis, τ_off < τ_on) for ≥12 frames (~400ms @30fps).
- Discard segments shorter than 10 frames or longer than 6s (return to idle).
- Include 4 frames of pre-roll before the trigger frame.
- Segment: `{ frames: number[][], t0, t1, durationMs, recordingLabel? }`

## Matching (DTW template matcher)

- Resample every sequence to L=32 frames (linear interpolation per dim).
- Normalize each dim by fixed per-dim scale table (blendshapes already 0..1; scale
  geometric dims to comparable range). No per-sequence z-norm (unstable on
  near-constant dims).
- DTW with Sakoe-Chiba band w=8, squared-euclidean local cost, path-length
  normalized.
- Score phrase = min distance over its templates.
- Accept if best < τ_accept(sensitivity) AND best2/best ≥ 1.12 (margin) — else null.
- `confidence = clamp(1 - best/τ_reject, 0, 1)`; τ_reject ≈ 2·τ_accept.
- Runtime: vocab 30 × 4 templates × DTW(32×32×22) must stay < 8ms. Preallocate.

## Loop & lifecycle

- Prefer `video.requestVideoFrameCallback`, fallback rAF. Never process the same
  video timestamp twice (guard — FaceLandmarker VIDEO mode throws on repeat ts;
  use a monotonically increasing ts guard regardless).
- Pause processing when `document.hidden`; resume cleanly.
- `stop()` must release all MediaStream tracks and cancel callbacks.
- No allocation in the hot path where avoidable (reuse Float32Arrays).

## Storage shape (sotto.library.v1)

```json
{ "version": 1,
  "phrases": [ { "label": "hello", "createdAt": 1723500000000,
                 "templates": [ [[0.1, ...22 numbers], ...frames] ] } ] }
```
Cap: 60 phrases, 8 templates each. Reject imports exceeding caps.
