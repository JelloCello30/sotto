/*
 * Sotto — recognition engine (js/engine.js)
 *
 * Silent-speech phrase recognition over MediaPipe face blendshapes.
 * Fully on-device: vendored model, no network. The engine captures no
 * audio; at most it reads a mic RMS number from an app-supplied provider
 * (whisper mode, setAudioLevelProvider), and lip motion still gates every
 * segment.
 *
 * Exports:
 *   SottoEngine (default + named)  — the engine class per SPEC.md
 *   SottoEngineError               — typed error with .code
 *   resampleSequence, dtwDistance, frameEnergy, scaleDimsInPlace
 *                                  — the pure, unit-testable core
 *   CONSTANTS                      — frozen tunables, for UI display
 */

import { FaceLandmarker, FilesetResolver } from '../vendor/mediapipe/vision_bundle.js';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** Feature vector length: 18 blendshapes + 4 geometric scalars. */
export const F_DIM = 22;

/** DTW resample length and Sakoe-Chiba band half-width. */
const L_RESAMPLE = 32;
const DTW_BAND = 8;

/**
 * Segmentation energy. Energy is the mean |delta| per frame across all 22
 * scale-normalized dims, EMA-smoothed. Deliberate mouthing moves roughly six
 * dims by 0.03-0.06 per frame at 30 fps (mean over 22 dims: ~0.010-0.016);
 * a resting face jitters around 0.001-0.003. v0.5 (SPEC-V5 section 1)
 * derives the live thresholds from a streaming noise-floor estimate; the
 * fixed values below stay as the cold-start fallback until the floor has
 * FLOOR_MIN_SAMPLES idle samples.
 */
const EMA_ALPHA = 0.4;    // per SPEC: alpha ~= 0.4
const TAU_ON = 0.010;     // cold-start fixed trigger threshold
const TAU_OFF = 0.005;    // cold-start fixed release threshold
const MAX_SEG_MS = 6000;
/** Of the trailing still frames, keep this many as closing context. */
const TAIL_KEEP = 3;

/**
 * v0.4 frame-count hysteresis, superseded by the ms durations below. No
 * longer read by the segmenter; kept exported in CONSTANTS because the
 * names are part of the public tunables surface.
 */
const ON_FRAMES = 3;
const OFF_FRAMES = 12;
const PRE_ROLL = 4;
const MIN_SEG_FRAMES = 10;

/**
 * v0.5 time-based hysteresis (SPEC-V5): durations accumulated from real
 * frame timestamps, so segmentation behaves the same at 15/30/60 fps. Each
 * frame contributes its dt since the previous processed frame, clamped to
 * HYST_DT_MAX_MS so one janky frame cannot stand in for a whole
 * confirmation or stillness run. At exactly 30 fps (dt 33.3 ms) the trigger
 * fires on the 3rd consecutive above-threshold frame (accumulated 100 ms
 * >= 90) and release on the 12th still frame (400 ms) — the v0.4 counts.
 */
const ON_MS = 90;           // continuous above-tauOn time to start a segment
const OFF_MS = 400;         // continuous below-tauOff time to end one
const MIN_SEG_MS = 320;     // emitted segments must span at least this
const PRE_ROLL_MS = 130;    // ring context kept from before the on-run
const HYST_DT_MAX_MS = 100; // per-frame dt clamp for the accumulators

/**
 * v0.5 adaptive noise floor (SPEC-V5). While idle with a face tracked, two
 * streaming quantile trackers estimate the median (p50) and p90 of the EMA
 * energy. Live thresholds:
 *   tauOn  = clamp(p50 + K * (p90 - p50 + FLOOR_EPS), TAU_ON_MIN, TAU_ON_MAX)
 *   tauOff = max(p50 + K_OFF * (p90 - p50 + FLOOR_EPS), tauOn * TAU_OFF_MIN_RATIO)
 * with K = K_ON * (1.6 - sensitivity), so higher sensitivity lowers the bar
 * (0..1 -> 1.6x..0.6x). The floor freezes while speaking, during
 * refractory, and for FLOOR_REACQ_FREEZE_MS after face (re)acquisition or a
 * visibility gap. Until FLOOR_MIN_SAMPLES idle samples exist, the fixed
 * TAU_ON/TAU_OFF above apply (cold-start safety).
 *
 * Estimator (method, per SPEC-V5): frugal-style stochastic quantile
 * approximation on the pinball loss with an exponentially-weighted step.
 * Each idle sample x updates a tracker q for target quantile p as
 *   q <- q + 2 * w * s * (p - I(x <= q))
 * where s is an EMA of |x - p50| (adaptive scale, floored at FLOOR_DEV_MIN)
 * and w = 1 - exp(-dt / FLOOR_TC_MS) the per-sample weight from the real
 * inter-sample dt (fps-independent). At equilibrium P(x <= q) = p. The
 * gain is lower-bounded by 1/(n+1) for the first ~150 samples (decreasing-
 * gain burn-in) so estimates are already usable when adaptive thresholds
 * engage at FLOOR_MIN_SAMPLES. FLOOR_TC_MS = 5000 is an EMA time constant
 * of 5 s — comparable to a ~10 s sliding window (FLOOR_WINDOW_MS).
 */
const K_ON = 4.0;
const K_OFF = 1.5;
const FLOOR_EPS = 0.0008;          // the spec's EPS spread pad
const TAU_ON_MIN = 0.006;
const TAU_ON_MAX = 0.030;
const TAU_OFF_MIN_RATIO = 0.45;    // tauOff >= tauOn * this
const FLOOR_MIN_SAMPLES = 60;      // idle samples before adaptive engages
const FLOOR_WINDOW_MS = 10000;     // documented effective window (~10 s)
const FLOOR_TC_MS = 5000;          // EMA time constant behind that window
const FLOOR_REACQ_FREEZE_MS = 500; // floor freeze after face reacquisition
const FLOOR_DT_MAX_MS = 200;       // per-sample weight clamp across gaps
const FLOOR_DEV_MIN = 0.00001;     // scale floor so the tracker never stalls

/** setSegmentOptions clamp range for maxSegMs (the mic ring holds 30 s). */
const MAX_SEG_MS_MIN = 2000;
const MAX_SEG_MS_MAX = 30000;

/**
 * Audio+lip fusion (whisper mode; active only while an audio-level provider
 * is set). AUDIO_ACTIVE_RMS is whisper-level speech with AGC on. While the
 * mic RMS is above it, the segment-start threshold drops to
 * tauOn * AUDIO_TAU_ON_FACTOR (quiet-lipped speech triggers; silent lip
 * motion — chewing, smiling — still needs the full threshold) and the
 * stillness time required to end a segment doubles (OFF_MS 400 -> 800;
 * talking through a lip-still moment does not truncate the utterance;
 * bounded by maxSegMs). Energy itself is lips-only, so audio can never
 * start a segment alone.
 */
const AUDIO_ACTIVE_RMS = 0.012;
const AUDIO_TAU_ON_FACTOR = 0.55;
const AUDIO_OFF_FRAMES_FACTOR = 2;
/** Consecutive provider exceptions before the provider is auto-cleared. */
const AUDIO_PROVIDER_MAX_THROWS = 3;

/**
 * Accept threshold for the path-normalized squared-euclidean DTW distance.
 * Same-phrase re-articulations land around 0.01-0.04; different phrases sit
 * at 0.15 and above (several dims differing by ~0.3 squared and summed).
 * tau_accept(s) = 0.030 + 0.050 * s, so sensitivity 0.5 gives 0.055 —
 * comfortably above the same-phrase cloud, well below the impostor cloud.
 * Range: 0.030 (strict) to 0.080 (permissive).
 */
const TAU_ACCEPT_BASE = 0.030;
const TAU_ACCEPT_SPAN = 0.050;
const MARGIN_RATIO = 1.12;   // best2/best must be at least this, else ambiguous
const REJECT_FACTOR = 2;     // tau_reject = 2 * tau_accept (confidence scale)

/** Library caps (per SPEC storage section). */
const MAX_PHRASES = 60;
const MAX_TAKES = 8;
const MAX_LABEL_LEN = 64;
const STORAGE_KEY = 'sotto.library.v1';

/** touchPhrase() persistence throttle: at most one touch-driven write per. */
const TOUCH_PERSIST_MS = 5000;

/** Face-loss grace period before reporting 'no-face' from idle. */
const FACE_LOST_MS = 1000;

/** Geometric features are clamped to [0, GEOM_CLAMP] (inter-ocular units). */
const GEOM_CLAMP = 2;

/** Smoothing for the reported processing fps. */
const FPS_ALPHA = 0.2;

/**
 * Ring buffer size: must cover ON_MS of confirmation plus PRE_ROLL_MS of
 * context at 60 fps (~220 ms, about 14 frames) with slack.
 */
const RING_SIZE = 16;

/** The 18 mouth-relevant blendshapes, in feature order (SPEC.md). */
const BLENDSHAPE_NAMES = [
  'jawOpen', 'jawForward', 'mouthClose', 'mouthFunnel', 'mouthPucker',
  'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'cheekPuff',
];

/** FaceMesh canonical landmark indices used for geometry. */
const LM_EYE_L = 33, LM_EYE_R = 263;         // inter-ocular reference
const LM_LIP_UP_IN = 13, LM_LIP_LO_IN = 14;  // lip aperture
const LM_MOUTH_L = 61, LM_MOUTH_R = 291;     // lip width
const LM_LIP_UP_OUT = 0, LM_LIP_LO_OUT = 17; // outer lip mids

/**
 * Fixed per-dim scale table. Blendshapes are already 0..1, so scale 1.
 * Geometric dims are in inter-ocular units and get divided by a typical
 * full-range value to land in a comparable 0..1-ish range:
 *   lip aperture  / 0.40  (jaw fully open spans ~0.35-0.45 x IOD)
 *   lip width     / 0.80  (~0.55 x IOD at rest, ~0.75 in a wide smile)
 *   upper lip ht  / 0.15  (~0.05-0.14 x IOD)
 *   lower lip ht  / 0.15  (same order)
 * No per-sequence z-norm — unstable on near-constant dims (SPEC).
 */
const DIM_SCALE = new Float32Array(F_DIM).fill(1);
DIM_SCALE[18] = 0.40;
DIM_SCALE[19] = 0.80;
DIM_SCALE[20] = 0.15;
DIM_SCALE[21] = 0.15;
const INV_DIM_SCALE = new Float32Array(F_DIM);
for (let i = 0; i < F_DIM; i++) INV_DIM_SCALE[i] = 1 / DIM_SCALE[i];

/** Frozen tunables for UI display (practice mode, debug panels). */
export const CONSTANTS = Object.freeze({
  F_DIM, L_RESAMPLE, DTW_BAND, EMA_ALPHA, TAU_ON, TAU_OFF, ON_FRAMES,
  OFF_FRAMES, PRE_ROLL, MIN_SEG_FRAMES, MAX_SEG_MS, TAU_ACCEPT_BASE,
  TAU_ACCEPT_SPAN, MARGIN_RATIO, REJECT_FACTOR, MAX_PHRASES, MAX_TAKES,
  FACE_LOST_MS, STORAGE_KEY, MAX_SEG_MS_MIN, MAX_SEG_MS_MAX,
  AUDIO_ACTIVE_RMS, AUDIO_TAU_ON_FACTOR, AUDIO_OFF_FRAMES_FACTOR,
  // v0.5 adaptive segmentation (SPEC-V5 section 1).
  ON_MS, OFF_MS, MIN_SEG_MS, PRE_ROLL_MS, HYST_DT_MAX_MS,
  K_ON, K_OFF, FLOOR_EPS, TAU_ON_MIN, TAU_ON_MAX, TAU_OFF_MIN_RATIO,
  FLOOR_MIN_SAMPLES, FLOOR_WINDOW_MS, FLOOR_TC_MS, FLOOR_REACQ_FREEZE_MS,
});

// ---------------------------------------------------------------------------
// Typed error
// ---------------------------------------------------------------------------

/**
 * Engine error carrying a machine-readable code for the UI.
 * Codes: 'camera-denied' | 'camera-none' | 'model-failed' | 'bad-args' |
 *        'bad-label' | 'invalid-frames' | 'invalid-library' | 'library-full' |
 *        'phrase-full' | 'not-found' | 'label-exists'
 */
export class SottoEngineError extends Error {
  /**
   * @param {string} code machine-readable error code
   * @param {string} message human-readable detail
   */
  constructor(code, message) {
    super(message);
    this.name = 'SottoEngineError';
    this.code = code;
  }
}

// ---------------------------------------------------------------------------
// Pure core (unit-testable, no DOM, no state)
// ---------------------------------------------------------------------------

/**
 * Linearly resample a sequence of frames to exactly L rows.
 * Endpoints are preserved; a single-frame input repeats.
 * @param {ArrayLike<ArrayLike<number>>} frames input, N rows of F values (N >= 1)
 * @param {number} L output row count
 * @param {number} F values per row
 * @param {Float32Array} [out] optional preallocated output (length L*F)
 * @returns {Float32Array} row-major L*F resampled sequence
 */
export function resampleSequence(frames, L, F, out) {
  const N = frames.length;
  if (N === 0) throw new SottoEngineError('invalid-frames', 'cannot resample an empty sequence');
  const dst = out || new Float32Array(L * F);
  if (N === 1) {
    const r = frames[0];
    for (let i = 0; i < L; i++) {
      const off = i * F;
      for (let k = 0; k < F; k++) dst[off + k] = r[k];
    }
    return dst;
  }
  const step = (N - 1) / (L - 1);
  for (let i = 0; i < L; i++) {
    const x = i * step;
    let i0 = Math.floor(x);
    if (i0 > N - 2) i0 = N - 2; // clamp so i0+1 is valid (guards float edge at x = N-1)
    const frac = x - i0;
    const r0 = frames[i0], r1 = frames[i0 + 1];
    const off = i * F;
    for (let k = 0; k < F; k++) dst[off + k] = r0[k] + (r1[k] - r0[k]) * frac;
  }
  return dst;
}

/**
 * Divide each dim of a row-major L*F sequence by its fixed per-dim scale
 * (multiplies by the reciprocal), in place.
 * @param {Float32Array} buf row-major L*F sequence, mutated
 * @param {number} L rows
 * @param {number} F dims
 * @param {Float32Array} invScale reciprocal scale per dim (length F)
 * @returns {Float32Array} buf
 */
export function scaleDimsInPlace(buf, L, F, invScale) {
  for (let i = 0; i < L; i++) {
    const off = i * F;
    for (let k = 0; k < F; k++) buf[off + k] *= invScale[k];
  }
  return buf;
}

/**
 * Per-frame articulation energy: mean |delta| across dims, each delta
 * normalized by the fixed per-dim scale.
 * @param {ArrayLike<number>} curr current feature row (length F)
 * @param {ArrayLike<number>} prev previous feature row (length F)
 * @param {Float32Array} invScale reciprocal scale per dim (length F)
 * @returns {number} mean scaled absolute delta
 */
export function frameEnergy(curr, prev, invScale) {
  const F = curr.length;
  let s = 0;
  for (let k = 0; k < F; k++) {
    const d = curr[k] - prev[k];
    s += (d >= 0 ? d : -d) * invScale[k];
  }
  return s / F;
}

/**
 * DTW distance between two row-major L*F sequences with a Sakoe-Chiba band
 * of half-width w. Local cost is squared euclidean over the F dims; the
 * returned distance is the accumulated cost divided by the number of steps
 * on the optimal path (path-length normalized). Cells outside the band are
 * unreachable (treated as Infinity), and the j range is clamped to [0, L-1]
 * per row, so any 0 < w < L is handled correctly.
 * @param {Float32Array} a row-major L*F sequence
 * @param {Float32Array} b row-major L*F sequence
 * @param {number} L rows per sequence
 * @param {number} F dims per row
 * @param {number} w Sakoe-Chiba half-width
 * @param {{costPrev: Float64Array, costCur: Float64Array,
 *          lenPrev: Int32Array, lenCur: Int32Array}} [scratch]
 *        optional preallocated rows (each length L) to avoid allocation
 * @returns {number} path-normalized DTW distance
 */
export function dtwDistance(a, b, L, F, w, scratch) {
  const s = scratch || {
    costPrev: new Float64Array(L), costCur: new Float64Array(L),
    lenPrev: new Int32Array(L), lenCur: new Int32Array(L),
  };
  let costPrev = s.costPrev, costCur = s.costCur;
  let lenPrev = s.lenPrev, lenCur = s.lenCur;
  costPrev.fill(Infinity);
  lenPrev.fill(0);
  for (let i = 0; i < L; i++) {
    costCur.fill(Infinity);
    lenCur.fill(0);
    const jLo = i - w > 0 ? i - w : 0;
    const jHi = i + w < L - 1 ? i + w : L - 1;
    const ao = i * F;
    for (let j = jLo; j <= jHi; j++) {
      // Local squared-euclidean cost.
      let d = 0;
      const bo = j * F;
      for (let k = 0; k < F; k++) {
        const e = a[ao + k] - b[bo + k];
        d += e * e;
      }
      if (i === 0 && j === 0) {
        costCur[0] = d;
        lenCur[0] = 1;
        continue;
      }
      // Best predecessor among up / left / diagonal. Out-of-band neighbors
      // read Infinity from the filled rows, which excludes them naturally.
      let bc = Infinity, bn = 0;
      if (i > 0) {
        const c = costPrev[j];
        if (c < bc) { bc = c; bn = lenPrev[j]; }
      }
      if (j > 0) {
        const c = costCur[j - 1];
        if (c < bc) { bc = c; bn = lenCur[j - 1]; }
      }
      if (i > 0 && j > 0) {
        const c = costPrev[j - 1];
        if (c < bc) { bc = c; bn = lenPrev[j - 1]; }
      }
      if (bc === Infinity) continue; // disconnected cell (degenerate band)
      costCur[j] = bc + d;
      lenCur[j] = bn + 1;
    }
    // Swap rolling rows.
    let t = costPrev; costPrev = costCur; costCur = t;
    t = lenPrev; lenPrev = lenCur; lenCur = t;
  }
  const n = lenPrev[L - 1];
  return n > 0 ? costPrev[L - 1] / n : Infinity;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function clampGeom(x) { return clamp(x, 0, GEOM_CLAMP); }

function nowMs() {
  return typeof performance !== 'undefined' && performance.now
    ? performance.now() : Date.now();
}

/** Mean of a numeric array; 0 for an empty one (segment audio level). */
function meanLevel(levels) {
  const n = levels.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += levels[i];
  return s / n;
}

/** 2D landmark distance with x corrected by frame aspect ratio (landmarks
 *  are normalized to the frame, so ratios need isotropic units). */
function landmarkDist(a, b, aspect) {
  const dx = (a.x - b.x) * aspect;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function normalizeLabel(label) {
  if (typeof label !== 'string') {
    throw new SottoEngineError('bad-label', 'phrase label must be a string');
  }
  const lab = label.trim();
  if (lab.length === 0 || lab.length > MAX_LABEL_LEN) {
    throw new SottoEngineError('bad-label',
      `phrase label must be 1-${MAX_LABEL_LEN} characters after trimming`);
  }
  return lab;
}

/** Validate a frames array: >= 2 rows, each row exactly F_DIM finite numbers. */
function validateFrames(frames, what) {
  const name = what || 'frames';
  if (!Array.isArray(frames) || frames.length < 2) {
    throw new SottoEngineError('invalid-frames', `${name} must be an array of at least 2 rows`);
  }
  for (let i = 0; i < frames.length; i++) {
    const row = frames[i];
    if (!row || typeof row.length !== 'number' || row.length !== F_DIM) {
      throw new SottoEngineError('invalid-frames', `${name}[${i}] must have exactly ${F_DIM} values`);
    }
    for (let k = 0; k < F_DIM; k++) {
      const v = row[k];
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        throw new SottoEngineError('invalid-frames', `${name}[${i}][${k}] is not a finite number`);
      }
    }
  }
}

/** Validate the full library object shape (used for load and import). */
function validateLibraryShape(obj) {
  const bad = (msg) => new SottoEngineError('invalid-library', msg);
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) throw bad('library must be an object');
  if (obj.version !== 1) throw bad('unsupported library version (expected 1)');
  if (!Array.isArray(obj.phrases)) throw bad('library.phrases must be an array');
  if (obj.phrases.length > MAX_PHRASES) {
    throw bad(`library has ${obj.phrases.length} phrases (max ${MAX_PHRASES})`);
  }
  const seen = new Set();
  for (const p of obj.phrases) {
    if (!p || typeof p !== 'object') throw bad('each phrase must be an object');
    if (typeof p.label !== 'string' || p.label.trim().length === 0 || p.label.length > MAX_LABEL_LEN) {
      throw bad('phrase label must be a non-empty string');
    }
    // Normalize in place so stored labels always match what every mutator's
    // normalizeLabel() will look up — an untrimmed import ("  hello  ") must
    // not alias onto a different phrase ("hello"), and trimming must not
    // silently create duplicates.
    p.label = p.label.trim();
    if (seen.has(p.label)) throw bad(`duplicate phrase label "${p.label}"`);
    seen.add(p.label);
    if (typeof p.createdAt !== 'number' || !Number.isFinite(p.createdAt)) {
      throw bad(`phrase "${p.label}" is missing a numeric createdAt`);
    }
    // Optional v0.4 fields: tolerated when absent (absent = not learned),
    // type-checked when present, and preserved as-is through load/import.
    if ('learned' in p && typeof p.learned !== 'boolean') {
      throw bad(`phrase "${p.label}" has a non-boolean learned flag`);
    }
    if ('lastUsedAt' in p && p.lastUsedAt !== null
        && (typeof p.lastUsedAt !== 'number' || !Number.isFinite(p.lastUsedAt))) {
      throw bad(`phrase "${p.label}" has an invalid lastUsedAt`);
    }
    if (!Array.isArray(p.templates) || p.templates.length === 0 || p.templates.length > MAX_TAKES) {
      throw bad(`phrase "${p.label}" must have 1-${MAX_TAKES} templates`);
    }
    for (const tpl of p.templates) validateFrames(tpl, `template of "${p.label}"`);
  }
}

function deepCopyTemplates(templates) {
  return templates.map((tpl) => tpl.map((row) => Array.from(row, Number)));
}

/** Deterministic small PRNG (for syntheticSegment micro-variation only). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(s) {
  let h = 2166136261 >>> 0; // FNV-1a
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

// ---------------------------------------------------------------------------
// JSDoc shapes
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} FrameInfo
 * @property {number} t ms timestamp of the processed frame
 * @property {Float32Array} features length-22 feature vector (REUSED between
 *           frames — copy if you retain it)
 * @property {number} mouthOpen 0..1 convenience scalar (jawOpen blendshape)
 * @property {boolean} faceOk whether a face was tracked this frame
 * @property {number} fps smoothed processing fps
 * @property {number} energy current EMA articulation energy (the signal the
 *           segmenter thresholds against)
 * @property {number} tauOn live segment-start threshold (adaptive once the
 *           floor has enough samples; includes the audio assist while
 *           audio-active)
 * @property {number} tauOff live segment-release threshold
 */

/**
 * @typedef {Object} Segment
 * @property {number[][]} frames feature rows, each length 22
 * @property {number} t0 ms timestamp of the first frame
 * @property {number} t1 ms timestamp of the last frame
 * @property {number} durationMs t1 - t0
 * @property {number} audioLevel mean polled audio-provider level (mic RMS
 *           0..1) across this segment's frames; 0 when no provider is set.
 *           Split pieces average only their own span.
 * @property {string} [recordingLabel] set when this segment was captured as a
 *           calibration template
 * @property {string} [recordingError] set if storing the template failed
 */

/**
 * @typedef {Object} Match
 * @property {string} label matched phrase label
 * @property {number} distance path-normalized DTW distance of the best template
 * @property {number} confidence clamp(1 - distance/tau_reject, 0, 1)
 * @property {number} margin best2/best distance ratio (Infinity if only one
 *           phrase is enrolled)
 */

/**
 * @typedef {Object} PhraseSummary
 * @property {string} label
 * @property {number} templates number of stored takes
 * @property {number} createdAt ms epoch
 * @property {boolean} learned true when the phrase was auto-learned from
 *           speech (enrollSegment) rather than calibrated
 * @property {number|null} lastUsedAt ms epoch of the last touchPhrase(), or
 *           null if never touched
 */

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/**
 * Silent-speech recognition engine. See SPEC.md for the contract.
 *
 * Lifecycle: construct with callbacks, `await start(videoEl)`, `stop()`.
 * States via onState: 'loading' -> 'ready' -> 'idle' <-> 'speaking',
 * plus 'no-camera', 'no-face', 'error'.
 */
export class SottoEngine {
  /**
   * @param {Object} [callbacks]
   * @param {(state: string, detail?: string) => void} [callbacks.onState]
   * @param {(frame: FrameInfo) => void} [callbacks.onFrame] every processed
   *        frame; the FrameInfo object and its features array are reused
   * @param {(seg: Segment) => void} [callbacks.onSegment] each completed
   *        utterance segment
   * @param {(m: Match|null, seg: Segment) => void} [callbacks.onMatch]
   *        recognition result for non-recording segments (null = no match)
   */
  constructor({ onState, onFrame, onSegment, onMatch } = {}) {
    this._onState = typeof onState === 'function' ? onState : null;
    this._onFrame = typeof onFrame === 'function' ? onFrame : null;
    this._onSegment = typeof onSegment === 'function' ? onSegment : null;
    this._onMatch = typeof onMatch === 'function' ? onMatch : null;

    this._state = null;
    this._stateDetail = undefined;
    this._sensitivity = 0.5;
    this._paused = false;
    this._running = false;
    this._starting = false;

    this._video = null;
    this._stream = null;
    this._landmarker = null;
    this._useVfc = false;
    this._vfcId = 0;
    this._rafId = 0;
    this._lastTs = -1;        // strictly-increasing detector timestamp guard
    this._lastMediaTime = -1; // rAF fallback: skip repeated video frames

    // Feature extraction (all buffers preallocated; hot path allocates nothing).
    this._features = new Float32Array(F_DIM);
    this._prevFeatures = new Float32Array(F_DIM);
    this._haveFeatures = false;
    this._blendIdx = null; // Int16Array(18): our order -> categories index
    this._frameInfo = {
      t: 0, features: this._features, mouthOpen: 0, faceOk: false, fps: 0,
      energy: 0, tauOn: TAU_ON, tauOff: TAU_OFF,
    };
    this._fps = 0;
    this._lastFrameT = 0;

    // Pre-roll ring buffer.
    this._ring = [];
    for (let i = 0; i < RING_SIZE; i++) this._ring.push(new Float32Array(F_DIM));
    this._ringTs = new Float64Array(RING_SIZE);
    this._ringLevels = new Float64Array(RING_SIZE); // polled audio level per slot
    this._ringHead = 0;
    this._ringCount = 0;

    // Segmentation state (v0.5: ms-based hysteresis over real frame times).
    this._energy = 0;
    this._onMs = 0;          // accumulated continuous above-tauOn time
    this._onRunStartT = 0;   // timestamp of the first frame of the on-run
    this._offMs = 0;         // accumulated continuous below-tauOff time
    this._offRun = 0;        // frames in the current still run (tail trim)
    this._segLastT = -1;     // previous _segmentStep timestamp (-1 = none)
    this._speaking = false;
    this._segFrames = null;
    this._segTs = null;
    this._segLevels = null; // per-frame polled audio levels, parallel to _segTs
    this._refractory = false; // after an over-long discard: wait for stillness
    // v0.5 adaptive noise floor (SPEC-V5; frozen while speaking, during
    // refractory, and for FLOOR_REACQ_FREEZE_MS after face reacquisition).
    this._floorP50 = 0;
    this._floorP90 = 0;
    this._floorDev = 0;       // EMA of |x - p50|: the quantile step scale
    this._floorSamples = 0;
    this._floorLastT = 0;
    this._floorHoldUntil = 0;
    this._tauOnBase = TAU_ON; // live start threshold before audio assist
    this._tauOn = TAU_ON;     // live start threshold (assist included)
    this._tauOff = TAU_OFF;   // live release threshold
    // Segment options + audio fusion (defaults preserve v0.1 behavior).
    this._maxSegMs = MAX_SEG_MS;
    this._splitOnMax = false;
    this._audioLevelProvider = null;
    this._audioProviderThrows = 0;
    this._faceOk = false;
    this._faceEverSeen = false;
    this._faceRegained = false;
    this._lastFaceT = 0;

    // Matching scratch (preallocated; DTW runs allocation-free).
    this._queryBuf = new Float32Array(L_RESAMPLE * F_DIM);
    this._dtwScratch = {
      costPrev: new Float64Array(L_RESAMPLE),
      costCur: new Float64Array(L_RESAMPLE),
      lenPrev: new Int32Array(L_RESAMPLE),
      lenCur: new Int32Array(L_RESAMPLE),
    };

    this._pendingRecording = null;

    // Library. Falls back to in-memory when localStorage is unavailable.
    this._storage = (() => {
      try { return typeof localStorage !== 'undefined' ? localStorage : null; }
      catch { return null; }
    })();
    this._library = this._loadLibrary();
    this._cache = []; // [{label, takes: Float32Array(L*F)[]}]
    this._rebuildCache();

    // touchPhrase() persistence throttle (leading write + one trailing write).
    this._lastTouchPersistT = -Infinity;
    this._touchPersistTimer = 0;

    this._boundVfc = this._vfcStep.bind(this);
    this._boundRaf = this._rafStep.bind(this);
    this._boundVisibility = this._handleVisibility.bind(this);
  }

  /** Current engine state string (read-only convenience for the UI). */
  get state() { return this._state; }

  // -- lifecycle ------------------------------------------------------------

  /**
   * Open the camera, load the landmark model (GPU delegate, one CPU retry),
   * attach the stream to the given video element, and start the frame loop.
   * Never requests audio.
   * @param {HTMLVideoElement} videoEl target video element
   * @returns {Promise<void>}
   * @throws {SottoEngineError} code 'camera-denied' | 'camera-none' on camera
   *         failure, 'model-failed' on model load failure
   */
  async start(videoEl) {
    if (this._running || this._starting) return;
    if (!videoEl || typeof videoEl.play !== 'function') {
      throw new SottoEngineError('bad-args', 'start(videoEl) requires a video element');
    }
    this._starting = true;
    this._setState('loading');
    let stream = null;
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (err) {
        const name = err && err.name;
        const code = (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError')
          ? 'camera-denied' : 'camera-none';
        this._setState('no-camera', code);
        throw new SottoEngineError(code,
          code === 'camera-denied' ? 'camera permission was denied' : 'no usable camera was found');
      }

      try {
        await this._ensureLandmarker();
      } catch (err) {
        const detail = 'face landmark model failed to load: ' + (err && err.message ? err.message : String(err));
        this._setState('error', detail);
        throw new SottoEngineError('model-failed', detail);
      }

      this._stream = stream;
      this._video = videoEl;
      videoEl.muted = true;
      videoEl.playsInline = true;
      videoEl.setAttribute('playsinline', '');
      videoEl.srcObject = stream;
      await videoEl.play();

      this._useVfc = typeof videoEl.requestVideoFrameCallback === 'function';
      // A camera can vanish mid-session (USB unplug, OS-level revocation).
      // track.stop() does not fire 'ended', so this only catches external loss.
      const vTrack = stream.getVideoTracks()[0];
      if (vTrack) {
        vTrack.addEventListener('ended', () => {
          if (this._running) {
            this.stop();
            this._setState('no-camera', 'camera-ended');
          }
        });
      }
      document.addEventListener('visibilitychange', this._boundVisibility);
      this._resetRuntime();
      this._running = true;
      this._setState('ready');
      this._scheduleFrame();
    } catch (err) {
      if (stream && !this._running) {
        for (const track of stream.getTracks()) track.stop();
      }
      this._stream = null;
      throw err;
    } finally {
      this._starting = false;
    }
  }

  /**
   * Stop the frame loop and the camera: cancels pending video-frame/rAF
   * callbacks, detaches the stream, and releases every MediaStream track.
   * The loaded model is kept so a later start() is fast. Emits no state.
   * @returns {void}
   */
  stop() {
    this._running = false;
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._boundVisibility);
    }
    this._cancelScheduled();
    if (this._video) this._video.srcObject = null;
    if (this._stream) {
      for (const track of this._stream.getTracks()) track.stop();
      this._stream = null;
    }
    this._video = null;
    this._pendingRecording = null;
    this._resetRuntime();
  }

  /**
   * Set the accept-threshold sensitivity.
   * @param {number} x 0..1 (clamped); 0.5 is the default
   * @returns {void}
   */
  setSensitivity(x) {
    const v = Number(x);
    this._sensitivity = Number.isFinite(v) ? clamp01(v) : 0.5;
  }

  /**
   * Suspend or resume utterance segmentation while keeping the camera and
   * per-frame callbacks running. Entering pause discards any in-progress
   * utterance.
   * @param {boolean} paused
   * @returns {void}
   */
  setPaused(paused) {
    const p = !!paused;
    if (p === this._paused) return;
    this._paused = p;
    if (p) {
      if (this._speaking) this._discardSegment();
      this._resetHysteresis();
      this._segLastT = -1; // no dt accumulation across the pause
    }
  }

  /**
   * Configure segmentation limits. Omitted fields revert to their defaults,
   * which reproduce v0.1 behavior exactly (maxSegMs 6000, splitOnMax false).
   * @param {Object} [opts]
   * @param {number} [opts.maxSegMs] utterance length limit in ms, clamped to
   *        2000..30000 (the mic ring holds 30 s)
   * @param {boolean} [opts.splitOnMax] when true, an utterance reaching
   *        maxSegMs is emitted as a completed segment and the speaking state
   *        continues into a fresh segment seeded from the ring frames inside
   *        PRE_ROLL_MS — no discard, no refractory. When false (default), the
   *        over-long utterance is discarded and stillness is required before
   *        re-arming. Splitting never fires while a calibration recording is
   *        armed; the discard behavior applies so phrase takes stay bounded.
   * @returns {void}
   */
  setSegmentOptions(opts) {
    const o = opts && typeof opts === 'object' ? opts : {};
    const v = Number(o.maxSegMs);
    this._maxSegMs = Number.isFinite(v) ? clamp(v, MAX_SEG_MS_MIN, MAX_SEG_MS_MAX) : MAX_SEG_MS;
    this._splitOnMax = !!o.splitOnMax;
  }

  /**
   * Install or clear the audio-level provider for audio+lip fusion (whisper
   * mode). The provider is called at most once per processed frame from
   * inside the segmentation step and must return the current mic RMS in
   * 0..1 (e.g. SottoMic.level()). While the level is above AUDIO_ACTIVE_RMS
   * the segment-start threshold drops to the live tauOn * 0.55 and the
   * stillness time required to end a segment doubles (OFF_MS 400 -> 800) —
   * but audio alone never starts a segment: energy is computed from lip
   * motion only, and lips gate when the engine listens. A provider
   * exception is treated as level 0; after 3
   * consecutive exceptions the provider is auto-cleared. Non-function
   * values (including null) restore lips-only segmentation (v0.1 behavior;
   * phrases mode).
   * @param {(() => number)|null} fn returns current mic RMS 0..1, or null
   * @returns {void}
   */
  setAudioLevelProvider(fn) {
    this._audioLevelProvider = typeof fn === 'function' ? fn : null;
    this._audioProviderThrows = 0;
  }

  // -- calibration ----------------------------------------------------------

  /**
   * Arm calibration: the next detected utterance is stored as a template for
   * the given phrase, and the resulting segment carries recordingLabel.
   * Matching is suppressed for that one segment.
   * @param {string} label phrase label (trimmed, 1-64 chars)
   * @returns {void}
   * @throws {SottoEngineError} 'bad-label', 'library-full' (60 phrases) or
   *         'phrase-full' (8 takes)
   */
  beginRecording(label) {
    const lab = normalizeLabel(label);
    const phrase = this._findPhrase(lab);
    if (!phrase && this._library.phrases.length >= MAX_PHRASES) {
      throw new SottoEngineError('library-full', `library is capped at ${MAX_PHRASES} phrases`);
    }
    if (phrase && phrase.templates.length >= MAX_TAKES) {
      throw new SottoEngineError('phrase-full', `"${lab}" already has ${MAX_TAKES} takes`);
    }
    this._pendingRecording = lab;
  }

  /**
   * Disarm calibration. A no-op if nothing was armed.
   * @returns {void}
   */
  cancelRecording() {
    this._pendingRecording = null;
  }

  // -- library --------------------------------------------------------------

  /**
   * Summarize the stored phrase library.
   * @returns {PhraseSummary[]} one entry per phrase, insertion order
   */
  getLibrary() {
    return this._library.phrases.map((p) => ({
      label: p.label,
      templates: p.templates.length,
      createdAt: p.createdAt,
      learned: p.learned === true,
      lastUsedAt: typeof p.lastUsedAt === 'number' ? p.lastUsedAt : null,
    }));
  }

  /**
   * Delete one take from a phrase. Deleting the last take removes the phrase
   * (a phrase with zero templates cannot match anything).
   * @param {string} label phrase label
   * @param {number} idx take index (0-based)
   * @returns {void}
   * @throws {SottoEngineError} 'not-found' on unknown label or bad index
   */
  deleteTemplate(label, idx) {
    const lab = normalizeLabel(label);
    const phrase = this._findPhrase(lab);
    if (!phrase) throw new SottoEngineError('not-found', `no phrase "${lab}"`);
    if (!Number.isInteger(idx) || idx < 0 || idx >= phrase.templates.length) {
      throw new SottoEngineError('not-found', `"${lab}" has no take ${idx}`);
    }
    phrase.templates.splice(idx, 1);
    if (phrase.templates.length === 0) {
      this._library.phrases = this._library.phrases.filter((p) => p !== phrase);
    }
    this._persist();
    this._rebuildCache();
  }

  /**
   * Delete a phrase and all of its takes.
   * @param {string} label phrase label
   * @returns {void}
   * @throws {SottoEngineError} 'not-found' on unknown label
   */
  deletePhrase(label) {
    const lab = normalizeLabel(label);
    const before = this._library.phrases.length;
    this._library.phrases = this._library.phrases.filter((p) => p.label !== lab);
    if (this._library.phrases.length === before) {
      throw new SottoEngineError('not-found', `no phrase "${lab}"`);
    }
    this._persist();
    this._rebuildCache();
  }

  /**
   * Rename a phrase, keeping its takes.
   * @param {string} oldLabel current label
   * @param {string} newLabel new label (must not already exist)
   * @returns {void}
   * @throws {SottoEngineError} 'not-found' | 'label-exists' | 'bad-label'
   */
  renamePhrase(oldLabel, newLabel) {
    const oldLab = normalizeLabel(oldLabel);
    const newLab = normalizeLabel(newLabel);
    const phrase = this._findPhrase(oldLab);
    if (!phrase) throw new SottoEngineError('not-found', `no phrase "${oldLab}"`);
    if (newLab !== oldLab && this._findPhrase(newLab)) {
      throw new SottoEngineError('label-exists', `a phrase named "${newLab}" already exists`);
    }
    phrase.label = newLab;
    if (this._pendingRecording === oldLab) this._pendingRecording = newLab;
    this._persist();
    this._rebuildCache();
  }

  /**
   * Enroll a segment's frames as a template for a phrase, outside the
   * calibration flow (v0.4 auto-learning). Caps match the recording path
   * with one difference: when the phrase already exists and is at the
   * 8-take cap, take 0 (the oldest) is evicted silently and the new take
   * added — no throw. Creating a new phrase stores learned and
   * lastUsedAt: null; adding a take to an existing phrase changes neither
   * field. Persists immediately (not throttled).
   * @param {string} label phrase label (trimmed, 1-64 chars)
   * @param {number[][]} frames feature rows, each exactly 22 finite numbers
   *        (>= 2 rows; e.g. a Segment's frames)
   * @param {Object} [opts]
   * @param {boolean} [opts.learned=false] mark a newly created phrase as
   *        auto-learned (ignored when the phrase already exists)
   * @returns {{label: string, templates: number}} the label and its take
   *          count after enrollment
   * @throws {SottoEngineError} 'bad-label' | 'invalid-frames' |
   *         'library-full' (creating beyond 60 phrases)
   */
  enrollSegment(label, frames, { learned = false } = {}) {
    const lab = normalizeLabel(label);
    validateFrames(frames, 'enrolled frames');
    let phrase = this._findPhrase(lab);
    if (!phrase) {
      if (this._library.phrases.length >= MAX_PHRASES) {
        throw new SottoEngineError('library-full', `library is capped at ${MAX_PHRASES} phrases`);
      }
      phrase = {
        label: lab,
        createdAt: Date.now(),
        templates: [],
        learned: !!learned,
        lastUsedAt: null,
      };
      this._library.phrases.push(phrase);
    } else {
      while (phrase.templates.length >= MAX_TAKES) phrase.templates.shift();
    }
    phrase.templates.push(frames.map((row) => Array.from(row, Number)));
    this._persist();
    this._rebuildCache();
    return { label: lab, templates: phrase.templates.length };
  }

  /**
   * Mark a phrase as just used: sets lastUsedAt to now, in memory
   * immediately, with touch-driven persistence throttled to at most one
   * localStorage write per 5 s (leading write, then one trailing write at
   * the window's end so the last touch is never lost). Any full persist
   * from another mutation cancels a pending trailing write — the data is
   * already on disk.
   * @param {string} label phrase label
   * @returns {void}
   * @throws {SottoEngineError} 'bad-label' | 'not-found'
   */
  touchPhrase(label) {
    const lab = normalizeLabel(label);
    const phrase = this._findPhrase(lab);
    if (!phrase) throw new SottoEngineError('not-found', `no phrase "${lab}"`);
    phrase.lastUsedAt = Date.now();
    const now = nowMs();
    const since = now - this._lastTouchPersistT;
    if (since >= TOUCH_PERSIST_MS) {
      this._lastTouchPersistT = now;
      this._persist();
    } else if (!this._touchPersistTimer) {
      this._touchPersistTimer = setTimeout(() => {
        this._touchPersistTimer = 0;
        this._lastTouchPersistT = nowMs();
        this._persist();
      }, TOUCH_PERSIST_MS - since);
    }
  }

  /**
   * Export the full library as a JSON string (the storage shape, version 1).
   * @returns {string}
   */
  exportLibrary() {
    return JSON.stringify(this._library);
  }

  /**
   * Merge a previously exported library into this one. Validation is strict
   * and the merge is atomic: on any error nothing changes. Same-label phrases
   * have their takes concatenated; the merge is rejected if it would exceed
   * the caps (60 phrases, 8 takes per phrase).
   * @param {string} json JSON string from exportLibrary()
   * @returns {PhraseSummary[]} the merged library summary
   * @throws {SottoEngineError} 'invalid-library' | 'library-full'
   */
  importLibrary(json) {
    let obj;
    try {
      obj = JSON.parse(json);
    } catch {
      throw new SottoEngineError('invalid-library', 'import is not valid JSON');
    }
    validateLibraryShape(obj);
    // The optional learned/lastUsedAt fields are copied only when present
    // (absent = not learned, and absence itself is preserved). On a
    // same-label merge the local phrase's fields win: takes are added, but
    // a calibrated phrase never becomes learned (learned entries are
    // evictable; calibrated ones must not be).
    const copyMeta = (dst, src) => {
      if ('learned' in src) dst.learned = src.learned;
      if ('lastUsedAt' in src) dst.lastUsedAt = src.lastUsedAt;
      return dst;
    };
    const merged = this._library.phrases.map((p) => copyMeta({
      label: p.label, createdAt: p.createdAt, templates: p.templates.slice(),
    }, p));
    for (const inc of obj.phrases) {
      const existing = merged.find((p) => p.label === inc.label);
      if (existing) {
        const total = existing.templates.length + inc.templates.length;
        if (total > MAX_TAKES) {
          throw new SottoEngineError('library-full',
            `merging would give "${inc.label}" ${total} takes (max ${MAX_TAKES})`);
        }
        existing.templates = existing.templates.concat(deepCopyTemplates(inc.templates));
      } else {
        if (merged.length >= MAX_PHRASES) {
          throw new SottoEngineError('library-full',
            `merging would exceed ${MAX_PHRASES} phrases`);
        }
        merged.push(copyMeta({
          label: inc.label,
          createdAt: inc.createdAt,
          templates: deepCopyTemplates(inc.templates),
        }, inc));
      }
    }
    this._library = { version: 1, phrases: merged };
    this._persist();
    this._rebuildCache();
    return this.getLibrary();
  }

  /**
   * Rank every enrolled phrase against a segment, for practice mode. Applies
   * no accept threshold — every phrase appears, best first.
   * @param {Segment|number[][]} seg a segment (or a raw frames array)
   * @returns {{label: string, distance: number, confidence: number}[]}
   * @throws {SottoEngineError} 'invalid-frames' on a malformed segment
   */
  matchStats(seg) {
    const frames = Array.isArray(seg) ? seg : seg && seg.frames;
    validateFrames(frames, 'segment frames');
    this._prepareQuery(frames);
    const tauReject = REJECT_FACTOR * this._tauAccept();
    const out = [];
    for (const entry of this._cache) {
      let d = Infinity;
      for (const take of entry.takes) {
        const dist = dtwDistance(this._queryBuf, take, L_RESAMPLE, F_DIM, DTW_BAND, this._dtwScratch);
        if (dist < d) d = dist;
      }
      out.push({ label: entry.label, distance: d, confidence: clamp01(1 - d / tauReject) });
    }
    out.sort((a, b) => a.distance - b.distance);
    return out;
  }

  // -- diagnostics ----------------------------------------------------------

  /**
   * Snapshot of the live segmentation diagnostics (SPEC-V5): the EMA
   * energy, the live thresholds (tauOn includes the audio assist while
   * audio-active), the adaptive floor quantiles and idle sample count,
   * smoothed fps, engine state, and whether adaptive thresholds are engaged
   * (false = cold-start fixed values). Returns a fresh plain object; safe
   * to call at any rate, camera running or not.
   * @returns {{energy: number, tauOn: number, tauOff: number,
   *            floorP50: number, floorP90: number, samples: number,
   *            fps: number, state: string|null, adaptive: boolean}}
   */
  getDiagnostics() {
    return {
      energy: this._energy,
      tauOn: this._tauOn,
      tauOff: this._tauOff,
      floorP50: this._floorP50,
      floorP90: this._floorP90,
      samples: this._floorSamples,
      fps: this._fps,
      state: this._state,
      adaptive: this._floorSamples >= FLOOR_MIN_SAMPLES,
    };
  }

  // -- testing / simulation -------------------------------------------------

  /**
   * Run a frames array through the full segment pipeline exactly as a live
   * utterance would be: recording-if-armed, then onSegment, then matching and
   * onMatch. Works with the camera stopped. UI must label results simulated.
   * audioLevel is one provider poll at injection time (0 with no provider),
   * so tests can stub a voiced segment by installing a constant provider.
   * @param {number[][]} frames rows of 22 finite numbers (>= 2 rows)
   * @returns {Segment} the synthesized segment (timestamps assume ~30 fps)
   * @throws {SottoEngineError} 'invalid-frames'
   */
  debugInjectSegment(frames) {
    validateFrames(frames, 'injected frames');
    const copy = frames.map((row) => Array.from(row, Number));
    const t1 = nowMs();
    const durationMs = ((copy.length - 1) / 30) * 1000;
    const seg = {
      frames: copy, t0: t1 - durationMs, t1, durationMs,
      audioLevel: this._pollAudioLevel(),
    };
    this._finishSegment(seg);
    return seg;
  }

  /**
   * Build a plausible, deterministic fake utterance (~35 frames of 22 dims)
   * for camera-free testing. Kinds 'a' and 'b' have clearly distinct temporal
   * shapes: 'a' is two jaw-open bursts with lateral stretch (a "pa-pa"
   * gesture); 'b' is one slow pucker-and-funnel arc with a late smile (an
   * "ooo" gesture). Any other kind string yields a deterministic variant.
   * Same kind in, same frames out — distances between repeats are exactly 0.
   * @param {string} [kind='a'] which fake utterance to produce
   * @returns {number[][]} frames array suitable for debugInjectSegment()
   */
  static syntheticSegment(kind = 'a') {
    const key = String(kind);
    const h = hashString(key);
    const rand = mulberry32(h);
    const archetype = key === 'a' ? 0 : key === 'b' ? 1 : h % 2;
    const phase = rand() * Math.PI * 2;
    const speed = 0.9 + 0.2 * rand();
    // Two syllable centers for the 'a' archetype; derived kinds shift them.
    const c1 = key === 'a' ? 0.26 : 0.22 + 0.10 * rand();
    const c2 = key === 'a' ? 0.64 : 0.58 + 0.14 * rand();
    const N = 35;
    const gauss = (t, c, s) => Math.exp(-((t - c) * (t - c)) / (2 * s * s));
    const frames = [];
    for (let n = 0; n < N; n++) {
      const t = n / (N - 1);
      const f = new Array(F_DIM).fill(0);
      let jaw = 0, pucker = 0, funnel = 0, smile = 0, stretch = 0, close = 0, shrug = 0;
      if (archetype === 0) {
        const env = gauss(t * speed, c1, 0.09) + gauss(t * speed, c2, 0.09);
        jaw = 0.55 * env;
        stretch = 0.20 * env;
        shrug = 0.10 * env;
        funnel = 0.06 * env;
        smile = 0.05 * env;
      } else {
        const arc = Math.sin(Math.PI * clamp01(t * speed * 1.02));
        const env = Math.pow(arc < 0 ? 0 : arc, 1.5);
        pucker = 0.58 * env;
        funnel = 0.46 * env;
        close = 0.16 * env;
        jaw = 0.10 * env;
        smile = 0.24 * gauss(t, 0.86, 0.10);
        stretch = 0.04 * env;
        shrug = 0.12 * env;
      }
      const active = jaw + pucker > 0.05 ? 1 : 0;
      const ripple = 0.012 * Math.sin(2 * Math.PI * 3.2 * t + phase) * active;
      f[0] = clamp01(jaw + ripple);                    // jawOpen
      f[1] = clamp01(0.05 * jaw);                      // jawForward
      f[2] = clamp01(close);                           // mouthClose
      f[3] = clamp01(funnel + 0.6 * ripple);           // mouthFunnel
      f[4] = clamp01(pucker + ripple);                 // mouthPucker
      f[5] = clamp01(0.02 * stretch);                  // mouthLeft
      f[6] = clamp01(0.03 * stretch);                  // mouthRight
      f[7] = clamp01(smile);                           // mouthSmileLeft
      f[8] = clamp01(0.92 * smile);                    // mouthSmileRight
      f[9] = clamp01(0.03 * close);                    // mouthFrownLeft
      f[10] = clamp01(0.03 * close);                   // mouthFrownRight
      f[11] = clamp01(stretch);                        // mouthStretchLeft
      f[12] = clamp01(0.9 * stretch);                  // mouthStretchRight
      f[13] = clamp01(0.30 * close + 0.15 * pucker);   // mouthRollLower
      f[14] = clamp01(0.20 * close + 0.12 * funnel);   // mouthRollUpper
      f[15] = clamp01(shrug);                          // mouthShrugLower
      f[16] = clamp01(0.7 * shrug);                    // mouthShrugUpper
      f[17] = clamp01(0.04 * (jaw + pucker));          // cheekPuff
      // Geometry in raw inter-ocular units (matches live extraction ranges).
      f[18] = clampGeom(0.02 + 0.60 * jaw + 0.15 * ripple);          // aperture
      f[19] = clampGeom(0.55 + 0.10 * (smile + stretch) - 0.16 * pucker); // width
      f[20] = clampGeom(0.070 + 0.030 * funnel + 0.010 * jaw);       // upper lip
      f[21] = clampGeom(0.085 + 0.050 * jaw + 0.020 * close);        // lower lip
      frames.push(f);
    }
    return frames;
  }

  // -- internals: model & loop ----------------------------------------------

  async _ensureLandmarker() {
    if (this._landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks('vendor/mediapipe/wasm');
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: 'vendor/mediapipe/face_landmarker.task', delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
    try {
      this._landmarker = await FaceLandmarker.createFromOptions(fileset, options('GPU'));
    } catch (gpuErr) {
      // GPU delegate unavailable (headless, driver, blocklist): one CPU retry.
      this._landmarker = await FaceLandmarker.createFromOptions(fileset, options('CPU'));
    }
  }

  _scheduleFrame() {
    if (!this._running || document.hidden) return;
    if (this._useVfc) {
      this._vfcId = this._video.requestVideoFrameCallback(this._boundVfc);
    } else {
      this._rafId = requestAnimationFrame(this._boundRaf);
    }
  }

  _cancelScheduled() {
    if (this._vfcId && this._video && typeof this._video.cancelVideoFrameCallback === 'function') {
      this._video.cancelVideoFrameCallback(this._vfcId);
    }
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._vfcId = 0;
    this._rafId = 0;
  }

  _vfcStep() {
    this._vfcId = 0;
    if (!this._running) return;
    this._processFrame();
    this._scheduleFrame();
  }

  _rafStep() {
    this._rafId = 0;
    if (!this._running) return;
    // rAF can outpace the camera; only process genuinely new video frames.
    const mt = this._video.currentTime;
    if (mt !== this._lastMediaTime) {
      this._lastMediaTime = mt;
      this._processFrame();
    }
    this._scheduleFrame();
  }

  _handleVisibility() {
    if (document.hidden) {
      this._cancelScheduled();
      if (this._speaking) this._discardSegment();
      this._energy = 0;
      this._resetHysteresis();
      this._segLastT = -1;        // no dt accumulation across the gap
      this._haveFeatures = false; // skip one energy sample on resume
      this._ringHead = 0;         // drop stale pre-roll across the gap
      this._ringCount = 0;
    } else if (this._running) {
      this._lastFrameT = 0; // avoid a bogus fps spike after the gap
      // The EMA energy re-ramps from 0 after the gap; hold the noise floor
      // so the ramp does not read as a quieter face (same transient class
      // as a face reacquisition).
      this._floorHoldUntil = nowMs() + FLOOR_REACQ_FREEZE_MS;
      this._scheduleFrame();
    }
  }

  _processFrame() {
    const video = this._video;
    if (!this._landmarker || !video || video.readyState < 2) return;
    const t = nowMs();
    // FaceLandmarker VIDEO mode requires strictly increasing timestamps.
    let ts = Math.round(t);
    if (ts <= this._lastTs) ts = this._lastTs + 1;
    this._lastTs = ts;

    let result;
    try {
      result = this._landmarker.detectForVideo(video, ts);
    } catch (err) {
      const detail = 'landmark detection failed: ' + (err && err.message ? err.message : String(err));
      this._setState('error', detail);
      this.stop();
      return;
    }

    if (this._lastFrameT > 0) {
      const dt = t - this._lastFrameT;
      if (dt > 0) {
        const inst = 1000 / dt;
        this._fps = this._fps === 0 ? inst : this._fps + FPS_ALPHA * (inst - this._fps);
      }
    }
    this._lastFrameT = t;

    const lms = result.faceLandmarks && result.faceLandmarks[0];
    const bs = result.faceBlendshapes && result.faceBlendshapes[0];
    const faceOk = !!(lms && lms.length > LM_MOUTH_R && bs && bs.categories && bs.categories.length > 0);

    if (faceOk) {
      if (!this._blendIdx) this._buildBlendIndex(bs.categories);
      this._extractFeatures(lms, bs.categories, video);
      if (!this._faceOk) {
        // Face (re)acquired: freeze the noise floor while the EMA energy
        // settles (SPEC-V5: 500 ms post-reacquisition).
        this._floorHoldUntil = t + FLOOR_REACQ_FREEZE_MS;
        if (this._haveFeatures) this._faceRegained = true;
      }
      this._faceOk = true;
      this._faceEverSeen = true;
      this._lastFaceT = t;
      if (this._state === 'ready' || this._state === 'no-face') this._setState('idle');

      // Energy: mean scaled |delta| vs the previous frame, EMA-smoothed.
      // The first frame after start or after face loss contributes zero
      // (no trustworthy delta), which also prevents re-acquisition spikes.
      let raw = 0;
      if (this._haveFeatures && !this._faceRegained) {
        raw = frameEnergy(this._features, this._prevFeatures, INV_DIM_SCALE);
      }
      this._faceRegained = false;
      this._prevFeatures.set(this._features);
      this._haveFeatures = true;
      this._energy += EMA_ALPHA * (raw - this._energy);

      // Pre-roll ring push (copies into preallocated slots). The level slot
      // is zeroed here and overwritten with the polled audio level inside
      // _segmentStep, so frames pushed while paused never carry a stale
      // level from an earlier ring lap.
      const slot = this._ring[this._ringHead];
      slot.set(this._features);
      this._ringTs[this._ringHead] = t;
      this._ringLevels[this._ringHead] = 0;
      this._ringHead = (this._ringHead + 1) % RING_SIZE;
      if (this._ringCount < RING_SIZE) this._ringCount++;

      if (!this._paused) this._segmentStep(t);
    } else {
      // No face: hold the last features and suppress energy, so segments can
      // neither start nor continue. A segment in flight is discarded, and the
      // pre-roll ring is cleared so the next segment can only be seeded with
      // frames captured after the face is reacquired (stale pre-gap frames
      // would corrupt t0/duration and can push a segment past MAX_SEG_MS).
      this._faceOk = false;
      if (this._speaking) this._discardSegment();
      this._energy = 0;
      this._resetHysteresis();
      this._segLastT = -1; // no dt accumulation across the face gap
      this._ringHead = 0;
      this._ringCount = 0;
      if (this._faceEverSeen && this._state === 'idle' && t - this._lastFaceT > FACE_LOST_MS) {
        this._setState('no-face');
      }
    }

    if (this._onFrame) {
      const fi = this._frameInfo;
      fi.t = t;
      fi.mouthOpen = this._features[0];
      fi.faceOk = faceOk;
      fi.fps = this._fps;
      fi.energy = this._energy;
      fi.tauOn = this._tauOn;
      fi.tauOff = this._tauOff;
      this._emit(this._onFrame, fi);
    }
  }

  _buildBlendIndex(categories) {
    const byName = new Map();
    for (let i = 0; i < categories.length; i++) byName.set(categories[i].categoryName, i);
    const idx = new Int16Array(BLENDSHAPE_NAMES.length);
    for (let k = 0; k < BLENDSHAPE_NAMES.length; k++) {
      const i = byName.get(BLENDSHAPE_NAMES[k]);
      idx[k] = i === undefined ? -1 : i;
    }
    this._blendIdx = idx;
  }

  _extractFeatures(lms, categories, video) {
    const f = this._features;
    const bi = this._blendIdx;
    for (let k = 0; k < BLENDSHAPE_NAMES.length; k++) {
      const i = bi[k];
      f[k] = i >= 0 ? categories[i].score : 0;
    }
    // Landmarks are normalized to the frame; correct x by aspect so the
    // ratios below are isotropic regardless of camera resolution.
    const aspect = video.videoWidth > 0 && video.videoHeight > 0
      ? video.videoWidth / video.videoHeight : 16 / 9;
    const iod = landmarkDist(lms[LM_EYE_L], lms[LM_EYE_R], aspect);
    if (iod > 1e-6) {
      f[18] = clampGeom(landmarkDist(lms[LM_LIP_UP_IN], lms[LM_LIP_LO_IN], aspect) / iod);
      f[19] = clampGeom(landmarkDist(lms[LM_MOUTH_L], lms[LM_MOUTH_R], aspect) / iod);
      f[20] = clampGeom(landmarkDist(lms[LM_LIP_UP_OUT], lms[LM_LIP_UP_IN], aspect) / iod);
      f[21] = clampGeom(landmarkDist(lms[LM_LIP_LO_IN], lms[LM_LIP_LO_OUT], aspect) / iod);
    }
    // On a degenerate inter-ocular distance the geometric dims hold their
    // previous values (f persists between frames).
  }

  // -- internals: segmentation ----------------------------------------------

  _segmentStep(t) {
    // Audio+lip fusion: poll the provider at most once per frame (0 with no
    // provider) and record the level against the ring slot pushed for this
    // frame, so seeded pre-roll frames carry their true levels into
    // Segment.audioLevel. Energy is lips-only, so audio can only soften the
    // thresholds below — it can never start a segment by itself.
    const level = this._pollAudioLevel();
    this._ringLevels[(this._ringHead - 1 + RING_SIZE) % RING_SIZE] = level;
    // Real-frame dt for the ms hysteresis. The clamp keeps one janky frame
    // from standing in for a whole confirmation or stillness run.
    let dt = this._segLastT >= 0 ? t - this._segLastT : 0;
    if (dt < 0) dt = 0;
    else if (dt > HYST_DT_MAX_MS) dt = HYST_DT_MAX_MS;
    this._segLastT = t;
    // Live thresholds (SPEC-V5): adaptive once the floor has enough idle
    // samples, cold-start fixed values before that. The audio assist lowers
    // the start threshold exactly as in v0.4.
    this._refreshThresholds();
    const audioActive = level > AUDIO_ACTIVE_RMS;
    const tauOn = audioActive ? this._tauOnBase * AUDIO_TAU_ON_FACTOR : this._tauOnBase;
    this._tauOn = tauOn;
    // Hysteresis invariant: the release threshold must sit below the
    // effective start threshold, or a started segment's frames can count as
    // "still" and end it early. Reachable without this cap: tauOn clamped to
    // its max while the tauOff candidate is not, or the audio assist scaling
    // only tauOn. Report the capped value so the HUD shows the live pair.
    const tauOff = Math.min(this._tauOff, tauOn * 0.9);
    this._tauOff = tauOff;
    const e = this._energy;
    if (this._refractory) {
      // After discarding an over-long utterance: require stillness before
      // re-arming, so one continuous movement cannot retrigger instantly.
      // (The noise floor stays frozen through refractory.)
      if (e < tauOff) this._refractory = false;
      return;
    }
    if (!this._speaking) {
      // Idle with a face tracked: feed the noise-floor trackers unless a
      // post-reacquisition hold is active (speaking and refractory frames
      // never reach this point, so the floor is frozen there by
      // construction).
      if (t >= this._floorHoldUntil && this._state === 'idle') {
        this._floorUpdate(t, e);
      }
      if (e > tauOn) {
        if (this._onMs === 0) this._onRunStartT = t;
        this._onMs += dt;
        if (this._onMs >= ON_MS) this._beginSegment();
      } else {
        this._onMs = 0;
      }
      return;
    }
    // Active utterance: append the current frame (bounded copy — allowed).
    this._segFrames.push(Array.from(this._features));
    this._segTs.push(t);
    this._segLevels.push(level);
    if (t - this._segTs[0] > this._maxSegMs) {
      if (this._splitOnMax && !this._pendingRecording) {
        this._splitSegment(t);
      } else {
        // v0.1 behavior, kept as the default and during calibration
        // recording so phrase takes stay bounded.
        this._discardSegment();
        this._refractory = true;
      }
      return;
    }
    // While audio stays active a lip-still moment needs twice the stillness
    // to end the utterance (bounded: the maxSegMs check above runs first).
    const offNeededMs = audioActive ? OFF_MS * AUDIO_OFF_FRAMES_FACTOR : OFF_MS;
    if (e < tauOff) {
      this._offMs += dt;
      this._offRun++;
      if (this._offMs >= offNeededMs) this._endSegment();
    } else {
      this._offMs = 0;
      this._offRun = 0;
    }
  }

  /**
   * Recompute the base thresholds from the adaptive floor (SPEC-V5):
   *   tauOn  = clamp(p50 + K * spread, TAU_ON_MIN, TAU_ON_MAX)
   *   tauOff = max(p50 + K_OFF * spread, tauOn * TAU_OFF_MIN_RATIO)
   * with spread = p90 - p50 + FLOOR_EPS and K = K_ON * (1.6 - sensitivity):
   * higher sensitivity lowers the bar (setSensitivity keeps its DTW-accept
   * role unchanged). Until FLOOR_MIN_SAMPLES idle samples exist, the fixed
   * v0.4 TAU_ON/TAU_OFF apply (cold-start safety).
   * @returns {void}
   */
  _refreshThresholds() {
    if (this._floorSamples >= FLOOR_MIN_SAMPLES) {
      const spread = this._floorP90 - this._floorP50 + FLOOR_EPS;
      const kOn = K_ON * (1.6 - this._sensitivity);
      const tauOn = clamp(this._floorP50 + kOn * spread, TAU_ON_MIN, TAU_ON_MAX);
      const off = this._floorP50 + K_OFF * spread;
      const offMin = tauOn * TAU_OFF_MIN_RATIO;
      this._tauOnBase = tauOn;
      this._tauOff = off > offMin ? off : offMin;
    } else {
      this._tauOnBase = TAU_ON;
      this._tauOff = TAU_OFF;
    }
  }

  /**
   * One idle sample into the streaming noise-floor estimate. Method (see
   * the tunables comment): stochastic pinball-loss quantile trackers for
   * p50 and p90 — q += 2*w*s*(p - I(x <= q)) — with per-sample weight
   * w = 1 - exp(-dt / FLOOR_TC_MS) from the real inter-sample dt (clamped
   * to FLOOR_DT_MAX_MS across gaps) and step scale s = EMA of |x - p50|
   * floored at FLOOR_DEV_MIN. The gain is lower-bounded by 1/(n+1) as a
   * decreasing-gain burn-in. Effective window is roughly FLOOR_WINDOW_MS
   * regardless of fps.
   * @param {number} t sample timestamp (ms)
   * @param {number} x current EMA energy
   * @returns {void}
   */
  _floorUpdate(t, x) {
    if (this._floorSamples === 0) {
      this._floorP50 = x;
      this._floorP90 = x;
      this._floorDev = 0;
      this._floorSamples = 1;
      this._floorLastT = t;
      return;
    }
    let dt = t - this._floorLastT;
    this._floorLastT = t;
    if (!(dt > 0)) dt = 1;
    else if (dt > FLOOR_DT_MAX_MS) dt = FLOOR_DT_MAX_MS;
    let w = 1 - Math.exp(-dt / FLOOR_TC_MS);
    const burnIn = 1 / (this._floorSamples + 1);
    if (burnIn > w) w = burnIn;
    const dev = Math.abs(x - this._floorP50);
    this._floorDev += w * (dev - this._floorDev);
    const s = this._floorDev > FLOOR_DEV_MIN ? this._floorDev : FLOOR_DEV_MIN;
    this._floorP50 += 2 * w * s * (x <= this._floorP50 ? -0.5 : 0.5);
    this._floorP90 += 2 * w * s * (x <= this._floorP90 ? -0.1 : 0.9);
    if (this._floorP50 < 0) this._floorP50 = 0;
    if (this._floorP90 < this._floorP50) this._floorP90 = this._floorP50;
    this._floorSamples++;
  }

  /** Reset the ms hysteresis accumulators (both directions). */
  _resetHysteresis() {
    this._onMs = 0;
    this._offMs = 0;
    this._offRun = 0;
  }

  /**
   * Read the audio-level provider defensively. An exception counts as level
   * 0, and after AUDIO_PROVIDER_MAX_THROWS consecutive exceptions the
   * provider is cleared so a broken provider cannot poison every frame.
   * Non-finite returns are treated as 0 (without counting as exceptions).
   * @returns {number} current mic RMS clamped to 0..1, or 0
   */
  _pollAudioLevel() {
    const fn = this._audioLevelProvider;
    if (!fn) return 0;
    let v;
    try {
      v = Number(fn());
    } catch (err) {
      if (++this._audioProviderThrows >= AUDIO_PROVIDER_MAX_THROWS) {
        this._audioLevelProvider = null;
        this._audioProviderThrows = 0;
        console.warn('sotto: audio level provider cleared after repeated exceptions —', err);
      }
      return 0;
    }
    this._audioProviderThrows = 0;
    return Number.isFinite(v) ? clamp01(v) : 0;
  }

  /**
   * Append every ring frame with timestamp >= cutoffT (oldest first) to the
   * in-progress segment arrays. Ring timestamps increase, so this takes the
   * newest contiguous run — "whatever ring frames fall inside" the pre-roll
   * window (SPEC-V5).
   * @param {number} cutoffT ms timestamp lower bound (inclusive)
   * @returns {void}
   */
  _seedFromRing(cutoffT) {
    for (let k = this._ringCount; k >= 1; k--) {
      const idx = (this._ringHead - k + RING_SIZE * 2) % RING_SIZE;
      if (this._ringTs[idx] < cutoffT) continue;
      this._segFrames.push(Array.from(this._ring[idx]));
      this._segTs.push(this._ringTs[idx]);
      this._segLevels.push(this._ringLevels[idx]);
    }
  }

  /**
   * splitOnMax: the in-flight utterance reached maxSegMs. Emit it as a
   * completed segment (recordingLabel handling unchanged — though splitting
   * never fires while a recording is armed), then immediately continue the
   * speaking state with a fresh segment seeded from the ring frames inside
   * PRE_ROLL_MS of the split point, which overlap the emitted tail
   * (deliberate closing/opening context). Energy and hysteresis accumulators
   * carry over untouched; no refractory, no discard, no state change. A
   * later true still-tail ends the final piece through _endSegment as usual.
   * @param {number} t timestamp of the frame that crossed maxSegMs
   * @returns {void}
   */
  _splitSegment(t) {
    const frames = this._segFrames;
    const ts = this._segTs;
    const seg = {
      frames,
      t0: ts[0],
      t1: ts[ts.length - 1],
      durationMs: ts[ts.length - 1] - ts[0],
      audioLevel: meanLevel(this._segLevels),
    };
    // Reseed before emitting so callbacks observe a consistent engine. The
    // fresh level array starts from the reseeded ring frames only, so each
    // split piece averages exactly its own span.
    this._segFrames = [];
    this._segTs = [];
    this._segLevels = [];
    this._seedFromRing(t - PRE_ROLL_MS);
    this._finishSegment(seg);
  }

  _beginSegment() {
    // Seed the segment with the on-run frames plus whatever ring frames
    // fall inside PRE_ROLL_MS before the run started (the most recent ring
    // entry is the current frame).
    this._segFrames = [];
    this._segTs = [];
    this._segLevels = [];
    this._seedFromRing(this._onRunStartT - PRE_ROLL_MS);
    this._speaking = true;
    this._resetHysteresis();
    this._setState('speaking');
  }

  _discardSegment() {
    this._speaking = false;
    this._segFrames = null;
    this._segTs = null;
    this._segLevels = null;
    this._resetHysteresis();
    if (this._state === 'speaking') this._setState('idle');
  }

  _endSegment() {
    const frames = this._segFrames;
    const ts = this._segTs;
    const levels = this._segLevels;
    // The trailing still run is exactly the frames whose accumulated dt
    // reached OFF_MS (doubled under audio fusion); keep a few as closing
    // context and drop the rest so segments are not padded with 400 ms or
    // more of nothing.
    const stillRun = this._offRun;
    this._speaking = false;
    this._segFrames = null;
    this._segTs = null;
    this._segLevels = null;
    this._resetHysteresis();
    this._setState('idle');
    const keep = frames.length - (stillRun - TAIL_KEEP);
    // Too short — discard silently. Duration-based per SPEC-V5 (MIN_SEG_MS);
    // the 2-row floor keeps downstream frame validation satisfied.
    if (keep < 2 || ts[keep - 1] - ts[0] < MIN_SEG_MS) return;
    frames.length = keep;
    ts.length = keep;
    levels.length = keep; // audioLevel averages only the emitted frames
    const seg = {
      frames,
      t0: ts[0],
      t1: ts[keep - 1],
      durationMs: ts[keep - 1] - ts[0],
      audioLevel: meanLevel(levels),
    };
    this._finishSegment(seg);
  }

  _finishSegment(seg) {
    if (this._pendingRecording) {
      const label = this._pendingRecording;
      this._pendingRecording = null;
      seg.recordingLabel = label;
      try {
        this._storeTemplate(label, seg.frames);
      } catch (err) {
        seg.recordingError = err && err.message ? err.message : String(err);
      }
      if (this._onSegment) this._emit(this._onSegment, seg);
      return;
    }
    if (this._onSegment) this._emit(this._onSegment, seg);
    const m = this._matchFrames(seg.frames);
    if (this._onMatch) this._emit(this._onMatch, m, seg);
  }

  // -- internals: matching --------------------------------------------------

  _tauAccept() {
    return TAU_ACCEPT_BASE + TAU_ACCEPT_SPAN * this._sensitivity;
  }

  _prepareQuery(frames) {
    resampleSequence(frames, L_RESAMPLE, F_DIM, this._queryBuf);
    scaleDimsInPlace(this._queryBuf, L_RESAMPLE, F_DIM, INV_DIM_SCALE);
  }

  _matchFrames(frames) {
    if (this._cache.length === 0 || frames.length < 2) return null;
    this._prepareQuery(frames);
    let best = Infinity, best2 = Infinity, bestLabel = null;
    for (const entry of this._cache) {
      let d = Infinity;
      for (const take of entry.takes) {
        const dist = dtwDistance(this._queryBuf, take, L_RESAMPLE, F_DIM, DTW_BAND, this._dtwScratch);
        if (dist < d) d = dist;
      }
      if (d < best) {
        best2 = best;
        best = d;
        bestLabel = entry.label;
      } else if (d < best2) {
        best2 = d;
      }
    }
    const tauAccept = this._tauAccept();
    if (best >= tauAccept) return null;
    const margin = best > 0 ? best2 / best : Infinity;
    if (margin < MARGIN_RATIO) return null; // ambiguous between two phrases
    const tauReject = REJECT_FACTOR * tauAccept;
    return {
      label: bestLabel,
      distance: best,
      confidence: clamp01(1 - best / tauReject),
      margin,
    };
  }

  _rebuildCache() {
    // Templates are resampled and dim-scaled once here, so live matching only
    // prepares the query and runs allocation-free DTW per take.
    this._cache = this._library.phrases.map((p) => ({
      label: p.label,
      takes: p.templates.map((tpl) =>
        scaleDimsInPlace(
          resampleSequence(tpl, L_RESAMPLE, F_DIM),
          L_RESAMPLE, F_DIM, INV_DIM_SCALE)),
    }));
  }

  // -- internals: library ---------------------------------------------------

  _findPhrase(label) {
    return this._library.phrases.find((p) => p.label === label) || null;
  }

  _storeTemplate(label, frames) {
    let phrase = this._findPhrase(label);
    if (!phrase) {
      if (this._library.phrases.length >= MAX_PHRASES) {
        throw new SottoEngineError('library-full', `library is capped at ${MAX_PHRASES} phrases`);
      }
      phrase = { label, createdAt: Date.now(), templates: [] };
      this._library.phrases.push(phrase);
    }
    if (phrase.templates.length >= MAX_TAKES) {
      throw new SottoEngineError('phrase-full', `"${label}" already has ${MAX_TAKES} takes`);
    }
    phrase.templates.push(frames.map((row) => Array.from(row, Number)));
    this._persist();
    this._rebuildCache();
  }

  _loadLibrary() {
    const empty = { version: 1, phrases: [] };
    if (!this._storage) return empty;
    let raw = null;
    try {
      raw = this._storage.getItem(STORAGE_KEY);
    } catch {
      return empty;
    }
    if (!raw) return empty;
    try {
      const parsed = JSON.parse(raw);
      validateLibraryShape(parsed);
      return parsed;
    } catch (err) {
      // Corrupt stored data: start empty rather than crash. The stored value
      // is left in place until the next successful persist overwrites it.
      console.warn('sotto: stored library is invalid, starting empty —', err);
      return empty;
    }
  }

  _persist() {
    // A full write covers any pending throttled touchPhrase() write.
    if (this._touchPersistTimer) {
      clearTimeout(this._touchPersistTimer);
      this._touchPersistTimer = 0;
    }
    if (!this._storage) return;
    try {
      this._storage.setItem(STORAGE_KEY, JSON.stringify(this._library));
    } catch (err) {
      console.warn('sotto: could not persist library (storage full or blocked) —', err);
    }
  }

  // -- internals: misc ------------------------------------------------------

  _resetRuntime() {
    this._energy = 0;
    this._resetHysteresis();
    this._segLastT = -1;
    this._onRunStartT = 0;
    this._speaking = false;
    this._segFrames = null;
    this._segTs = null;
    this._segLevels = null;
    this._refractory = false;
    this._haveFeatures = false;
    this._faceOk = false;
    this._faceEverSeen = false;
    this._faceRegained = false;
    this._ringHead = 0;
    this._ringCount = 0;
    this._fps = 0;
    this._lastFrameT = 0;
    this._lastMediaTime = -1;
    // The noise floor is re-learned each start(): camera, lighting, and
    // seating may all have changed. Cold-start fixed thresholds cover the
    // first FLOOR_MIN_SAMPLES idle frames (~2 s at 30 fps).
    this._floorP50 = 0;
    this._floorP90 = 0;
    this._floorDev = 0;
    this._floorSamples = 0;
    this._floorLastT = 0;
    this._floorHoldUntil = 0;
    this._tauOnBase = TAU_ON;
    this._tauOn = TAU_ON;
    this._tauOff = TAU_OFF;
    // _lastTs is intentionally NOT reset: the detector requires strictly
    // increasing timestamps across stop()/start() with a kept model.
  }

  _setState(state, detail) {
    if (state === this._state && detail === this._stateDetail) return;
    this._state = state;
    this._stateDetail = detail;
    if (this._onState) this._emit(this._onState, state, detail);
  }

  _emit(cb, a, b) {
    try {
      cb(a, b);
    } catch (err) {
      console.error('sotto: callback threw —', err);
    }
  }
}

export default SottoEngine;
