# Sotto v0.5 — Self-calibrating spec: adaptive thresholds, forgiving routing, correction loop, diagnostics

Motivation (be precise about the failure): every v0.1-0.4 threshold was derived
from synthetic tests. On a real face, (a) fixed TAU_ON/TAU_OFF may sit above or
below the user's actual resting noise, so segments under- or over-trigger; (b)
frame-count hysteresis breaks off 30fps; (c) the fixed 0.012 audio gate can
route real whispers to the silent-miss path; (d) the 0.80 instant-path gate is
MISCALIBRATED BY CONSTRUCTION: confidence = 1 - d/(2*tau_accept), so a good
match at d = 0.5*tau_accept scores 0.75 and fails the gate even though the
engine accepted it with margin. v0.5 removes fixed guesses from the critical
path.

## 1. Adaptive segmentation (js/engine.js)

- Noise floor: while idle with a face tracked (state 'idle', not speaking),
  maintain a rolling estimate of energy noise: exponential quantile trackers
  for the median (p50) and p90 of per-frame energy over an effective window of
  ~10s (frugal streaming quantile estimation is fine; document the method).
  Freeze the floor while speaking, during refractory, and for 500ms after
  face reacquisition.
- Dynamic thresholds (replacing fixed TAU_ON/TAU_OFF in _segmentStep):
  tauOn = clamp(p50 + K_ON * (p90 - p50 + EPS), TAU_ON_MIN, TAU_ON_MAX)
  tauOff = max(p50 + K_OFF * (p90 - p50 + EPS), tauOn * 0.45)
  with K_ON = 4.0, K_OFF = 1.5, EPS = 0.0008, TAU_ON_MIN = 0.006,
  TAU_ON_MAX = 0.030. Until the trackers have >= 60 idle samples, use the
  v0.4 fixed values (cold-start safety). Sensitivity (existing slider) scales
  K_ON: effectiveK = K_ON * (1.6 - sensitivity) so higher sensitivity lowers
  the bar (range 0..1 -> 1.6x..0.6x); setSensitivity keeps its DTW role too.
- Time-based hysteresis: replace ON_FRAMES/OFF_FRAMES counts with durations
  measured from real frame timestamps — ON_MS = 90, OFF_MS = 400 (doubled
  while audio-active as today: 800). MIN_SEG duration stays frame-count-free:
  MIN_SEG_MS = 320. PRE_ROLL becomes PRE_ROLL_MS = 130 (take whatever ring
  frames fall inside it). All conversions must behave identically at 30fps
  and sanely at 15/60fps.
- The audio-fusion start assist stays: while audio-active, tauOn *= 0.55.
- FrameInfo gains: energy (current EMA energy), tauOn, tauOff (the live
  thresholds), so the app can draw the diagnostics HUD from onFrame alone.
- getDiagnostics() -> {energy, tauOn, tauOff, floorP50, floorP90, samples,
  fps, state, adaptive: boolean} snapshot for the report.
- CONSTANTS: add the new tunables. Existing exported names stay (other code
  reads MAX_SEG_MS etc.). This changes live behavior only in _segmentStep
  threshold selection and timing; DTW matching is untouched.

## 2. Forgiving routing (js/app.js)

Replace v0.4's rule 2/3/4 with:
- Rule 2a (silent segment, i.e. audioLevel below the voiced bar): if the
  engine ACCEPTED a match (m !== null — acceptance already encodes threshold
  + margin), type it instantly. No secondary confidence gate. Delete
  PHRASE_FAST_CONF as a routing gate for silent segments.
- Rule 2b (voiced segment): if m !== null AND m.confidence >= 0.60, type
  instantly (phrase beats Whisper for speed/consistency); else fall through
  to Whisper (audio is ground truth when we have it).
- Rule 3 (voiced, no strong match): Whisper as today.
- Rule 4 (SILENT MISS — the big change): if voice assist is on and the mic is
  running, DO NOT give up — slice the audio window anyway and send it to
  Whisper ("speculative pass"): real whispers mis-gated by the voiced bar are
  recovered; genuinely silent audio returns '' via the worker's silence
  suppression, and only THEN show the silent-miss hint. If the speculative
  pass returns text, it types (and auto-learn runs). The voiced bar therefore
  stops being a correctness gate; it only decides rule ORDER. Skip the
  speculative pass only when the ASR is not ready or the mic is off (hint
  directly). Ticker: show the hint only after '' comes back; while the
  speculative pass runs show nothing (no false "transcribing…" pill —
  actually DO show transcribing, it is true).
- Voiced bar (app side): voiced = seg.audioLevel >= max(0.008, micFloor * 3)
  where micFloor is a rolling p50 of mic.level() sampled ~4x/s while the
  status is idle (not during speaking). Cold start: 0.008. This replaces the
  raw CONSTANTS.AUDIO_ACTIVE_RMS comparison in routing (the engine's internal
  fusion assist keeps its own constant — fine).

## 3. Correction loop (js/app.js)

- Keep the most recent unmatched-silent segment (frames, t1, durationMs) for
  CORRECTION_WINDOW_MS = 8000 (one slot; overwritten by newer misses; cleared
  on camera stop / voice off).
- When a Whisper final text T qualifies for auto-learn (existing guards) AND
  the slot holds a miss whose duration is within [0.5x, 2x] of the voiced
  segment's duration AND whose t1 is within the window: enroll the SILENT
  frames under T as well (a second enrollSegment call, learned: true), then
  clear the slot. Toast stays single ("learned ..." once) — no extra toast
  for the silent take. This teaches true silent articulation, which differs
  measurably from voiced articulation for the same words.

## 4. Diagnostics (js/app.js, app.html, css/app.css)

- Settings drawer gains a "Diagnostics" block with a toggle. When on, a
  compact HUD card renders under the camera card: live numeric readouts
  (fps, energy vs tauOn/tauOff as a small live bar pair, mic level vs voiced
  bar, adaptive floor p50/p90 + sample count, ASR device/model/state, library
  size taught/learned) plus the last 5 events (segment emitted with duration
  + audioLevel + route taken + match label/distance/confidence or miss;
  whisper final or ''; learn events). Ring buffer of events, plain text list,
  newest first, no scrollback beyond 5.
- "Copy report" button: puts a compact JSON on the clipboard: {ua, ts,
  engine: getDiagnostics(), asr: {device, model, ready}, mic: {running,
  floor}, lib: {taught, learned}, lastEvents: [...last 20]}. Toast "report
  copied". This is what the user pastes back when something misbehaves.
- HUD styling: card, mono numerics, tokens only, both themes; zero cost when
  the toggle is off (no listeners doing work beyond cheap assignments).

## 5. Defaults & copy (small)

- Default sensitivity moves 0.50 -> 0.60 (slider unchanged otherwise).
- index.html FAQ: add one Q "It's not catching my phrases — what do I do?"
  with candid tuning steps (face the camera, mouth deliberately, sensitivity
  up, practice mode to see near-misses, diagnostics report). Footer v0.5.
- README: new short "Self-calibration" subsection (what adapts, what is
  fixed); routing section updated to the new rules including the speculative
  pass; diagnostics documented. sw.js CACHE -> 'sotto-v5'.

## Honesty (binding)

The words 'perfect' and 'flawless' stay banned everywhere. Numbers stay the
measured ones. The speculative pass is described plainly (misses cost one
silent ASR pass when the mic is on).
