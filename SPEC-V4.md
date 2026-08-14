# Sotto v0.4 — Unified multimodal mode: one flow, auto-routing, self-learning vocabulary

The Phrases/Whisper mode switch is REMOVED. There is one mode. Per utterance,
the app routes automatically; the user never chooses. Calibration becomes
optional: the silent vocabulary grows automatically from speech ("say it once,
mouth it silently after"). Physics honesty is unchanged and stays in the copy:
arbitrary NEVER-heard silent speech is not readable by anyone; Sotto's silent
coverage is exactly the phrases it has learned or been taught.

## 1. Permission & lifecycle (js/app.js)

- One "Start" flow: Start camera as today. If the voice-assist setting is on
  (localStorage 'sotto.voice.v1', default ON), after the camera is running the
  app starts the mic and lazily loads the ASR exactly as v0.3's enterWhisper
  did (deferred-gesture rules preserved: nothing touches the mic before a user
  gesture; page-load restore never hot-mics).
- Mic denied / unavailable / disconnected mid-session → voice assist drops to
  OFF with the existing mic empty-state copy adapted ("Silent-only until you
  re-enable the microphone") — camera flow continues; retry affordance stays.
- Settings drawer: "Voice assist" switch replaces the mode radiogroup
  semantics; turning it off releases the mic immediately (v0.3 leaveWhisper
  semantics: ticker cleared, queue cleared, provider null) but keeps
  splitOnMax ON (long silent mouthing should also split rather than discard —
  deliberate v0.4 change); turning it on re-runs the mic+ASR entry.
- Engine wiring when mic is live: setAudioLevelProvider + splitOnMax as in
  v0.3. When mic is off: provider null, splitOnMax STAYS true (see above),
  segment matching still runs.
- localStorage 'sotto.mode.v1' is obsolete: delete the key on init; remove all
  mode radiogroup UI and code. The nav pill vocabulary stays (watching /
  speaking… / transcribing…).

## 2. Per-utterance routing (js/app.js)

Engine tags every completed segment with `audioLevel` (see section 4). Rules,
in order, for each onSegment/onMatch pair (engine already DTW-matches every
segment; onMatch fires with match|null after onSegment):

1. Calibration recording pending → exactly today's behavior (takes flow).
2. If onMatch fired with confidence >= PHRASE_FAST_CONF (0.80) → type the
   phrase immediately (toast as today), DO NOT send to Whisper (instant path;
   also update that phrase's lastUsedAt). This applies whether the segment was
   silent or voiced — known phrases are instant either way.
3. Else if seg.audioLevel >= AUDIO_ACTIVE_RMS (engine constant, 0.012) AND
   voice assist is on AND ASR is ready → Whisper path exactly as v0.3
   (slice ±250ms, queue cap 3, streaming ticker, pause rules). On final text:
   type it, then run AUTO-LEARN (section 3).
4. Else (silent, no confident match) → nothing types; show the throttled
   silent-miss hint (section 5). A low-confidence match (below 0.80) types
   NOTHING in this branch (it would type wrong words; the hint is better).

Ordering note: onMatch fires synchronously after onSegment in the engine —
restructure the app's handlers so routing happens once per segment with both
pieces available (e.g. buffer the segment in onSegment, decide in onMatch,
which always follows; guard for segments discarded between the two).

## 3. Auto-learning (js/app.js + engine.enrollSegment)

After a Whisper final text T for segment S, auto-enroll S's lip frames as a
silent template for T when ALL hold:
- T, stripped of surrounding punctuation and collapsed whitespace, is 1-4
  words, 2-40 chars, and matches /^[\p{L}\p{N}' -]+$/u (no URLs, no symbols).
- Segment duration 400-3000 ms; not a split-piece mid-stream? (split pieces
  ARE eligible — they carry their own frames — but only if the FINAL text for
  that piece alone met the rules).
- The pad actually received T (not paused, not stale-session).
- Label = the stripped text, lowercased. If a phrase with that label exists
  (calibrated or learned), add as an extra take (engine handles the 8-take
  cap by evicting take 0 — same guard as calibration re-record). Else create
  it flagged learned.
- Caps: at most LEARNED_MAX = 40 learned phrases; creating one beyond the cap
  (or beyond the engine's 60 total) first evicts the learned phrase with the
  oldest (lastUsedAt ?? createdAt) — NEVER evict a calibrated phrase. If the
  library is 60/60 with fewer than needed learned entries, skip enrollment
  silently.
- First-time learn of a label → one toast: `learned "<label>" — mouth it
  silently next time` (no toast for added takes; no toast spam: at most one
  learn toast per 10s, extras silent).

## 4. Engine additions (js/engine.js) — additive only, v0.1 behavior untouched

- `Segment.audioLevel`: mean of the polled provider values over the segment's
  frames (0 when no provider). Set on every emitted segment including splits.
- `enrollSegment(label, frames, {learned = false} = {})`: public method —
  validates label (normalizeLabel) and frames (validateFrames), enforces
  MAX_PHRASES/MAX_TAKES with the same semantics as the recording path except:
  when the phrase exists and is at the take cap, evict take 0 and add (no
  throw); when creating, store `learned: !!learned` and `lastUsedAt: null`.
  Persists. Returns {label, templates}.
- Library entries gain optional fields `learned` (boolean) and `lastUsedAt`
  (number|null); getLibrary() exposes both; a new `touchPhrase(label)` sets
  lastUsedAt = now and persists (throttle persistence: at most one write per
  5s for touches). validateLibraryShape/import tolerate + preserve both
  fields (absent = not learned). Storage stays 'sotto.library.v1' — additive
  optional fields, old libraries load unchanged.
- No changes to matching, thresholds, or segmentation beyond reading
  audioLevel that _segmentStep already polls.

## 5. UI changes (app.html, css/app.css, js/app.js)

- Mode radiogroup: removed. In its place a compact status strip: "Silent
  phrases + voice assist" with a live mic state chip (on / off / blocked) that
  is a button opening settings.
- Phrasebook: entries show a small "learned" pill (--ink-soft outline) for
  learned:true; calibrated entries unchanged. "Add phrase" (manual
  calibration) stays; the STARTER-PACK AUTO-OFFER IS REMOVED (delete the
  first-run modal offer; keep a "Teach the starter six" button in the
  settings drawer and an inline link in the phrasebook empty state). The
  phrasebook empty-state copy becomes: silence works for phrases Sotto has
  learned — whisper things once and they show up here, or teach phrases
  manually.
- Silent-miss hint: when routing rule 4 fires, show in the ticker (not a
  toast): `couldn't read that silently — whisper it once and Sotto learns it`
  for 4s, throttled to once per 30s. aria-live already on the ticker.
- Per-mode privacy line replaced by one honest line that reflects mic state:
  mic on → "Camera and microphone are live. Everything runs on this device;
  nothing is uploaded."; mic off/blocked → "Camera only. The microphone is
  off; silent phrases still work."
- Pause toggle label: "Pause input" (covers both paths; behavior: pauses
  segmentation as today + drops queued/in-flight results per v0.2 rules).

## 6. Copy (index.html, README.md, sw.js)

- Landing hero/lede: one-mode story — mouth a known phrase in silence, or
  whisper anything; it types both, and it learns your lips while you speak so
  your silent vocabulary grows on its own. Keep the honesty strip (updated
  wording: silent coverage = taught + learned phrases; never-heard silent
  speech stays out of reach for everyone).
- How-it-works ledger becomes: 01 Start (camera, optional mic) / 02 Say it or
  mouth it (auto-routing explained in one breath) / 03 It types — and learns
  (the say-once-mouth-later loop). Pipeline paragraph: add one sentence about
  auto-enrollment (audio supervises the visual templates; on-device).
- FAQ updates: "Do I have to calibrate?" → No — optional; speaking teaches it
  automatically; fully-silent-from-scratch still needs taught/learned
  phrases and why. Keep audio-privacy answer, updated for voice assist
  (default on, one switch, mic released instantly when off).
- README: rewrite mode sections into the unified story (routing rules stated
  plainly, auto-learn guards listed, privacy per mic state). Footer/version
  strings → v0.4. sw.js CACHE → 'sotto-v4'.

## Honesty (binding)

No claim that silent open-vocabulary works. The learning loop is described
exactly: it recognizes silently only what it has been taught or has heard you
say. Banned words stay banned. State the instant-path latency (~10ms match)
and Whisper path (~1s warm) as measured.
