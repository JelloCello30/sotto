# Sotto

*sotto voce* — under the voice.

Sotto is a quiet-speech input instrument with one flow. Start it, then speak,
whisper, or mouth in silence: per utterance, Sotto routes on its own. A phrase
it recognizes types instantly from lip movement alone; anything it can hear is
transcribed by a vendored Whisper model on-device. And while you speak, it
learns — the lip motion of short spoken phrases is enrolled under their
transcripts, so your silent vocabulary grows without a calibration session.

Everything runs in your browser, on your machine. Nothing is uploaded.

It is a **research preview**, and it is honest about what that means. In
silence, Sotto recognizes exactly the phrases it has been taught or has heard
you say — a list that grows as you speak, but a list. Arbitrary never-heard
silent speech is not readable by Sotto or by anyone else; open-vocabulary
silent lipreading does not exist yet. With at least a whisper, the vocabulary
is open. Within those constraints it works, and it is quietly satisfying to
use.

## How it works

The whole pipeline runs in your browser, on your device:

1. **Camera, and usually a microphone.** `getUserMedia` video first. If voice
   assist is on (it is by default), the microphone starts after the camera is
   running — never before a user gesture, and never on a page-load restore.
   Deny or disable the mic and Sotto carries on camera-only.
2. **Face tracking.** A vendored copy of MediaPipe Tasks Vision (`FaceLandmarker`,
   WASM) tracks one face per frame, GPU delegate with CPU fallback.
3. **Features.** Each frame is reduced to a 22-dimensional vector: 18
   mouth-relevant blendshape scores (jaw, lips, cheeks) plus 4 geometric
   measurements — lip aperture, lip width, upper- and lower-lip height — each
   normalized by inter-ocular distance so the numbers survive you leaning toward
   the camera.
4. **Segmentation.** Frame-to-frame feature energy, smoothed and run through an
   on/off hysteresis, cuts the stream into utterance segments: mouth starts
   moving, mouth stops moving, that was an utterance. The on/off thresholds
   calibrate themselves to your resting face, and the hysteresis is timed in
   milliseconds rather than frames (see Self-calibration below). Each segment
   also carries its mean audio level, so routing knows whether it was voiced
   or silent.
5. **Matching.** Every segment is resampled to 32 frames and compared against
   your phrase library — taught and learned alike — with dynamic time warping
   (Sakoe-Chiba band). A phrase is accepted only if the best match is close
   enough *and* clearly ahead of the runner-up.
6. **Routing.** What happens next is decided per utterance (rules below): type
   the matched phrase, hand the audio to Whisper, or say nothing.
7. **Whisper branch.** For voiced utterances, the segmentation timestamps clip
   a padded window from a rolling in-memory microphone buffer; the clipped
   audio is transcribed by a local Whisper model in a worker, and the text
   streams into the composer as it is recognized.
8. **Output.** Typed text lands in the composer. Your phrase library lives in
   `localStorage` and can be exported and imported as JSON.

## Routing, stated plainly

Per utterance, in order:

1. If a calibration recording is pending, the segment is a take — exactly the
   manual-teaching flow.
2. If the segment was silent (audio level under the voiced bar) and the matcher
   accepted a phrase, it types immediately — about 10 ms. Acceptance already
   demands a close match that is clearly ahead of the runner-up, so there is no
   second confidence gate. (v0.4 had one, at 0.80, and by construction it
   rejected good matches the engine had already accepted with margin. It is
   gone.)
3. If the segment was voiced and the matcher accepted a phrase with confidence
   at or above 0.60, it also types immediately — the phrase path beats Whisper
   on speed and consistency. Below that bar the match is discarded and the
   utterance falls through to the next rule: when audio exists, audio is
   ground truth.
4. If the segment was voiced with no phrase typed, the clipped audio goes to
   Whisper and the transcript types as it streams — about a second per
   utterance once warm.
5. If the segment was silent with no match — and voice assist is on with the
   model ready — Sotto does not give up. The audio window is clipped and sent
   to Whisper anyway: the speculative pass. Stated plainly, a silent miss
   costs one quiet ASR pass whenever the mic is on. If you actually whispered
   and the voiced bar mis-filed it as silence, the transcript comes back,
   types, and feeds auto-learning like any other. If the audio was genuinely
   silent, the worker's silence suppression returns nothing, and only then
   does the ticker hint appear — suggesting you whisper it once so Sotto can
   learn it. With the mic off or the model not ready, the hint shows directly.

The voiced bar itself is measured, not guessed: a segment counts as voiced when
its mean audio level clears max(0.008, three times your microphone's idle noise
floor), with the floor tracked while you are not speaking. And because of the
speculative pass, that bar no longer decides whether a whisper gets heard —
only which rule runs first.

## Self-calibration

Every threshold on the critical path used to be a fixed number derived from
synthetic tests. On a real face, in real light, fixed numbers sit above or
below your actual resting noise — so v0.5 measures instead. What adapts:

- **The segment trigger and release.** While you are idle with a face tracked,
  the engine keeps a rolling estimate of your face's resting motion noise —
  streaming median and 90th-percentile trackers over roughly the last ten
  seconds. The thresholds that decide "mouth started moving" and "mouth
  stopped" are derived from that floor, clamped to sane bounds, and frozen
  while you speak. The sensitivity slider scales the trigger bar. Until the
  trackers have 60 idle samples (about two seconds at 30fps), the fixed v0.4
  values apply — cold-start safety.
- **Timing.** Hysteresis, minimum-segment, and pre-roll windows are measured
  in milliseconds from real frame timestamps, not frame counts, so a camera
  delivering 15, 30, or 60fps behaves the same.
- **The voiced bar.** Whether a segment counts as voiced is judged against
  your microphone's measured idle noise floor, not a constant.

What stays fixed: the DTW matching itself (distance threshold, runner-up
margin), the library caps, and every privacy property. Self-calibration adds
measurements, not storage or network — the floors live in memory and die with
the session.

## Diagnostics

A toggle in the settings drawer. On, a compact HUD renders under the camera
card: live fps, motion energy against the live trigger thresholds, mic level
against the voiced bar, the adaptive floor and its sample count, speech-model
device and state, library counts, and the last five events — segments with
their route and match detail, Whisper results, learn events. Off, it costs
nothing. A "Copy report" button puts a compact JSON report on the clipboard
for pasting into a bug report: your user agent, a timestamp, the engine's
threshold snapshot, speech-model device and state, mic state and noise floor,
library counts, and recent event stats — which include the labels of phrases
that matched. No audio and no video are ever in the report. It goes to your
clipboard and nowhere else; it travels only if you paste it somewhere.

## Auto-learning

Say it once, mouth it later. After Whisper delivers a final transcript for an
utterance, Sotto enrolls that utterance's lip frames as a silent template — the
audio supervises the visual templates, entirely on-device. Guards, all of which
must hold:

- The text, stripped of surrounding punctuation and lowercased, is 1–4 words
  and 2–40 characters, made only of letters, digits, apostrophes, hyphens, and
  spaces — no URLs, no symbols.
- The segment ran 400–3000 ms. Split pieces of a long utterance qualify on
  their own text and their own frames.
- The transcript actually landed in the pad (not paused, not a stale session).
- If a phrase with that label already exists — taught or learned — the frames
  are added as an extra take; at the 8-take cap the oldest take is evicted.
- At most 40 learned phrases, 60 phrases total. Past the cap, the least
  recently used learned phrase is evicted first; a calibrated phrase is never
  evicted to make room for a learned one. If the library is full of taught
  phrases, learning skips silently.
- The first time a label is learned you get one toast; added takes are silent,
  and learn toasts are throttled to at most one per 10 seconds.

Learned phrases show a small "learned" pill in the phrasebook, and manual
teaching stays available for phrases you would rather never say out loud.

v0.5 adds a correction loop for the natural gesture of trying again: when a
silent mouthing misses and you then whisper the same thing within about eight
seconds, the transcript is enrolled twice — once with the voiced take's lip
frames, as always, and once with the frames of the silent miss, provided the
two durations roughly agree. Truly silent articulation differs measurably from
voiced articulation of the same words, so that second take is often the one
that makes silent matching land. One toast either way.

## Voice assist

**What it is.** Open-vocabulary dictation, transcribed by a vendored copy of
OpenAI's Whisper (quantized ONNX encoder and decoder plus tokenizer, under
`vendor/whisper/`) running through transformers.js on ONNX Runtime — WebGPU
when the browser offers it, WASM otherwise. Fully on-device; the model files
ship with the site, so no network is involved at any point.

**The fusion design.** A microphone alone would transcribe everything — your
podcast, your officemate, the television. So the lip tracker gates it. Audio is
captured through an AudioWorklet into a rolling 30-second ring buffer at 16kHz
(raw PCM, in memory, never persisted). When the engine's segmentation sees
*your* mouth start and stop moving, that utterance's time window — padded a
quarter second on each side — is clipped from the buffer and handed to the
model. Long utterances are not cut off: past about six seconds the segment is
split, the finished piece goes off to decode, and the next begins without
dropping a frame, so a long thought arrives in pieces instead of not at all.
(Long silent mouthing splits the same way; calibration takes stay bounded — a
take that runs past the limit is still discarded, on purpose.) Speech that
happens while your mouth is still is never transcribed. The lips decide *when*;
the audio decides *what*. It is sensor fusion, not lipreading.

**Cost.** The speech model (Whisper base.en, 8-bit ONNX) is a one-time download
of roughly 75MB, precached by the service worker like everything else. Measured
on an M-series MacBook: about a second per utterance once warm, on
single-threaded WASM — threads were benchmarked and do not help this workload;
the decoder is sequential. The first utterance after a load takes a few
seconds. While the model decodes, partial text streams into the app live, so
you read words as they are recognized instead of watching a status light.
Startup depends on history. WebGPU is attempted first and used where it works;
a built-in canary transcription (a public-domain 1961 JFK inaugural excerpt)
checks the GPU build actually decodes speech, because some GPU stacks produce
garbage without ever throwing an error. On a machine whose GPU fails that
check, the very first visit can take about 20 seconds before the fallback
settles; Sotto remembers the verdict, so repeat visits skip the detour and the
model is ready in about 4 to 6 seconds. Inference runs in a module worker, so
the interface stays responsive throughout.

**The switch.** Voice assist is one switch in the settings drawer, on by
default. Turning it off releases the microphone immediately — the browser's
mic indicator turns off with it — and Sotto continues camera-only: silent
phrases keep working. The speech model stays loaded, so turning it back on is
quick. If the mic is denied or disconnects mid-session, voice assist drops to
off on its own and the app tells you; the camera flow is unaffected.

## Running it

No build step, no dependencies beyond Python's standard library.

```sh
python3 serve.py
```

Then open <http://localhost:4173>. A server is required (rather than `file://`)
because camera access and WASM want a secure context, and `localhost` counts as
one. Use a browser with camera support; Chrome, Edge, and Safari are all fine.

## Installing as a PWA

Sotto ships a web app manifest and a service worker that precaches everything,
including both models. After the first load it works fully offline — on a deployed
(non-localhost) origin. Under `serve.py` on localhost the service worker
deliberately does not register, so development never fights a stale cache;
install-to-Dock/Home-Screen still works from localhost, but offline does not.

- **Chrome / Edge:** click the install icon at the right end of the address bar,
  or browser menu → "Install Sotto".
- **Safari (macOS):** File → "Add to Dock".
- **Safari (iOS):** Share → "Add to Home Screen".

The installed app opens straight into the input surface (`app.html`).

## Project layout

```
sotto/
├── index.html               landing page
├── app.html                 the instrument: calibration, practice, input
├── css/
│   ├── styles.css           shared tokens + core components (see DESIGN.md)
│   ├── landing.css          landing-only styles
│   └── app.css              app-only styles
├── js/
│   ├── engine.js            SottoEngine: tracking, segmentation, DTW matching
│   ├── audio.js             SottoMic: mic capture into a 16kHz ring buffer
│   ├── audio-worklet.js     AudioWorklet processor feeding that buffer
│   ├── asr.js               SottoASR: main-thread wrapper around the ASR worker
│   ├── asr-worker.js        module worker running Whisper via transformers.js
│   ├── app.js               app UI wiring
│   └── landing.js           landing page behavior
├── assets/                  logo, icons, favicon
├── vendor/
│   ├── inter/               InterVariable (woff2)
│   ├── mediapipe/           Tasks Vision bundle, WASM fileset, face model
│   ├── transformers/        transformers.js ESM + ONNX Runtime backend
│   └── whisper/             Whisper checkpoint: quantized ONNX + tokenizer
├── sw.js                    service worker (offline shell)
├── manifest.webmanifest     PWA manifest
├── serve.py                 static dev server, port 4173
├── DESIGN.md                design system and voice
├── SPEC.md                  v0.1 engine specification
├── SPEC-V2.md               v0.2 Whisper mode specification
├── SPEC-V3.md               v0.3 fast-feel specification
├── SPEC-V4.md               v0.4 unified-mode specification
└── SPEC-V5.md               v0.5 self-calibration specification
```

## Privacy

Short, and stated per mic state.

Always:

- Everything runs on-device. Video frames are processed in memory and discarded.
- No network requests after load. Fonts, models, and WASM are all vendored —
  there are no CDNs, no analytics, no telemetry, no accounts.
- Your phrase library — taught and learned alike — is stored in your browser's
  `localStorage` and goes nowhere unless you export it yourself.

Voice assist on (the default):

- The microphone is live. Audio sits in a 30-second in-memory ring buffer, and
  clipped utterances are transcribed on this device by the local model. Nothing
  is written to disk, stored, or uploaded — there is no server in the pipeline.

Voice assist off (or mic blocked):

- Camera only. The microphone is off — released the instant the switch flips —
  and silent phrases still work.

## Current limitations

Stated plainly, because they are real:

- **Silent coverage is the library.** In silence, Sotto recognizes the phrases
  it has been taught or has learned from your speech — up to 60 phrases (at
  most 40 of them learned) with up to 8 takes each. It cannot read arbitrary
  silent speech; nothing can, yet.
- **Dictation needs sound.** The voiced path needs at least whispered speech —
  some audible airflow through real words. Mouthing in complete silence gives
  the speech model nothing to work with; the lips only gate when it listens.
- **The voiced path inherits Whisper's habits.** Very faint whispers, unusual
  vocabulary, and noisy rooms degrade transcription, and the model can
  hallucinate on near-silence (Sotto filters the classic cases, but not all of
  them).
- **A learned phrase starts from one voiced take.** Saying a phrase aloud and
  mouthing it silently are not identical motions, so a freshly learned phrase
  may need to be heard once or twice more before silent matching lands
  reliably. Extra takes accumulate automatically as you keep talking, and the
  correction loop shortens the path: miss silently, whisper it, and the silent
  take is enrolled too.
- **Visually similar phrases collide.** Many sounds are indistinguishable on the
  lips ("pat" and "bat" are the same movie). Phrases that look alike when mouthed
  will confuse the matcher; pick phrases that move your mouth differently.
- **Templates are personal.** Your library encodes your mouth, your camera, your
  framing. It will not transfer well to another person, and may degrade if you
  change setup drastically.
- **Conditions matter.** Dim lighting, strong side angles, occlusion (hands,
  mugs, enthusiastic beards) all degrade tracking.
- **Short phrases are harder.** A single quick syllable gives the matcher little
  signal. Two- to four-word phrases work best.

## Roadmap

Directions under exploration — not promises:

- More robust features under head-pose variation.
- Smarter segmentation for fast back-to-back phrases.
- Per-phrase threshold adaptation from practice-mode statistics.
- A proper study of vocabulary size vs. accuracy, published with the numbers,
  whatever they turn out to be.

## License

All rights reserved (prototype). This is research-preview code, provided to look
at and learn from; ask before reusing it.
