# Sotto

*sotto voce* — under the voice.

Sotto is a silent-speech input instrument. You mouth words at your camera; it types
them. No audio is ever captured, and nothing leaves your machine.

It is a **research preview**, and it is honest about what that means: Sotto does
**calibrated-vocabulary silent phrase input**, not open-vocabulary dictation. You
teach it a small set of phrases by mouthing each one a few times. Afterwards it
recognizes those phrases — and only those — from lip movement alone. Within that
constraint, it works, and it is quietly satisfying to use.

## How it works

The whole pipeline runs in your browser, on your device:

1. **Camera.** `getUserMedia` video only. The microphone is never requested.
2. **Face tracking.** A vendored copy of MediaPipe Tasks Vision (`FaceLandmarker`,
   WASM) tracks one face per frame, GPU delegate with CPU fallback.
3. **Features.** Each frame is reduced to a 22-dimensional vector: 18
   mouth-relevant blendshape scores (jaw, lips, cheeks) plus 4 geometric
   measurements — lip aperture, lip width, upper- and lower-lip height — each
   normalized by inter-ocular distance so the numbers survive you leaning toward
   the camera.
4. **Segmentation.** Frame-to-frame feature energy, smoothed and run through an
   on/off hysteresis, cuts the stream into utterance segments: mouth starts
   moving, mouth stops moving, that was a phrase.
5. **Matching.** Each segment is resampled to 32 frames and compared against your
   recorded templates with dynamic time warping (Sakoe-Chiba band). A phrase is
   accepted only if the best match is close enough *and* clearly ahead of the
   runner-up. Otherwise Sotto says nothing, which is the correct thing to say.
6. **Output.** The matched phrase is typed into the composer. Your phrase library
   lives in `localStorage` and can be exported and imported as JSON.

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
including the model. After the first load it works fully offline — on a deployed
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
│   ├── app.js               app UI wiring
│   └── landing.js           landing page behavior
├── assets/                  logo, icons, favicon
├── vendor/
│   ├── inter/               InterVariable (woff2)
│   └── mediapipe/           Tasks Vision bundle, WASM fileset, face model
├── sw.js                    service worker (offline shell)
├── manifest.webmanifest     PWA manifest
├── serve.py                 static dev server, port 4173
├── DESIGN.md                design system and voice
└── SPEC.md                  engine specification
```

## Privacy

This part is short because there is nothing to disclose.

- Everything runs on-device. Video frames are processed in memory and discarded.
- No audio is captured; the microphone permission is never requested.
- No network requests after load. Fonts, model, and WASM are all vendored — there
  are no CDNs, no analytics, no telemetry, no accounts.
- Your phrase library is stored in your browser's `localStorage` and goes nowhere
  unless you export it yourself.

## Current limitations

Stated plainly, because they are real:

- **Calibrated vocabulary only.** Sotto recognizes phrases you have taught it —
  up to 60 phrases with up to 8 takes each. It cannot transcribe arbitrary speech.
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
