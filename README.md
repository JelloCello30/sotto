# Sotto

*sotto voce* — under the voice.

Sotto is a quiet-speech input instrument with two modes:

- **Phrase mode** (the default): calibrated silent-phrase input. You mouth
  phrases you have taught it; it types them from lip movement alone. Camera
  only — the microphone is never requested.
- **Whisper mode** (new in v0.2): open-vocabulary dictation. You whisper (or
  speak), and a vendored Whisper model transcribes it on-device, with your lip
  movement deciding when the microphone is worth listening to.

Everything runs in your browser, on your machine. Nothing is uploaded in either
mode.

It is a **research preview**, and it is honest about what that means. Phrase
mode is vocabulary-bound: it recognizes the phrases you taught it, and only
those, but it does so in complete silence. Whisper mode is open-vocabulary, but
it is not lipreading: the audio decides *what* you said; the lips only decide
*when* to listen. Within those constraints, both work, and they are quietly
satisfying to use.

## How it works

The whole pipeline runs in your browser, on your device:

1. **Camera.** `getUserMedia` video only. In Phrase mode the microphone is
   never requested.
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
7. **Whisper branch (Whisper mode only).** The same segmentation timestamps clip
   a padded window from a rolling in-memory microphone buffer; the clipped audio
   is transcribed by a local Whisper model in a worker, and the text lands in
   the composer verbatim. Template matching (step 5) is bypassed in this mode.

## Whisper mode

Phrase mode's constraint is its vocabulary. Whisper mode removes it, at an
honestly stated price: it needs to hear you, at least a little.

**What it is.** Open-vocabulary dictation, transcribed by a vendored copy of
OpenAI's Whisper (quantized ONNX encoder and decoder plus tokenizer, under
`vendor/whisper/`) running through transformers.js on ONNX Runtime — WebGPU
when the browser offers it, WASM otherwise. Fully on-device; the model files
ship with the site, so no network is involved at any point.

**The fusion design.** A microphone alone would transcribe everything — your
podcast, your officemate, the television. So the lip tracker gates it. Audio is
captured through an AudioWorklet into a rolling 30-second ring buffer at 16kHz
(raw PCM, in memory, never persisted). When the v0.1 engine's segmentation sees
*your* mouth start and stop moving, that utterance's time window — padded a
quarter second on each side — is clipped from the buffer and handed to the
model. Long utterances are not cut off: past about six seconds the segment is
split, the finished piece goes off to decode, and the next begins without
dropping a frame, so a long thought arrives in pieces instead of not at all.
(Phrase-mode calibration takes stay bounded — a take that runs past the limit
is still discarded, on purpose.) Speech that happens while your mouth is still
is never transcribed. The lips decide *when*; the audio decides *what*. It is
sensor fusion, not lipreading.

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

**Switching.** The mode control lives in the app's camera column, and the
choice persists. Leaving Whisper mode releases the microphone immediately — the
browser's mic indicator turns off with it. The speech model stays loaded so
switching back is instant.

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
└── SPEC-V3.md               v0.3 fast-feel specification
```

## Privacy

Short, and stated per mode.

Both modes:

- Everything runs on-device. Video frames are processed in memory and discarded.
- No network requests after load. Fonts, models, and WASM are all vendored —
  there are no CDNs, no analytics, no telemetry, no accounts.
- Your phrase library is stored in your browser's `localStorage` and goes
  nowhere unless you export it yourself.

Phrase mode:

- No audio, ever. The microphone permission is never requested, so the browser
  has nothing to hand over.

Whisper mode:

- The microphone is live while the mode is on. Audio sits in a 30-second
  in-memory ring buffer, and clipped utterances are transcribed on this device
  by the local model. Nothing is written to disk, stored, or uploaded — there
  is no server in the pipeline. Switching back to Phrase mode stops the
  microphone immediately.

## Current limitations

Stated plainly, because they are real:

- **Phrase mode is calibrated vocabulary only.** It recognizes phrases you have
  taught it — up to 60 phrases with up to 8 takes each. It cannot transcribe
  arbitrary speech; that is what Whisper mode is for, and Whisper mode needs
  sound.
- **Whisper mode is not lipreading.** It needs at least whispered speech — some
  audible airflow through real words. Mouthing in complete silence gives the
  model nothing to work with. The lips only gate when it listens.
- **Whisper mode inherits Whisper's habits.** Very faint whispers, unusual
  vocabulary, and noisy rooms degrade transcription, and the model can
  hallucinate on near-silence (Sotto filters the classic cases, but not all of
  them).
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
