/* Sotto — app wiring. Engine contract: see SPEC.md (js/engine.js). */

import { SottoEngine, CONSTANTS } from './engine.js';

/* ---------------------------------------------------------------- helpers */

const $ = (id) => document.getElementById(id);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const fmtConf = (c) => clamp01(Number(c) || 0).toFixed(2);
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/* ---------------------------------------------------------------- elements */

const video = $('cam');
const camPill = $('cam-pill');
const camPillText = $('cam-pill-text');
const lipwave = $('lipwave');
const fpsEl = $('fps');
const camNote = $('cam-note');
const camEmpty = $('cam-empty');
const camEmptyTitle = $('cam-empty-title');
const camEmptyBody = $('cam-empty-body');
const camEmptyBtn = $('cam-empty-btn');
const toastRegion = $('toast-region');

const statusPill = $('status');
const statusText = $('status-text');

const camToggle = $('cam-toggle');
const sensitivity = $('sensitivity');
const sensOut = $('sens-out');
const pauseToggle = $('pause-toggle');

const pad = $('pad');
const copyBtn = $('copy-btn');
const clearBtn = $('clear-btn');
const autocopyToggle = $('autocopy-toggle');

const practiceToggle = $('practice-toggle');
const practicePanel = $('practice-panel');
const practiceList = $('practice-list');
const practiceEmpty = $('practice-empty');

const phraseList = $('phrase-list');
const bookEmpty = $('book-empty');
const addPhraseBtn = $('add-phrase-btn');
const exportBtn = $('export-btn');
const importBtn = $('import-btn');
const importFile = $('import-file');

const appMain = $('main');
const modePhrases = $('mode-phrases');
const modeWhisper = $('mode-whisper');
const camPrivacy = $('cam-privacy');
const micMeter = $('mic-meter');
const micLevel = $('mic-level');
const asrNote = $('asr-note');
const asrNoteText = $('asr-note-text');
const asrProgress = $('asr-progress');
const asrProgressFill = $('asr-progress-fill');
const micEmpty = $('mic-empty');
const micEmptyTitle = $('mic-empty-title');
const micEmptyBody = $('mic-empty-body');
const micEmptyRetry = $('mic-empty-retry');
const micEmptyDismiss = $('mic-empty-dismiss');
const asrBlock = $('asr-block');
const asrInfo = $('asr-info');
const pauseLabel = $('pause-label');
const autocopyLabel = $('autocopy-label');
const appFoot = $('app-foot');

const settingsBtn = $('settings-btn');
const settingsDrawer = $('settings-drawer');
const settingsClose = $('settings-close');
const selftestBtn = $('selftest-btn');
const selftestOut = $('selftest-out');
const selftestMsg = $('selftest-msg');
const starterBtn = $('starter-btn');

const calDialog = $('cal-dialog');
const calSteps = {
  offer: $('cal-step-offer'),
  camera: $('cal-step-camera'),
  label: $('cal-step-label'),
  record: $('cal-step-record'),
  done: $('cal-step-done'),
};
const calCameraBody = $('cal-camera-body');
const calLabelInput = $('cal-label-input');
const calLabelError = $('cal-label-error');
const calProgress = $('cal-progress');
const calWord = $('cal-word');
const calLive = $('cal-live');
const calDots = $('cal-dots');
const calTakeText = $('cal-take-text');
const calRedo = $('cal-redo');
const calSkip = $('cal-skip');
const calDoneTitle = $('cal-done-title');
const calDoneMsg = $('cal-done-msg');
const calNextBtn = $('cal-next-btn');

/* ---------------------------------------------------------------- app state */

const STARTER = ['hello', 'yes', 'no', 'thank you', 'on my way', 'send it'];
const STARTER_FLAG = 'sotto.starter.decided.v1';
const TAKES_TARGET = 3;

let running = false;        // camera loop active
let engineState = 'off';    // last engine state string, or 'off'
let simulated = null;       // { timer } while a self-test injection is in flight
let starterOffered = false; // offered once per page load at most
let autoCopyWarned = false;

/* ------- whisper mode (v0.2) ------- */

const MODE_KEY = 'sotto.mode.v1';
let mode = 'phrases';       // 'phrases' | 'whisper'

const whisper = {
  session: 0,       // bumped on every leave; stale async completions check it and bail
  entering: false,  // enter lifecycle in flight (radios disabled meanwhile)
  deferred: false,  // restored from storage without a gesture; mic + model wait for one
  mic: null,        // SottoMic instance, created lazily on first entry
  asr: null,        // SottoASR instance, created lazily on first entry
  modPromise: null, // Promise for the dynamic imports of audio.js + asr.js
  ready: false,     // ASR has reported ready at least once (it stays loaded)
  queue: [],        // pending audio slices (Float32Array); cap 2, drop oldest
  busy: false,      // one transcribe call in flight
  droppedToast: false, // "transcriber is behind" toast shown at most once
  device: null,     // 'gpu' | 'cpu' once known
  model: null,      // model name once known
};

const cal = {
  open: false,
  mode: null,        // 'single' | 'starter' | 'rerecord'
  step: null,
  intent: null,      // pending intent while the camera step is up
  label: '',
  queue: [],
  queueIndex: 0,
  takesDone: 0,
  oldCount: 0,       // templates to trim after a re-record
  awaiting: false,   // beginRecording issued, waiting on a segment
  timer: null,
};
let lastFocus = null;

/* ---------------------------------------------------------------- lip-wave */

const RING_N = 120;
const ring = new Float32Array(RING_N);
let ringHead = 0; // next write position; oldest sample lives here too
const ctx2d = lipwave.getContext('2d');
let dpr = window.devicePixelRatio || 1;

let colors = { accent: '', ink: '' };
function refreshColors() {
  const cs = getComputedStyle(document.documentElement);
  colors = {
    accent: cs.getPropertyValue('--accent').trim(),
    ink: cs.getPropertyValue('--ink').trim(),
  };
}
refreshColors();
const darkMq = window.matchMedia('(prefers-color-scheme: dark)');
darkMq.addEventListener('change', () => { refreshColors(); drawWave(); });

function sizeWave() {
  dpr = window.devicePixelRatio || 1;
  const rect = lipwave.getBoundingClientRect();
  if (rect.width === 0) return;
  lipwave.width = Math.round(rect.width * dpr);
  lipwave.height = Math.round(rect.height * dpr);
}
new ResizeObserver(() => { sizeWave(); drawWave(); }).observe(lipwave);

function pushSample(v) {
  ring[ringHead] = clamp01(v);
  ringHead = (ringHead + 1) % RING_N;
}

function drawWave() {
  const w = lipwave.width;
  const h = lipwave.height;
  if (!w || !h) return;
  ctx2d.clearRect(0, 0, w, h);
  const speaking = engineState === 'speaking';
  ctx2d.fillStyle = speaking ? colors.accent : colors.ink;
  const slot = w / RING_N;
  const barW = Math.max(1, slot - 2 * dpr);
  const maxH = h - 10 * dpr;
  for (let i = 0; i < RING_N; i++) {
    const v = ring[(ringHead + i) % RING_N]; // oldest -> newest, left -> right
    const bh = Math.max(2 * dpr, v * maxH);
    const x = i * slot + (slot - barW) / 2;
    const y = (h - bh) / 2;
    ctx2d.globalAlpha = 0.25 + 0.65 * (i / (RING_N - 1));
    if (ctx2d.roundRect) {
      ctx2d.beginPath();
      ctx2d.roundRect(x, y, barW, bh, barW / 2);
      ctx2d.fill();
    } else {
      ctx2d.fillRect(x, y, barW, bh);
    }
  }
  ctx2d.globalAlpha = 1;
}

function clearWave() {
  ring.fill(0);
  ringHead = 0;
  drawWave();
}

/* ---------------------------------------------------------------- status pills */

const NAV_STATUS = {
  off: ['camera off', ''],
  loading: ['loading', 'is-warn'],
  ready: ['watching', 'is-live'],
  idle: ['watching', 'is-live'],
  speaking: ['speaking…', 'is-live'],
  'no-camera': ['camera blocked', 'is-bad'],
  'no-face': ['face lost', 'is-warn'],
  error: ['error', 'is-bad'],
};

const CAM_PILL_TEXT = {
  off: 'camera off',
  loading: 'loading model',
  ready: 'watching',
  idle: 'watching',
  speaking: 'speaking…',
  'no-face': 'watching',
  'no-camera': 'camera off',
  error: 'camera off',
};

function renderStatus() {
  const key = engineState;
  let cls;
  let text;
  const paused = running && pauseToggle.checked && !['no-camera', 'error', 'loading'].includes(key);
  // Whisper mode: watching -> speaking… -> transcribing… -> back. 'speaking…'
  // wins while a new utterance is in progress; 'transcribing…' covers the gap
  // while the ASR queue drains.
  const transcribing = mode === 'whisper' && !paused
    && (whisper.busy || whisper.queue.length > 0)
    && ['ready', 'idle', 'no-face'].includes(key);
  if (paused) {
    text = 'paused';
    cls = 'is-warn';
  } else if (transcribing) {
    text = 'transcribing…';
    cls = 'is-live';
  } else {
    [text, cls] = NAV_STATUS[key] || NAV_STATUS.off;
  }
  statusPill.className = `pill ${cls}`.trim();
  statusText.textContent = text;

  let camText = CAM_PILL_TEXT[key] || 'camera off';
  let camCls = 'pill cam-pill' +
    (key === 'speaking' ? ' is-live' : ['ready', 'idle', 'no-face'].includes(key) ? ' is-good' : '');
  if (transcribing) {
    camText = 'transcribing…';
    camCls = 'pill cam-pill is-live';
  }
  camPillText.textContent = camText;
  camPill.className = camCls;
}

/* ---------------------------------------------------------------- camera empty states */

const EMPTY_COPY = {
  off: {
    title: 'Camera is off',
    body: 'Sotto reads lip movement through your camera. Frames are processed on this device and discarded. No audio, no uploads.',
    btn: 'Start camera',
  },
  'no-camera': {
    title: 'Camera blocked',
    body: 'The browser refused camera access. Click the camera icon in the address bar (or check site permissions in browser settings), allow it, then try again.',
    btn: 'Try again',
  },
  error: {
    title: 'Something broke',
    body: 'The face tracker failed to start.',
    btn: 'Try again',
  },
};

// In Whisper mode the "no audio" line above would be false — the mic is on.
const EMPTY_COPY_WHISPER_OFF = {
  title: 'Camera is off',
  body: 'Whisper mode still needs the camera: lip movement decides when the microphone is listened to. Frames are processed on this device and discarded.',
  btn: 'Start camera',
};

// A restored Whisper preference waits for a gesture (see whisper.deferred);
// the button below is that gesture, so say what it will actually do.
const EMPTY_COPY_WHISPER_DEFERRED = {
  title: 'Camera is off',
  body: 'Whisper mode is selected, but nothing is listening yet. Starting the camera also turns on the microphone and loads the speech model — all on this device, nothing uploaded.',
  btn: 'Start camera',
};

let lastEmptyKind = 'off'; // so a mode switch can refresh the visible copy

function showCamEmpty(kind, detail) {
  lastEmptyKind = kind;
  const copy = (kind === 'off' && mode === 'whisper')
    ? (whisper.deferred ? EMPTY_COPY_WHISPER_DEFERRED : EMPTY_COPY_WHISPER_OFF)
    : (EMPTY_COPY[kind] || EMPTY_COPY.off);
  camEmptyTitle.textContent = copy.title;
  camEmptyBody.textContent = kind === 'error' && detail ? `${copy.body} Detail: ${detail}` : copy.body;
  camEmptyBtn.textContent = copy.btn;
  camEmpty.hidden = false;
  camNote.hidden = true;
  fpsEl.hidden = true;
}

function hideCamEmpty() {
  camEmpty.hidden = true;
}

/* ---------------------------------------------------------------- toasts */

function toast(label, conf, kind) {
  const chip = document.createElement('span');
  chip.className = 'toast-chip' + (kind ? ` is-${kind}` : '');
  const name = document.createElement('span');
  name.textContent = label;
  chip.appendChild(name);
  if (conf != null) {
    const c = document.createElement('span');
    c.className = 'conf';
    c.textContent = `· ${conf}`;
    chip.appendChild(c);
  }
  while (toastRegion.children.length >= 3) toastRegion.firstChild.remove();
  toastRegion.appendChild(chip);
  setTimeout(() => chip.remove(), 2600);
}

/* ---------------------------------------------------------------- the pad */

function growPad() {
  pad.style.height = 'auto';
  pad.style.height = `${pad.scrollHeight + 2}px`;
}
pad.addEventListener('input', growPad);

function appendToPad(text) {
  const v = pad.value;
  const base = v.replace(/[ \t]+$/, ''); // trim trailing spaces, keep newlines
  if (base === '') {
    pad.value = cap(text);
  } else if (base.endsWith('\n')) {
    pad.value = base + cap(text); // line start: capitalize, no extra space
  } else {
    const last = base.slice(-1);
    pad.value = base + ' ' + (/[.!?]/.test(last) ? cap(text) : text);
  }
  growPad();
  pad.scrollTop = pad.scrollHeight;
  if (autocopyToggle.checked) copyPad(true);
}

function copyPad(silent) {
  if (!pad.value) {
    if (!silent) toast('nothing to copy yet', null, 'miss');
    return;
  }
  if (!navigator.clipboard) {
    if (!silent || !autoCopyWarned) toast('clipboard unavailable in this browser', null, 'bad');
    if (silent) autoCopyWarned = true;
    return;
  }
  navigator.clipboard.writeText(pad.value).then(() => {
    if (!silent) {
      const prev = copyBtn.textContent;
      copyBtn.textContent = 'Copied';
      copyBtn.disabled = true;
      setTimeout(() => { copyBtn.textContent = prev; copyBtn.disabled = false; }, 1200);
    }
  }).catch(() => {
    if (silent && autoCopyWarned) return;
    if (silent) autoCopyWarned = true;
    toast('clipboard blocked by browser', null, 'bad');
  });
}

copyBtn.addEventListener('click', () => copyPad(false));
clearBtn.addEventListener('click', () => {
  if (!pad.value) return;
  pad.value = '';
  growPad();
  toast('pad cleared', null, 'miss');
  pad.focus();
});

/* ---------------------------------------------------------------- practice mode */

practiceToggle.addEventListener('change', () => {
  practicePanel.hidden = !practiceToggle.checked;
});

function renderPractice(stats, accepted) {
  practiceList.textContent = '';
  const top = (stats || []).slice(0, 5);
  practiceEmpty.hidden = top.length > 0;
  if (!top.length) {
    practiceEmpty.textContent = 'No ranking to show — the book may be empty.';
    return;
  }
  top.forEach((s, i) => {
    const li = document.createElement('li');
    li.className = 'practice-item' + (i === 0 ? ' is-top' : '');
    const name = document.createElement('span');
    name.className = 'plabel';
    name.textContent = s.label;
    const conf = document.createElement('span');
    conf.className = 'pconf';
    conf.textContent = fmtConf(s.confidence);
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(clamp01(s.confidence) * 100)}%`;
    bar.appendChild(fill);
    li.append(name, conf, bar);
    if (accepted && accepted.label === s.label) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = 'accepted';
      li.appendChild(tag);
    }
    practiceList.appendChild(li);
  });
}

/* ---------------------------------------------------------------- engine */

const engine = new SottoEngine({
  onState(state, detail) {
    const prev = engineState;
    engineState = state;
    trackLongUtterance(prev, state);
    switch (state) {
      case 'loading':
        hideCamEmpty();
        camNote.hidden = true;
        break;
      case 'ready':
      case 'idle':
        hideCamEmpty();
        camNote.hidden = true;
        fpsEl.hidden = false;
        maybeOfferStarter();
        break;
      case 'speaking':
        camNote.hidden = true;
        break;
      case 'no-face':
        camNote.hidden = false;
        break;
      case 'no-camera':
        running = false;
        camToggle.textContent = 'Start camera';
        showCamEmpty('no-camera');
        clearWave();
        break;
      case 'error':
        running = false;
        camToggle.textContent = 'Start camera';
        showCamEmpty('error', detail);
        clearWave();
        break;
    }
    renderStatus();
    updateCalLive();
  },

  onFrame(frame) {
    pushSample(frame.mouthOpen);
    drawWave();
    if (whisper.mic && whisper.mic.running) updateMicMeter();
    else checkMicDisconnect();
    const now = performance.now();
    if (now - lastFpsAt > 500) {
      lastFpsAt = now;
      fpsEl.textContent = `${Math.round(frame.fps)} fps`;
    }
  },

  onSegment(seg) {
    if (cal.open && cal.awaiting && seg.recordingLabel === cal.label) {
      cal.awaiting = false;
      cal.takesDone += 1;
      refreshPhrasebook();
      updateTakesUI();
      if (cal.takesDone < TAKES_TARGET) {
        setCalLive('Got it.');
        scheduleTake(900);
      } else {
        finishPhrase();
      }
      return;
    }
    // Fusion: in Whisper mode a lip-motion segment decides WHEN to listen; the
    // audio slice decides WHAT was said. Self-test injections stay out of it.
    if (mode === 'whisper' && !seg.recordingLabel && !simulated) {
      handleWhisperSegment(seg);
    }
  },

  onMatch(m, seg) {
    if (seg && seg.recordingLabel) return; // calibration takes are handled above

    if (simulated) {
      clearTimeout(simulated.timer);
      simulated = null;
      selftestBtn.disabled = false;
      reportSelfTest(m, seg);
      return;
    }

    if (mode === 'whisper') return; // phrase matching is bypassed in Whisper mode

    if (practiceToggle.checked) {
      let stats = [];
      try { stats = engine.matchStats(seg); } catch { stats = []; }
      renderPractice(stats, m);
    }

    if (m) {
      appendToPad(m.label);
      toast(m.label, fmtConf(m.confidence));
    } else {
      toast('no match', null, 'miss');
    }
  },
});

let lastFpsAt = 0;

window.__sotto = { engine, whisper };

/* ---------------------------------------------------------------- camera control */

async function startCamera() {
  if (running) return;
  // Starting the camera while a restored Whisper selection is waiting counts
  // as the gesture that arms it: mic and model start alongside the camera.
  runDeferredWhisper();
  camToggle.disabled = true;
  camEmptyBtn.disabled = true;
  engineState = 'loading';
  renderStatus();
  camPillText.textContent = 'loading model';
  hideCamEmpty();
  try {
    await engine.start(video);
    running = true;
    camToggle.textContent = 'Stop camera';
  } catch (err) {
    running = false;
    // Only camera errors get the "check your camera permission" copy — a
    // model/wasm load failure is not fixable from the browser's camera
    // settings and must keep its own explanation.
    const code = err && err.code;
    if (code === 'camera-denied' || code === 'camera-none') {
      engineState = 'no-camera';
      showCamEmpty('no-camera', err && err.message);
    } else {
      engineState = 'error';
      showCamEmpty('error', err && err.message);
    }
    renderStatus();
  } finally {
    camToggle.disabled = false;
    camEmptyBtn.disabled = false;
  }
}

function stopCamera() {
  engine.stop();
  running = false;
  engineState = 'off';
  camToggle.textContent = 'Start camera';
  showCamEmpty('off');
  clearWave();
  fpsEl.hidden = true;
  renderStatus();
  if (cal.open && (cal.step === 'record' || cal.step === 'label')) {
    abortTakesForCameraLoss();
  }
}

camToggle.addEventListener('click', () => (running ? stopCamera() : startCamera()));
camEmptyBtn.addEventListener('click', () => startCamera());

sensitivity.addEventListener('input', () => {
  const v = Number(sensitivity.value);
  engine.setSensitivity(v);
  sensOut.textContent = v.toFixed(2);
});

pauseToggle.addEventListener('change', () => {
  engine.setPaused(pauseToggle.checked);
  if (pauseToggle.checked && mode === 'whisper') {
    // Pause means pause: queued audio does not outlive it, and an in-flight
    // result is dropped when it lands (see pumpTranscribe).
    whisper.queue.length = 0;
  }
  renderStatus();
});

/* ---------------------------------------------------------------- whisper mode */

const ASR_LOADING_TEXT = 'Loading speech model — ~75MB, one time.';

const PRIVACY_PHRASES =
  'Phrase mode: camera only. No audio is captured; frames are processed on this device and discarded.';
const PRIVACY_WHISPER =
  'Whisper mode: microphone on, processed on this device, nothing uploaded. Lip movement decides when it listens; the audio decides what was said.';
// While a restored selection waits for a gesture, "microphone on" would be a lie.
const PRIVACY_WHISPER_DEFERRED =
  'Whisper mode: selected, microphone still off. Starting the camera turns it on — processed on this device, nothing uploaded.';

const FOOT_PHRASES =
  'No audio is captured. The camera feed, the model, and your phrasebook stay on this device.';
const FOOT_WHISPER =
  'Whisper mode listens through the microphone and transcribes on this device. Audio, frames, and text all stay here — nothing is uploaded.';

const MIC_EMPTY_COPY = {
  denied: {
    title: 'Microphone blocked',
    body: 'Whisper mode needs the microphone; the browser refused it. Allow it in the address bar (or site permissions), then try again. You are back in Phrases — camera only, no audio.',
    retry: 'Try Whisper again',
  },
  none: {
    title: 'No microphone found',
    body: 'Whisper mode needs a microphone and the browser could not find one. Connect one and try again. You are back in Phrases mode.',
    retry: 'Try Whisper again',
  },
  asr: {
    title: 'Speech model failed to load',
    body: 'Whisper mode could not start its on-device speech model. You are back in Phrases mode.',
    retry: 'Try again',
  },
  lost: {
    title: 'Microphone disconnected',
    body: 'The microphone went away mid-session — unplugged, or the system withdrew it. Nothing has been transcribed since. You are back in Phrases — camera only, no audio.',
    retry: 'Try Whisper again',
  },
};

function setMode(next) {
  next = next === 'whisper' ? 'whisper' : 'phrases';
  if (next === mode) {
    syncModeUI();
    return;
  }
  const prev = mode;
  mode = next;
  localStorage.setItem(MODE_KEY, mode);
  syncModeUI();
  if (mode === 'whisper') {
    enterWhisper();
  } else if (prev === 'whisper') {
    leaveWhisper();
    if (running && ['ready', 'idle'].includes(engineState)) maybeOfferStarter();
  }
}

function syncModeUI() {
  const whisperOn = mode === 'whisper';
  modePhrases.checked = !whisperOn;
  modeWhisper.checked = whisperOn;
  appMain.classList.toggle('mode-whisper', whisperOn);
  camPrivacy.textContent = whisperOn
    ? (whisper.deferred ? PRIVACY_WHISPER_DEFERRED : PRIVACY_WHISPER)
    : PRIVACY_PHRASES;
  appFoot.textContent = whisperOn ? FOOT_WHISPER : FOOT_PHRASES;
  pauseLabel.textContent = whisperOn ? 'Pause transcribing' : 'Pause matching';
  autocopyLabel.textContent = whisperOn ? 'Auto-copy each transcription' : 'Auto-copy each match';
  pad.placeholder = whisperOn ? 'Transcribed speech lands here.' : 'Matched phrases land here.';
  pad.setAttribute('aria-label', whisperOn
    ? 'The pad. Transcribed speech is appended here.'
    : 'The pad. Matched phrases are appended here.');
  if (!camEmpty.hidden && lastEmptyKind === 'off') showCamEmpty('off');
  renderStatus();
}

function setModeControlsDisabled(disabled) {
  modePhrases.disabled = disabled;
  modeWhisper.disabled = disabled;
}

async function enterWhisper() {
  const session = ++whisper.session;
  whisper.entering = true;
  micLostShown = false;
  // Disabling the radios below blurs a focused one to <body>; remember it so
  // the finally can hand focus back (a11y: keyboard mode switching).
  const focusedRadio =
    document.activeElement === modePhrases || document.activeElement === modeWhisper
      ? document.activeElement
      : null;
  setModeControlsDisabled(true);
  hideMicEmpty();
  try {
    // Lazy: Phrase-mode users never pay for the audio/ASR modules.
    if (!whisper.modPromise) {
      whisper.modPromise = Promise.all([import('./audio.js'), import('./asr.js')]);
    }
    let mods;
    try {
      mods = await whisper.modPromise;
    } catch (err) {
      whisper.modPromise = null; // a retry should re-attempt the import
      throw err;
    }
    if (session !== whisper.session) return;
    const [audioMod, asrMod] = mods;
    // stop() closes the AudioContext, so a stopped mic gets a fresh instance
    // rather than assuming it can restart.
    if (!whisper.mic || !whisper.mic.running) whisper.mic = new audioMod.SottoMic();
    if (!whisper.asr) whisper.asr = new asrMod.SottoASR({ onStatus: onAsrStatus });
    if (!whisper.mic.running) await whisper.mic.start();
    if (session !== whisper.session) {
      try { whisper.mic.stop(); } catch { /* already stopped */ }
      return;
    }
    startMicMeter();
    if (!whisper.ready) showAsrNote(ASR_LOADING_TEXT, null);
    await whisper.asr.load();
    if (session !== whisper.session) return;
    whisper.ready = true;
    hideAsrNote();
    renderAsrInfo();
    renderStatus();
  } catch (err) {
    if (session !== whisper.session) return;
    hideAsrNote();
    setMode('phrases'); // stops the mic and restores the Phrases UI cleanly
    const code = err && err.code;
    if (code === 'mic-denied') showMicEmpty('denied');
    else if (code === 'mic-none') showMicEmpty('none');
    else showMicEmpty('asr', err && err.message ? err.message : String(err || 'unknown error'));
  } finally {
    whisper.entering = false;
    setModeControlsDisabled(false);
    // If the blur landed on <body> and nothing (like the mic empty state)
    // claimed focus since, restore it to whichever radio is now checked —
    // Phrases after a failed entry, Whisper after a successful one.
    if (focusedRadio && document.activeElement === document.body) {
      (modeWhisper.checked ? modeWhisper : modePhrases).focus();
    }
  }
}

function leaveWhisper() {
  whisper.session += 1; // any in-flight transcription result is discarded
  whisper.deferred = false; // a deliberate move to Phrases cancels a deferred entry
  whisper.queue.length = 0;
  whisper.busy = false;
  cancelLongUtterToast();
  if (whisper.mic) {
    try { whisper.mic.stop(); } catch { /* already stopped */ }
  }
  stopMicMeter();
  hideAsrNote();
  renderStatus();
  // The ASR stays loaded — re-entering Whisper mode is instant.
}

/**
 * Run the entry lifecycle for a Whisper selection that was restored from
 * storage without a gesture (whisper.deferred). Called only from real user
 * gestures — the Start camera buttons, or a click on the Whisper radio — so
 * the mic permission prompt and the model load never happen on page load.
 * @returns {boolean} true if a deferred entry was started
 */
function runDeferredWhisper() {
  if (!whisper.deferred || mode !== 'whisper') return false;
  whisper.deferred = false;
  syncModeUI(); // drop the "still off" privacy line and empty-state copy
  enterWhisper();
  return true;
}

/* ------- fusion: lip segment -> audio slice -> transcription ------- */

function handleWhisperSegment(seg) {
  cancelLongUtterToast(); // a segment arrived, so the utterance was not discarded
  if (!whisper.mic || !whisper.mic.running || !whisper.asr || !whisper.ready) return;
  let audio;
  try {
    audio = whisper.mic.slice(seg.t0 - 250, seg.t1 + 250);
  } catch {
    return;
  }
  if (!audio || !audio.length) return;
  whisper.queue.push(audio);
  while (whisper.queue.length > 2) {
    whisper.queue.shift();
    if (!whisper.droppedToast) {
      whisper.droppedToast = true;
      toast('transcriber is behind — dropped a segment', null, 'miss');
    }
  }
  pumpTranscribe();
}

function pumpTranscribe() {
  if (whisper.busy || !whisper.queue.length) {
    renderStatus();
    return;
  }
  const session = whisper.session;
  const audio = whisper.queue.shift();
  whisper.busy = true;
  renderStatus();
  whisper.asr.transcribe(audio).then((text) => {
    if (session !== whisper.session) return;
    // Pause honesty: while the pill says "paused", a result that was already
    // in flight must not reach the pad, a toast, or the clipboard.
    if (pauseToggle.checked) return;
    const clean = (text || '').trim();
    if (clean) {
      appendToPad(clean); // verbatim beyond appendToPad's usual sentence-start casing
      toastTranscript(clean);
    }
  }).catch(() => {
    if (session !== whisper.session) return;
    toast('transcription failed', null, 'bad');
  }).then(() => {
    if (session !== whisper.session) return;
    whisper.busy = false;
    renderStatus();
    pumpTranscribe();
  });
}

function toastTranscript(text) {
  const words = text.split(/\s+/);
  const head = words.slice(0, 4).join(' ');
  toast(words.length > 4 ? `${head}…` : head, null);
}

/* ------- long-utterance feedback ------- */

// The engine discards any utterance longer than MAX_SEG_MS without emitting a
// segment, which in Whisper mode reads as words silently vanishing. Time the
// 'speaking' state app-side and say why nothing landed. The threshold sits
// below MAX_SEG_MS because the engine's segment clock starts on pre-roll
// frames captured before the state flips to 'speaking'.
const LONG_UTTER_MS = CONSTANTS.MAX_SEG_MS - 400;
let speakingSince = 0;     // performance.now() when 'speaking' began
let longUtterTimer = null; // pending "ran long" toast; an arriving segment cancels it

/**
 * Called on every engine state transition (from onState).
 * @param {string} prev previous engine state
 * @param {string} state new engine state
 */
function trackLongUtterance(prev, state) {
  if (state === 'speaking' && prev !== 'speaking') {
    speakingSince = performance.now();
    return;
  }
  if (prev !== 'speaking' || state === 'speaking') return;
  if (mode !== 'whisper' || simulated) return;
  if (performance.now() - speakingSince <= LONG_UTTER_MS) return;
  // Give a real segment 500 ms to arrive before concluding it was discarded.
  clearTimeout(longUtterTimer);
  longUtterTimer = setTimeout(() => {
    longUtterTimer = null;
    toast('that ran past six seconds — pause briefly between sentences', null, 'miss');
  }, 500);
}

function cancelLongUtterToast() {
  if (longUtterTimer) {
    clearTimeout(longUtterTimer);
    longUtterTimer = null;
  }
}

/* ------- ASR status + settings info ------- */

function onAsrStatus(status, detail) {
  // Defensive normalization: accept ('loading-model', {pct}) or ({state, pct}).
  let state = status;
  let info = detail;
  if (status && typeof status === 'object') {
    state = status.status || status.state;
    info = status;
  }
  let pct = null;
  if (typeof info === 'number') pct = info;
  else if (info && typeof info === 'object') {
    if (typeof info.pct === 'number') pct = info.pct;
    else if (typeof info.progress === 'number') pct = info.progress;
  }
  if (pct != null && pct <= 1) pct *= 100;

  switch (state) {
    case 'loading-model':
      if (mode === 'whisper') showAsrNote(ASR_LOADING_TEXT, pct);
      break;
    case 'warming':
      if (mode === 'whisper') showAsrNote('Warming up the speech model…', null);
      break;
    case 'ready':
      whisper.ready = true;
      captureAsrMeta(info);
      hideAsrNote();
      renderAsrInfo();
      break;
    case 'error':
      hideAsrNote(); // load() rejects too; the catch in enterWhisper reports it
      break;
  }
}

function captureAsrMeta(info) {
  const a = whisper.asr || {};
  const dev = (info && typeof info === 'object' && (info.device || info.backend))
    || a.device || a.backend || null;
  const model = (info && typeof info === 'object' && (info.model || info.modelId))
    || a.model || a.modelId || a.modelName || null;
  if (dev) whisper.device = /gpu/i.test(String(dev)) ? 'gpu' : 'cpu';
  if (model) whisper.model = String(model);
}

function renderAsrInfo() {
  if (!whisper.ready) return;
  const bits = [];
  bits.push(whisper.model ? `Speech model ${whisper.model}` : 'Speech model loaded');
  if (whisper.device) bits.push(`running on ${whisper.device}`);
  asrInfo.textContent = `${bits.join(' — ')}. Transcription happens on this device; audio never leaves it.`;
  asrBlock.hidden = false;
}

/* ------- loading note on the camera card ------- */

function showAsrNote(text, pct) {
  asrNoteText.textContent = pct != null ? `${text} (${Math.round(pct)}%)` : text;
  asrProgress.hidden = pct == null;
  asrProgressFill.style.width = pct != null ? `${Math.min(100, Math.max(0, pct))}%` : '0%';
  asrNote.hidden = false;
}

function hideAsrNote() {
  asrNote.hidden = true;
}

/* ------- mic-blocked empty state (mirrors the camera one) ------- */

function showMicEmpty(kind, detail) {
  const copy = MIC_EMPTY_COPY[kind] || MIC_EMPTY_COPY.asr;
  micEmptyTitle.textContent = copy.title;
  micEmptyBody.textContent = kind === 'asr' && detail ? `${copy.body} Detail: ${detail}` : copy.body;
  micEmptyRetry.textContent = copy.retry;
  micEmpty.hidden = false;
  micEmptyRetry.focus();
}

function hideMicEmpty() {
  micEmpty.hidden = true;
}

micEmptyRetry.addEventListener('click', () => {
  hideMicEmpty();
  setMode('whisper');
});
micEmptyDismiss.addEventListener('click', () => hideMicEmpty());

/* ------- mic disconnect watchdog ------- */

// One-shot per Whisper entry: set when the disconnect state has been shown,
// reset by enterWhisper, so the polling paths cannot spam it.
let micLostShown = false;

/**
 * The mic track can end under us — headset unplugged, OS-level revocation —
 * and SottoMic stops itself silently when it does. Without this check the
 * pill would keep saying "watching" while every segment gets dropped by
 * handleWhisperSegment's running guard. Polled from onFrame and micMeterLoop.
 */
function checkMicDisconnect() {
  if (micLostShown) return;
  if (mode !== 'whisper' || whisper.entering) return;
  if (!whisper.mic || whisper.mic.running) return;
  micLostShown = true;
  setMode('phrases'); // stops routing and stops the pill claiming otherwise
  showMicEmpty('lost');
}

/* ------- mic level meter ------- */

let micRafId = 0;

function updateMicMeter() {
  if (!whisper.mic || !whisper.mic.running) return;
  let lvl = 0;
  // Display scaling: SottoMic.level() is raw smoothed RMS, and its JSDoc puts
  // normal speech around 0.05-0.3 and whispers around 0.01-0.1 — raw values
  // would leave the bar nearly empty. sqrt lifts the whisper range; the 1.8
  // gain puts normal speech near full scale.
  try {
    lvl = clamp01(Math.sqrt(Number(whisper.mic.level()) || 0) * 1.8);
  } catch {
    lvl = 0;
  }
  micLevel.style.width = `${Math.round(lvl * 100)}%`;
}

function micMeterLoop() {
  micRafId = 0;
  if (!whisper.mic || !whisper.mic.running) {
    micMeter.hidden = true;
    checkMicDisconnect();
    return;
  }
  // The engine's onFrame drives the meter while the camera loop runs; this
  // fallback keeps it honest when the mic is live but the camera is off.
  if (!running) updateMicMeter();
  micRafId = requestAnimationFrame(micMeterLoop);
}

function startMicMeter() {
  micMeter.hidden = false;
  if (!micRafId) micRafId = requestAnimationFrame(micMeterLoop);
}

function stopMicMeter() {
  if (micRafId) {
    cancelAnimationFrame(micRafId);
    micRafId = 0;
  }
  micMeter.hidden = true;
  micLevel.style.width = '0%';
}

/* ------- mode switch wiring ------- */

modePhrases.addEventListener('change', () => {
  if (modePhrases.checked) setMode('phrases');
});
modeWhisper.addEventListener('change', () => {
  if (modeWhisper.checked) setMode('whisper');
});
modeWhisper.addEventListener('click', () => {
  // A deferred restore leaves this radio checked with the mic off, and
  // re-clicking a checked radio fires no 'change'. The click itself is the
  // gesture that arms the real entry lifecycle.
  runDeferredWhisper();
});

/* ---------------------------------------------------------------- phrasebook */

function libraryOf() {
  try { return engine.getLibrary() || []; } catch { return []; }
}

function templatesOf(label) {
  const entry = libraryOf().find((p) => p.label === label);
  return entry ? entry.templates : 0;
}

function refreshPhrasebook() {
  const lib = libraryOf();
  phraseList.textContent = '';
  bookEmpty.hidden = lib.length > 0;
  lib.forEach((p) => phraseList.appendChild(phraseRow(p)));
}

function phraseRow(p) {
  const li = document.createElement('li');
  li.className = 'phrase-row';

  const label = document.createElement('span');
  label.className = 'plabel';
  label.textContent = p.label;

  const count = document.createElement('span');
  count.className = 'pcount';
  count.textContent = `${p.templates} ${p.templates === 1 ? 'take' : 'takes'}`;

  const actions = document.createElement('span');
  actions.className = 'pactions';

  const renameBtn = smallBtn('Rename');
  renameBtn.addEventListener('click', () => enterRename(li, p.label));

  const rerecordBtn = smallBtn('Re-record');
  rerecordBtn.addEventListener('click', () => openCalibration('rerecord', p.label));

  const deleteBtn = smallBtn('Delete');
  let armed = null;
  deleteBtn.addEventListener('click', () => {
    if (armed) {
      clearTimeout(armed);
      try { engine.deletePhrase(p.label); } catch { /* already gone */ }
      refreshPhrasebook();
      toast(`"${p.label}" deleted`, null, 'miss');
    } else {
      deleteBtn.textContent = 'Sure?';
      deleteBtn.classList.add('is-armed');
      armed = setTimeout(() => {
        armed = null;
        deleteBtn.textContent = 'Delete';
        deleteBtn.classList.remove('is-armed');
      }, 2500);
    }
  });

  actions.append(renameBtn, rerecordBtn, deleteBtn);
  li.append(label, count, actions);
  return li;
}

function smallBtn(text) {
  const b = document.createElement('button');
  b.className = 'btn-ghost btn-sm';
  b.type = 'button';
  b.textContent = text;
  return b;
}

function enterRename(row, oldLabel) {
  row.textContent = '';
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'rename-input';
  input.maxLength = 40;
  input.value = oldLabel;
  input.setAttribute('aria-label', `Rename phrase ${oldLabel}`);

  const actions = document.createElement('span');
  actions.className = 'pactions';
  const save = smallBtn('Save');
  const cancel = smallBtn('Cancel');

  const commit = () => {
    const next = input.value.trim();
    if (!next || next === oldLabel) { refreshPhrasebook(); return; }
    if (libraryOf().some((p) => p.label === next)) {
      toast(`"${next}" is already in the book`, null, 'bad');
      input.focus();
      return;
    }
    try {
      engine.renamePhrase(oldLabel, next);
    } catch (err) {
      toast(err && err.message ? err.message : 'rename failed', null, 'bad');
    }
    refreshPhrasebook();
  };

  save.addEventListener('click', commit);
  cancel.addEventListener('click', () => refreshPhrasebook());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    if (e.key === 'Escape') { e.stopPropagation(); refreshPhrasebook(); }
  });

  actions.append(save, cancel);
  row.append(input, actions);
  input.focus();
  input.select();
}

/* ------- export / import ------- */

exportBtn.addEventListener('click', () => {
  let json;
  try { json = engine.exportLibrary(); } catch (err) {
    toast('export failed', null, 'bad');
    return;
  }
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sotto-phrasebook.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
});

importBtn.addEventListener('click', () => importFile.click());
importFile.addEventListener('change', async () => {
  const file = importFile.files && importFile.files[0];
  importFile.value = '';
  if (!file) return;
  let text;
  try { text = await file.text(); } catch {
    toast('could not read that file', null, 'bad');
    return;
  }
  try {
    engine.importLibrary(text);
    refreshPhrasebook();
    toast(`imported — ${libraryOf().length} phrases in the book`, null);
  } catch (err) {
    toast('that file is not a Sotto phrasebook — nothing changed', null, 'bad');
  }
});

/* ---------------------------------------------------------------- calibration flow */

function openDialog(step) {
  if (!calDialog.open) {
    lastFocus = document.activeElement;
    calDialog.showModal();
    cal.open = true;
  }
  showStep(step);
}

function showStep(step) {
  cal.step = step;
  for (const [name, el] of Object.entries(calSteps)) el.hidden = name !== step;
  const titles = {
    offer: 'Starter pack',
    camera: 'Camera needed',
    label: 'New phrase',
    record: 'Recording takes',
    done: 'Phrase learned',
  };
  calDialog.setAttribute('aria-label', `Calibration — ${titles[step] || 'Calibration'}`);
  const focusable = [...calSteps[step].querySelectorAll('input, button')]
    .find((el) => !el.disabled && !el.hidden);
  if (focusable) focusable.focus();
}

function closeDialog() {
  if (calDialog.open) calDialog.close();
}

calDialog.addEventListener('close', () => {
  clearTimeout(cal.timer);
  cal.timer = null;
  if (cal.awaiting) {
    try { engine.cancelRecording(); } catch { /* nothing pending */ }
    cal.awaiting = false;
  }
  if (cal.mode === 'starter') localStorage.setItem(STARTER_FLAG, '1');
  cal.open = false;
  cal.mode = null;
  cal.step = null;
  cal.intent = null;
  refreshPhrasebook();
  if (lastFocus && lastFocus.isConnected) lastFocus.focus();
  lastFocus = null;
});

function openCalibration(calMode, label) {
  // Calibration is Phrases-mode-only. The camera-first gate must never fire in
  // Whisper mode, so drop back first (synchronously stops the mic).
  if (mode === 'whisper') setMode('phrases');
  cal.intent = { mode: calMode, label: label || '' };
  if (!running) {
    calCameraBody.textContent = 'Calibration records real takes, which needs the camera running. Nothing leaves this device.';
    openDialog('camera');
    return;
  }
  proceedWithIntent();
}

function proceedWithIntent() {
  const intent = cal.intent || { mode: 'single' };
  cal.intent = null;
  if (intent.mode === 'starter') {
    startStarter();
  } else if (intent.mode === 'rerecord') {
    cal.mode = 'rerecord';
    cal.label = intent.label;
    cal.oldCount = templatesOf(intent.label);
    beginTakesFor(intent.label);
  } else {
    cal.mode = 'single';
    calLabelInput.value = '';
    calLabelError.hidden = true;
    openDialog('label');
  }
}

function startStarter() {
  const have = new Set(libraryOf().map((p) => p.label));
  cal.queue = STARTER.filter((l) => !have.has(l));
  cal.queueIndex = 0;
  cal.mode = 'starter';
  if (!cal.queue.length) {
    calDoneTitle.textContent = 'Already done';
    calDoneMsg.textContent = 'The starter six are all in the book. Mouth one at the camera and watch the pad.';
    calNextBtn.textContent = 'Close';
    openDialog('done');
    return;
  }
  beginTakesFor(cal.queue[0]);
}

function beginTakesFor(label) {
  cal.label = label;
  cal.takesDone = 0;
  cal.awaiting = false;
  calWord.textContent = label;
  calSkip.hidden = cal.mode !== 'starter';
  if (cal.mode === 'starter') {
    calProgress.hidden = false;
    calProgress.textContent = `Phrase ${cal.queueIndex + 1} of ${cal.queue.length}`;
  } else {
    calProgress.hidden = true;
  }
  updateTakesUI();
  openDialog('record');
  scheduleTake(600);
}

function scheduleTake(delay) {
  clearTimeout(cal.timer);
  cal.timer = setTimeout(() => {
    cal.timer = null;
    if (!cal.open || cal.step !== 'record') return;
    if (!running) { abortTakesForCameraLoss(); return; }
    if (pauseToggle.checked) {
      pauseToggle.checked = false;
      engine.setPaused(false);
      renderStatus();
    }
    try {
      // A phrase can sit at the 8-take cap (imports merge same-label takes;
      // an abandoned re-record leaves old takes in place). Make room by
      // dropping the oldest take so recording can never overflow and stall.
      while (templatesOf(cal.label) >= CONSTANTS.MAX_TAKES) {
        engine.deleteTemplate(cal.label, 0);
        if (cal.oldCount > 0) cal.oldCount--;
      }
      engine.beginRecording(cal.label);
    } catch (err) {
      setCalLive(`Recording failed: ${err && err.message ? err.message : err}`);
      return;
    }
    cal.awaiting = true;
    updateCalLive();
  }, delay);
}

function updateTakesUI() {
  const dots = calDots.children;
  for (let i = 0; i < dots.length; i++) {
    dots[i].classList.toggle('done', i < cal.takesDone);
  }
  const current = Math.min(cal.takesDone + 1, TAKES_TARGET);
  calTakeText.textContent = cal.takesDone >= TAKES_TARGET
    ? `all ${TAKES_TARGET} takes captured`
    : `take ${current} of ${TAKES_TARGET}`;
  calRedo.disabled = cal.takesDone === 0;
}

function setCalLive(text) {
  calLive.textContent = text;
}

function updateCalLive() {
  if (!cal.open || cal.step !== 'record') return;
  if (!cal.awaiting) return;
  if (engineState === 'speaking') setCalLive('Capturing…');
  else if (engineState === 'no-face') setCalLive('Face lost — center yourself, then mouth it.');
  else setCalLive('Watching — mouth it when ready.');
}

function finishPhrase() {
  const label = cal.label;
  if (cal.mode === 'rerecord' && cal.oldCount > 0) {
    for (let i = 0; i < cal.oldCount; i++) {
      try { engine.deleteTemplate(label, 0); } catch { break; }
    }
    cal.oldCount = 0;
  }
  refreshPhrasebook();

  if (cal.mode === 'starter') {
    const next = cal.queue[cal.queueIndex + 1];
    calDoneTitle.textContent = 'Learned';
    calDoneMsg.textContent = `"${label}" is in the book — three takes.`;
    calNextBtn.textContent = next ? `Next: "${next}"` : 'Finish';
  } else if (cal.mode === 'rerecord') {
    calDoneTitle.textContent = 'Re-recorded';
    calDoneMsg.textContent = `"${label}" has three fresh takes. The old ones are gone.`;
    calNextBtn.textContent = 'Done';
  } else {
    calDoneTitle.textContent = 'Learned';
    calDoneMsg.textContent = `"${label}" is in the book. Mouth it at the camera and watch the pad.`;
    calNextBtn.textContent = 'Done';
  }
  showStep('done');
}

calNextBtn.addEventListener('click', () => {
  if (cal.mode === 'starter' && cal.queueIndex + 1 < cal.queue.length) {
    cal.queueIndex += 1;
    beginTakesFor(cal.queue[cal.queueIndex]);
  } else {
    if (cal.mode === 'starter') {
      localStorage.setItem(STARTER_FLAG, '1');
      toast(`${libraryOf().length} phrases in the book`, null);
    }
    closeDialog();
  }
});

calRedo.addEventListener('click', () => {
  if (cal.takesDone === 0) return;
  clearTimeout(cal.timer);
  if (cal.awaiting) {
    try { engine.cancelRecording(); } catch { /* fine */ }
    cal.awaiting = false;
  }
  const count = templatesOf(cal.label);
  if (count > 0) {
    try { engine.deleteTemplate(cal.label, count - 1); } catch { /* fine */ }
  }
  cal.takesDone -= 1;
  refreshPhrasebook();
  updateTakesUI();
  setCalLive('Last take dropped — go again.');
  scheduleTake(500);
});

calSkip.addEventListener('click', () => {
  clearTimeout(cal.timer);
  if (cal.awaiting) {
    try { engine.cancelRecording(); } catch { /* fine */ }
    cal.awaiting = false;
  }
  if (cal.queueIndex + 1 < cal.queue.length) {
    cal.queueIndex += 1;
    beginTakesFor(cal.queue[cal.queueIndex]);
  } else {
    localStorage.setItem(STARTER_FLAG, '1');
    closeDialog();
  }
});

$('cal-cancel').addEventListener('click', closeDialog);
$('cal-label-cancel').addEventListener('click', closeDialog);
$('cal-offer-later').addEventListener('click', () => {
  localStorage.setItem(STARTER_FLAG, '1');
  cal.mode = null;
  closeDialog();
});
$('cal-offer-start').addEventListener('click', () => startStarter());
$('cal-cam-close').addEventListener('click', closeDialog);
$('cal-cam-start').addEventListener('click', async () => {
  await startCamera();
  if (running) {
    proceedWithIntent();
  } else {
    calCameraBody.textContent = 'The browser refused camera access. Allow it in the address bar, then try again.';
  }
});

$('cal-begin-btn').addEventListener('click', () => {
  const label = calLabelInput.value.trim();
  if (!label) {
    calLabelError.textContent = 'A phrase needs a label — that is what the pad will type.';
    calLabelError.hidden = false;
    calLabelInput.focus();
    return;
  }
  if (libraryOf().some((p) => p.label === label)) {
    calLabelError.textContent = 'Already in the book. Use Re-record on it instead.';
    calLabelError.hidden = false;
    calLabelInput.focus();
    return;
  }
  calLabelError.hidden = true;
  cal.mode = 'single';
  beginTakesFor(label);
});
calLabelInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); $('cal-begin-btn').click(); }
});

function abortTakesForCameraLoss() {
  clearTimeout(cal.timer);
  if (cal.awaiting) {
    try { engine.cancelRecording(); } catch { /* fine */ }
    cal.awaiting = false;
  }
  cal.intent = { mode: cal.mode || 'single', label: cal.label };
  calCameraBody.textContent = 'The camera stopped mid-calibration. Start it again to pick up where you left off.';
  showStep('camera');
}

addPhraseBtn.addEventListener('click', () => openCalibration('single'));

function maybeOfferStarter() {
  if (mode !== 'phrases') return; // never offer while in Whisper mode
  if (starterOffered || cal.open) return;
  if (localStorage.getItem(STARTER_FLAG)) return;
  if (libraryOf().length > 0) return;
  starterOffered = true;
  cal.mode = 'starter'; // an Escape here counts as "decided", via the close handler
  openDialog('offer');
}

/* ---------------------------------------------------------------- settings drawer */

function setDrawer(open) {
  settingsDrawer.hidden = !open;
  settingsBtn.setAttribute('aria-expanded', String(open));
  if (open) settingsClose.focus();
  else settingsBtn.focus();
}
settingsBtn.addEventListener('click', () => setDrawer(settingsDrawer.hidden));
settingsClose.addEventListener('click', () => setDrawer(false));
settingsDrawer.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') setDrawer(false);
});

starterBtn.addEventListener('click', () => {
  setDrawer(false);
  openCalibration('starter');
});

/* ------- pipeline self-test (simulated input) ------- */

function showSelfTest(msg) {
  selftestOut.hidden = false;
  selftestMsg.textContent = msg;
}

selftestBtn.addEventListener('click', () => {
  if (simulated) return;
  if (!libraryOf().length) {
    showSelfTest('Needs at least one calibrated phrase — the matcher has nothing to compare against.');
    return;
  }
  selftestBtn.disabled = true;
  simulated = {
    timer: setTimeout(() => {
      simulated = null;
      selftestBtn.disabled = false;
      showSelfTest('No result came back within two seconds. If the model is still loading, wait for "watching" and run it again.');
    }, 2000),
  };
  try {
    engine.debugInjectSegment(SottoEngine.syntheticSegment('a'));
  } catch (err) {
    clearTimeout(simulated.timer);
    simulated = null;
    selftestBtn.disabled = false;
    showSelfTest(`The injection itself failed: ${err && err.message ? err.message : 'unknown error'}.`);
  }
});

function reportSelfTest(m, seg) {
  let stats = [];
  try { stats = engine.matchStats(seg) || []; } catch { stats = []; }
  if (m) {
    showSelfTest(`Synthetic segment went through the full pipeline and matched "${m.label}" at confidence ${fmtConf(m.confidence)}. Segmenter and matcher are alive.`);
  } else if (stats.length) {
    showSelfTest(`Synthetic segment went through the full pipeline. No match — closest was "${stats[0].label}" at confidence ${fmtConf(stats[0].confidence)}, below the acceptance bar. For made-up input, that is the correct answer.`);
  } else {
    showSelfTest('Synthetic segment went through the full pipeline. No match, no candidates — the matcher declined it cleanly.');
  }
}

/* ---------------------------------------------------------------- service worker */

if ('serviceWorker' in navigator
    && location.hostname !== 'localhost'
    && location.hostname !== '127.0.0.1') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => { /* offline shell is optional */ });
  });
}

/* ---------------------------------------------------------------- init */

showCamEmpty('off');
renderStatus();
refreshPhrasebook();
sizeWave();
drawWave();
growPad();
sensOut.textContent = Number(sensitivity.value).toFixed(2);

// Restore the persisted input mode. A stored 'whisper' only sets the UI —
// no getUserMedia, no model load, no permission prompt without a gesture.
// The first gesture that implies Whisper use (Start camera, or a click on
// the Whisper radio) runs the real entry lifecycle via runDeferredWhisper.
syncModeUI();
let storedMode = null;
try { storedMode = localStorage.getItem(MODE_KEY); } catch { storedMode = null; }
if (storedMode === 'whisper') {
  mode = 'whisper';
  whisper.deferred = true;
  syncModeUI();
}
