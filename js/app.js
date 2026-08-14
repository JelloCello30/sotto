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

const micChip = $('mic-chip');
const micChipText = $('mic-chip-text');
const camPrivacy = $('cam-privacy');
const micMeter = $('mic-meter');
const micLevel = $('mic-level');
const asrNote = $('asr-note');
const asrNoteText = $('asr-note-text');
const asrProgress = $('asr-progress');
const asrProgressFill = $('asr-progress-fill');
const ticker = $('ticker');
const micEmpty = $('mic-empty');
const micEmptyTitle = $('mic-empty-title');
const micEmptyBody = $('mic-empty-body');
const micEmptyRetry = $('mic-empty-retry');
const micEmptyDismiss = $('mic-empty-dismiss');
const asrBlock = $('asr-block');
const asrInfo = $('asr-info');

const settingsBtn = $('settings-btn');
const settingsDrawer = $('settings-drawer');
const settingsClose = $('settings-close');
const selftestBtn = $('selftest-btn');
const selftestOut = $('selftest-out');
const selftestMsg = $('selftest-msg');
const starterBtn = $('starter-btn');
const voiceToggle = $('voice-toggle');
const bookEmptyStarter = $('book-empty-starter');

const calDialog = $('cal-dialog');
const calSteps = {
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
const TAKES_TARGET = 3;

let running = false;        // camera loop active
let engineState = 'off';    // last engine state string, or 'off'
let simulated = null;       // { timer } while a self-test injection is in flight
let autoCopyWarned = false;

/* ------- voice assist (v0.4: one mode, the mic assists it) ------- */

const VOICE_KEY = 'sotto.voice.v1';
const OBSOLETE_MODE_KEY = 'sotto.mode.v1'; // v0.2-v0.3 mode radiogroup; key removed on init

let voiceEnabled = true;    // the 'sotto.voice.v1' setting (default ON)
let micBlocked = false;     // mic denied / missing / lost since the last attempt

const voice = {
  session: 0,       // bumped on every stop; stale async completions check it and bail
  starting: false,  // start lifecycle in flight (re-entry guard)
  mic: null,        // SottoMic instance, created lazily on first start
  asr: null,        // SottoASR instance, created lazily on first start
  modPromise: null, // Promise for the dynamic imports of audio.js + asr.js
  ready: false,     // ASR has reported ready at least once (it stays loaded)
  queue: [],        // pending jobs {audio: Float32Array, seg}; cap 3, drop oldest
  busy: false,      // one transcribe call in flight
  droppedToast: false, // "transcriber is behind" toast shown at most once
  device: null,     // 'gpu' | 'cpu' once known
  model: null,      // model name once known
};

/* ------- per-utterance routing (v0.4) ------- */

const PHRASE_FAST_CONF = 0.80;   // instant-path confidence bar (SPEC-V4 rule 2)
const LEARNED_MAX = 40;          // cap on auto-learned phrases
const LEARN_TOAST_MS = 10000;    // at most one "learned" toast per 10 s
const HINT_THROTTLE_MS = 30000;  // silent-miss hint at most once per 30 s
const HINT_SHOW_MS = 4000;
const HINT_TEXT = "couldn't read that silently — whisper it once and Sotto learns it";
const LEARN_LABEL_RE = /^[\p{L}\p{N}' -]+$/u;

let pendingSeg = null; // buffered in onSegment; consumed by the onMatch that follows
let hintShownAt = 0;
let hintTimer = null;
let learnToastAt = 0;

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
  // Voice assist: watching -> speaking… -> transcribing… -> back. 'speaking…'
  // wins while a new utterance is in progress; 'transcribing…' covers the gap
  // while the ASR queue drains.
  const transcribing = !paused
    && (voice.busy || voice.queue.length > 0)
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

// With voice assist on, the "no audio" line above would break a promise:
// starting the camera is the gesture that also turns on the microphone.
const EMPTY_COPY_VOICE_OFF = {
  title: 'Camera is off',
  body: 'Sotto reads lip movement through your camera; while it runs, voice assist listens through the microphone. Starting the camera turns both on — everything is processed on this device, nothing uploaded.',
  btn: 'Start camera',
};

let lastEmptyKind = 'off'; // so a settings change can refresh the visible copy

function showCamEmpty(kind, detail) {
  lastEmptyKind = kind;
  const copy = (kind === 'off' && voiceEnabled)
    ? EMPTY_COPY_VOICE_OFF
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
    engineState = state;
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
    if (voice.mic && voice.mic.running) updateMicMeter();
    else checkMicDisconnect();
    const now = performance.now();
    if (now - lastFpsAt > 500) {
      lastFpsAt = now;
      fpsEl.textContent = `${Math.round(frame.fps)} fps`;
    }
  },

  onSegment(seg) {
    // Rule 1: a pending calibration take — exactly the v0.1 takes flow.
    if (cal.open && cal.awaiting && seg.recordingLabel === cal.label) {
      cal.awaiting = false;
      // Storage can fail mid-flow (auto-learn may race the caps between
      // beginRecording and this take landing) — do not count a failed take.
      if (seg.recordingError) {
        setCalLive(`Take failed: ${seg.recordingError} — go again.`);
        scheduleTake(900);
        return;
      }
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
    // Buffer for single-point routing: the engine fires onMatch synchronously
    // after onSegment for the same segment, so the decision happens there with
    // both pieces in hand. Self-test injections stay out of it.
    if (!seg.recordingLabel && !simulated) pendingSeg = seg;
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

    const buffered = pendingSeg;
    pendingSeg = null;
    if (!seg || buffered !== seg) return; // discarded between the two callbacks
    routeSegment(m, seg);
  },
});

/**
 * v0.4 single-point router (SPEC-V4 §2). Runs once per completed
 * non-recording segment, with the segment and its DTW match both in hand.
 * Rule order — first hit wins:
 *   2. a confident phrase (>= PHRASE_FAST_CONF) types instantly, silent or
 *      voiced, and never goes to the transcriber;
 *   3. a voiced segment (audioLevel >= AUDIO_ACTIVE_RMS) with voice assist
 *      ready goes to the Whisper path;
 *   4. anything else types nothing — a low-confidence match would type wrong
 *      words, so the throttled silent-miss hint explains the learning loop.
 * (Rule 1, calibration takes, never reaches this function — see onSegment.)
 */
function routeSegment(m, seg) {
  if (practiceToggle.checked) {
    let stats = [];
    try { stats = engine.matchStats(seg); } catch { stats = []; }
    renderPractice(stats, m && m.confidence >= PHRASE_FAST_CONF ? m : null);
  }

  if (m && m.confidence >= PHRASE_FAST_CONF) {
    appendToPad(m.label);
    toast(m.label, fmtConf(m.confidence));
    // Instant use freshens the phrase for learned-cap eviction ordering.
    if (typeof engine.touchPhrase === 'function') {
      try { engine.touchPhrase(m.label); } catch { /* bookkeeping only */ }
    }
    return;
  }

  const voiced = Number(seg.audioLevel) >= CONSTANTS.AUDIO_ACTIVE_RMS;
  if (voiced && voiceEnabled && voice.ready && voice.mic && voice.mic.running && voice.asr) {
    queueVoiceSegment(seg);
    return;
  }

  showSilentMissHint();
}

let lastFpsAt = 0;

window.__sotto = { engine, voice };

/* ---------------------------------------------------------------- camera control */

async function startCamera() {
  if (running) return;
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
    // The Start-camera click is the gesture that arms voice assist: only now,
    // with the camera running, may the mic start and the ASR load (deferred-
    // gesture rule — page-load restore never hot-mics).
    if (voiceEnabled) startVoiceAssist();
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
  // The mic exists to assist the camera flow; stopping the camera releases it
  // too ("Camera is off ... nothing uploaded" must stay true). The setting is
  // untouched — the next camera start re-arms voice assist.
  stopVoiceAssist();
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
  if (pauseToggle.checked) {
    // Pause means pause: queued audio does not outlive it, an in-flight
    // result is dropped when it lands (see pumpTranscribe), and the ticker
    // stops streaming its partials.
    voice.queue.length = 0;
    clearTicker();
  }
  renderStatus();
});

/* ---------------------------------------------------------------- voice assist */

// One honest line: app.js cannot reliably tell a first-ever load (network
// fetch) from a cached one, so the copy covers both without overpromising.
const ASR_LOADING_TEXT =
  'Loading the speech model. First ever load fetches ~75MB; after that it comes from cache.';

const PRIVACY_MIC_ON =
  'Camera and microphone are live. Everything runs on this device; nothing is uploaded.';
const PRIVACY_MIC_OFF =
  'Camera only. The microphone is off; silent phrases still work.';

const MIC_EMPTY_COPY = {
  denied: {
    title: 'Microphone blocked',
    body: 'Voice assist needs the microphone; the browser refused it. Allow it in the address bar (or site permissions), then try again. Silent-only until you re-enable the microphone — the camera keeps working.',
    retry: 'Try voice assist again',
  },
  none: {
    title: 'No microphone found',
    body: 'Voice assist needs a microphone and the browser could not find one. Connect one and try again. Silent-only until you re-enable the microphone — the camera keeps working.',
    retry: 'Try voice assist again',
  },
  asr: {
    title: 'Speech model failed to load',
    body: 'Voice assist could not start its on-device speech model. Silent phrases keep working.',
    retry: 'Try again',
  },
  lost: {
    title: 'Microphone disconnected',
    body: 'The microphone went away mid-session — unplugged, or the system withdrew it. Nothing has been transcribed since. Silent-only until you re-enable the microphone — the camera keeps working.',
    retry: 'Try voice assist again',
  },
};

/**
 * Set the voice-assist setting ('sotto.voice.v1'), sync the switch, and run
 * the matching lifecycle: OFF releases the mic immediately (queue and ticker
 * cleared, provider withdrawn); ON re-runs the mic+ASR entry — but only when
 * the camera is already running, since starting it is the arming gesture.
 * With the camera off the setting just waits for the next camera start, so
 * nothing can hot-mic outside a user gesture.
 * @param {boolean} on
 */
function setVoiceEnabled(on) {
  voiceEnabled = !!on;
  try { localStorage.setItem(VOICE_KEY, voiceEnabled ? '1' : '0'); } catch { /* not persisted */ }
  voiceToggle.checked = voiceEnabled;
  if (voiceEnabled) {
    micBlocked = false;
    micLostShown = false;
    hideMicEmpty();
    if (running) startVoiceAssist();
  } else {
    stopVoiceAssist();
  }
  renderMicChip();
  renderPrivacy();
  if (!camEmpty.hidden && lastEmptyKind === 'off') showCamEmpty('off');
}

/**
 * Start the mic and lazily load the ASR (v0.3 enterWhisper lifecycle).
 * Called only downstream of a real user gesture. Any failure — mic denied,
 * no mic, model load — drops voice assist to OFF; the camera flow continues
 * untouched and the mic empty state offers the retry.
 */
async function startVoiceAssist() {
  if (!voiceEnabled || voice.starting) return;
  if (voice.mic && voice.mic.running && voice.ready) return;
  const session = ++voice.session;
  voice.starting = true;
  micLostShown = false;
  hideMicEmpty();
  try {
    // Lazy: silent-only users never pay for the audio/ASR modules.
    if (!voice.modPromise) {
      voice.modPromise = Promise.all([import('./audio.js'), import('./asr.js')]);
    }
    let mods;
    try {
      mods = await voice.modPromise;
    } catch (err) {
      voice.modPromise = null; // a retry should re-attempt the import
      throw err;
    }
    if (session !== voice.session) return;
    const [audioMod, asrMod] = mods;
    // stop() closes the AudioContext, so a stopped mic gets a fresh instance
    // rather than assuming it can restart.
    if (!voice.mic || !voice.mic.running) voice.mic = new audioMod.SottoMic();
    if (!voice.asr) voice.asr = new asrMod.SottoASR({ onStatus: onAsrStatus });
    // Capture the instance this attempt owns: a stale attempt must never stop
    // voice.mic itself — a rapid off-then-on may have installed a newer mic
    // there, and stopping it would kill the live session's microphone.
    const mic = voice.mic;
    if (!mic.running) await mic.start();
    if (session !== voice.session) {
      if (voice.mic !== mic) {
        try { mic.stop(); } catch { /* already stopped */ }
      }
      return;
    }
    micBlocked = false;
    startMicMeter();
    renderMicChip();
    renderPrivacy();
    // Fusion wiring: the live mic level loosens/holds the lip gate. splitOnMax
    // is set once at init and stays on either way (v0.4). Guarded: the engine
    // API may lag this wiring during development.
    if (engine.setAudioLevelProvider) {
      engine.setAudioLevelProvider(() => (voice.mic && voice.mic.running ? voice.mic.level() : 0));
    }
    if (!voice.ready) showAsrNote(ASR_LOADING_TEXT, null);
    await voice.asr.load();
    if (session !== voice.session) return;
    voice.ready = true;
    hideAsrNote();
    renderAsrInfo();
    renderStatus();
  } catch (err) {
    if (session !== voice.session) return;
    hideAsrNote();
    const code = err && err.code;
    micBlocked = code === 'mic-denied' || code === 'mic-none';
    setVoiceEnabled(false); // releases the mic; the camera flow continues
    if (code === 'mic-denied') showMicEmpty('denied');
    else if (code === 'mic-none') showMicEmpty('none');
    else showMicEmpty('asr', err && err.message ? err.message : String(err || 'unknown error'));
  } finally {
    if (session === voice.session) voice.starting = false;
    renderMicChip();
    renderPrivacy();
  }
}

/**
 * Release the mic and stop routing to the transcriber (v0.3 leaveWhisper
 * semantics): ticker cleared, queue cleared, provider null. Deliberate v0.4
 * change: splitOnMax STAYS on, so long silent mouthing still splits into
 * matchable segments rather than being discarded. Segment matching keeps
 * running; the ASR stays loaded so re-enabling voice assist is instant.
 */
function stopVoiceAssist() {
  voice.session += 1; // any in-flight transcription result is discarded
  voice.starting = false;
  voice.queue.length = 0;
  voice.busy = false;
  clearTicker();
  if (engine.setAudioLevelProvider) engine.setAudioLevelProvider(null);
  if (voice.mic) {
    try { voice.mic.stop(); } catch { /* already stopped */ }
  }
  stopMicMeter();
  hideAsrNote();
  renderMicChip();
  renderPrivacy();
  renderStatus();
}

/* ------- status strip: mic chip + privacy line ------- */

function renderMicChip() {
  let text = 'mic off';
  let cls = '';
  if (voice.mic && voice.mic.running) {
    text = 'mic on';
    cls = ' is-good';
  } else if (micBlocked) {
    text = 'mic blocked';
    cls = ' is-bad';
  }
  micChipText.textContent = text;
  micChip.className = 'pill mic-chip' + cls;
  micChip.setAttribute('aria-label', `Microphone ${text.slice(4)} — voice assist settings`);
}

function renderPrivacy() {
  camPrivacy.textContent = voice.mic && voice.mic.running ? PRIVACY_MIC_ON : PRIVACY_MIC_OFF;
}

/* ------- fusion: lip segment -> audio slice -> transcription ------- */

function queueVoiceSegment(seg) {
  if (!voice.mic || !voice.mic.running || !voice.asr || !voice.ready) return;
  let audio;
  try {
    audio = voice.mic.slice(seg.t0 - 250, seg.t1 + 250);
  } catch {
    return;
  }
  if (!audio || !audio.length) return;
  // The segment rides along so a final text can auto-learn from its frames.
  voice.queue.push({ audio, seg });
  while (voice.queue.length > 3) {
    voice.queue.shift();
    if (!voice.droppedToast) {
      voice.droppedToast = true;
      toast('transcriber is behind — dropped a segment', null, 'miss');
    }
  }
  pumpTranscribe();
}

function pumpTranscribe() {
  if (voice.busy || !voice.queue.length) {
    renderStatus();
    return;
  }
  const session = voice.session;
  const job = voice.queue.shift();
  voice.busy = true;
  renderStatus();
  voice.asr.transcribe(job.audio, {
    // Accumulated partial text of this job, streamed from the decoder. Stale
    // sessions and paused states stay silent; the final result supersedes.
    onPartial: (text) => {
      if (session !== voice.session) return;
      if (pauseToggle.checked) return;
      setTicker(text);
    },
  }).then((text) => {
    if (session !== voice.session) return;
    // Pause honesty: while the pill says "paused", a result that was already
    // in flight must not reach the pad, a toast, or the clipboard.
    if (pauseToggle.checked) return;
    const clean = (text || '').trim();
    if (clean) {
      appendToPad(clean); // verbatim beyond appendToPad's usual sentence-start casing
      toastTranscript(clean);
      // The pad received it — this piece's own final text may now teach the
      // silent vocabulary from this piece's own frames (SPEC-V4 §3).
      maybeAutoLearn(clean, job.seg);
    }
  }).catch(() => {
    if (session !== voice.session) return;
    toast('transcription failed', null, 'bad');
  }).then(() => {
    if (session !== voice.session) return;
    clearTicker(); // this job is settled either way; the pad holds the final text
    voice.busy = false;
    renderStatus();
    pumpTranscribe();
  });
}

/* ------- auto-learning (SPEC-V4 §3) ------- */

/** Strip surrounding punctuation and collapse whitespace for a learn label. */
function strippedLearnLabel(text) {
  return String(text || '')
    .replace(/^[^\p{L}\p{N}]+/u, '')
    .replace(/[^\p{L}\p{N}]+$/u, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * After a Whisper final text reached the pad: enroll the segment's lip frames
 * as a silent template for that text, when every guard holds. Enrollment is
 * best-effort and silent on failure; at most one "learned" toast per 10 s,
 * and only for a first-time label. Guarded on engine.enrollSegment so a
 * lagging engine build degrades to no-op.
 */
function maybeAutoLearn(text, seg) {
  if (typeof engine.enrollSegment !== 'function') return;
  if (!seg || !Array.isArray(seg.frames) || seg.frames.length < 2) return;
  const dur = Number(seg.durationMs);
  if (!(dur >= 400 && dur <= 3000)) return;
  const stripped = strippedLearnLabel(text);
  if (stripped.length < 2 || stripped.length > 40) return;
  if (!LEARN_LABEL_RE.test(stripped)) return;
  if (stripped.split(' ').length > 4) return; // 1-4 words (never 0 after the strip)
  const label = stripped.toLowerCase();

  const lib = libraryOf();
  // Case-insensitive: a calibrated "On my way" must gain a take from a
  // transcribed "on my way", not a lowercase learned twin (twins split usage
  // credit and can push the DTW margin test into ambiguity).
  const existing = lib.find((p) => p.label.toLowerCase() === label);
  if (!existing) {
    // Caps: LEARNED_MAX learned phrases, MAX_PHRASES total. Creating beyond
    // either cap first evicts the learned phrase with the stalest
    // (lastUsedAt ?? createdAt) — never a calibrated phrase. A full library
    // with nothing learned to evict skips enrollment silently.
    const learned = lib.filter((p) => p.learned === true);
    if (learned.length >= LEARNED_MAX || lib.length >= CONSTANTS.MAX_PHRASES) {
      if (!learned.length) return;
      let oldest = learned[0];
      for (const p of learned) {
        if (((p.lastUsedAt ?? p.createdAt) || 0) < ((oldest.lastUsedAt ?? oldest.createdAt) || 0)) {
          oldest = p;
        }
      }
      try { engine.deletePhrase(oldest.label); } catch { return; }
    }
  }
  try {
    // Existing label (calibrated or learned): adds a take under the existing
    // casing, engine evicts take 0 at the cap. New label: created flagged
    // learned.
    engine.enrollSegment(existing ? existing.label : label, seg.frames, { learned: true });
  } catch {
    return; // invalid frames or a race on the caps — enrollment is best-effort
  }
  refreshPhrasebook();
  if (!existing) {
    const now = Date.now();
    if (now - learnToastAt >= LEARN_TOAST_MS) {
      learnToastAt = now;
      toast(`learned "${label}" — mouth it silently next time`, null);
    }
  }
}

function toastTranscript(text) {
  const words = text.split(/\s+/);
  const head = words.slice(0, 4).join(' ');
  toast(words.length > 4 ? `${head}…` : head, null);
}

/* ------- live ticker + silent-miss hint ------- */

// A single line above the pad. Two writers: the currently-decoding job's
// accumulated partial text, and the throttled silent-miss hint (routing rule
// 4). Partials always win — a hint never stomps a streaming job, and a new
// partial replaces a visible hint. Empty and hidden otherwise.

function setTicker(text, isHint) {
  const t = (text || '').trim();
  ticker.textContent = t;
  ticker.hidden = t === '';
  ticker.classList.toggle('is-hint', !!isHint && t !== '');
  if (!isHint && hintTimer) {
    clearTimeout(hintTimer);
    hintTimer = null;
  }
}

function clearTicker() {
  ticker.textContent = '';
  ticker.hidden = true;
  ticker.classList.remove('is-hint');
  if (hintTimer) {
    clearTimeout(hintTimer);
    hintTimer = null;
  }
}

function showSilentMissHint() {
  const now = Date.now();
  if (now - hintShownAt < HINT_THROTTLE_MS) return;
  if (voice.busy || voice.queue.length) return; // never stomp a streaming job
  hintShownAt = now;
  setTicker(HINT_TEXT, true);
  hintTimer = setTimeout(() => {
    hintTimer = null;
    if (ticker.classList.contains('is-hint')) clearTicker();
  }, HINT_SHOW_MS);
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
      if (voiceEnabled) showAsrNote(ASR_LOADING_TEXT, pct);
      break;
    case 'warming':
      if (voiceEnabled) showAsrNote('Warming up the speech model…', null);
      break;
    case 'ready':
      voice.ready = true;
      captureAsrMeta(info);
      hideAsrNote();
      renderAsrInfo();
      break;
    case 'error':
      hideAsrNote(); // load() rejects too; the catch in startVoiceAssist reports it
      break;
  }
}

function captureAsrMeta(info) {
  const a = voice.asr || {};
  const dev = (info && typeof info === 'object' && (info.device || info.backend))
    || a.device || a.backend || null;
  const model = (info && typeof info === 'object' && (info.model || info.modelId))
    || a.model || a.modelId || a.modelName || null;
  if (dev) voice.device = /gpu/i.test(String(dev)) ? 'gpu' : 'cpu';
  if (model) voice.model = String(model);
}

function renderAsrInfo() {
  if (!voice.ready) return;
  const bits = [];
  bits.push(voice.model ? `Speech model ${voice.model}` : 'Speech model loaded');
  if (voice.device) bits.push(`running on ${voice.device}`);
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
  setVoiceEnabled(true); // a real gesture: re-runs the mic+ASR entry
});
micEmptyDismiss.addEventListener('click', () => hideMicEmpty());

/* ------- mic disconnect watchdog ------- */

// One-shot per voice-assist start: set when the disconnect state has been
// shown, reset by startVoiceAssist, so the polling paths cannot spam it.
let micLostShown = false;

/**
 * The mic track can end under us — headset unplugged, OS-level revocation —
 * and SottoMic stops itself silently when it does. Without this check the
 * chip would keep saying "mic on" while every voiced segment gets dropped by
 * queueVoiceSegment's running guard. Polled from onFrame and micMeterLoop.
 */
function checkMicDisconnect() {
  if (micLostShown) return;
  if (!voiceEnabled || voice.starting) return;
  if (!voice.mic || voice.mic.running) return;
  micLostShown = true;
  micBlocked = true;
  setVoiceEnabled(false); // drops voice assist; camera and silent phrases continue
  showMicEmpty('lost');
}

/* ------- mic level meter ------- */

let micRafId = 0;

function updateMicMeter() {
  if (!voice.mic || !voice.mic.running) return;
  let lvl = 0;
  // Display scaling: SottoMic.level() is raw smoothed RMS, and its JSDoc puts
  // normal speech around 0.05-0.3 and whispers around 0.01-0.1 — raw values
  // would leave the bar nearly empty. sqrt lifts the whisper range; the 1.8
  // gain puts normal speech near full scale.
  try {
    lvl = clamp01(Math.sqrt(Number(voice.mic.level()) || 0) * 1.8);
  } catch {
    lvl = 0;
  }
  micLevel.style.width = `${Math.round(lvl * 100)}%`;
}

function micMeterLoop() {
  micRafId = 0;
  if (!voice.mic || !voice.mic.running) {
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

/* ------- voice-assist wiring: mic chip + settings switch ------- */

micChip.addEventListener('click', () => setDrawer(true));

voiceToggle.addEventListener('change', () => {
  setVoiceEnabled(voiceToggle.checked);
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
  // Auto-learn can call this from the async transcription path; never rebuild
  // the list out from under an in-progress rename (it would eat the input).
  if (phraseList.querySelector('.rename-input')) return;
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

  // Auto-learned entries carry a quiet pill; calibrated rows are unchanged.
  let learnedPill = null;
  if (p.learned === true) {
    learnedPill = document.createElement('span');
    learnedPill.className = 'learned-pill';
    learnedPill.textContent = 'learned';
  }

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
  if (learnedPill) li.append(label, learnedPill, count, actions);
  else li.append(label, count, actions);
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
  cal.open = false;
  cal.mode = null;
  cal.step = null;
  cal.intent = null;
  refreshPhrasebook();
  if (lastFocus && lastFocus.isConnected) lastFocus.focus();
  lastFocus = null;
});

function openCalibration(calMode, label) {
  // Calibration runs inside the one flow: recording segments carry
  // recordingLabel and take routing rule 1, so a live mic cannot steal takes.
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
    closeDialog();
  }
});

$('cal-cancel').addEventListener('click', closeDialog);
$('cal-label-cancel').addEventListener('click', closeDialog);
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

// v0.4: the starter pack is never auto-offered. It lives behind the settings
// button and the phrasebook empty-state link only.
bookEmptyStarter.addEventListener('click', () => openCalibration('starter'));

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

// v0.4: the mode radiogroup is gone; drop its obsolete stored key.
try { localStorage.removeItem(OBSOLETE_MODE_KEY); } catch { /* nothing stored */ }

// Restore the voice-assist setting (default ON). Restoring only sets state —
// no getUserMedia, no model load, no permission prompt without a gesture.
// The Start-camera click is the gesture that runs the real entry lifecycle.
try { voiceEnabled = localStorage.getItem(VOICE_KEY) !== '0'; } catch { voiceEnabled = true; }
voiceToggle.checked = voiceEnabled;

// v0.4 segmentation: splitOnMax stays on with or without the mic, so long
// silent mouthing splits into matchable pieces rather than being discarded.
// Guarded: the engine API may lag this wiring during development.
if (engine.setSegmentOptions) engine.setSegmentOptions({ splitOnMax: true });

showCamEmpty('off');
renderStatus();
renderMicChip();
renderPrivacy();
refreshPhrasebook();
sizeWave();
drawWave();
growPad();
sensOut.textContent = Number(sensitivity.value).toFixed(2);
