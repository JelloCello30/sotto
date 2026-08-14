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
  let key = engineState;
  let cls;
  let text;
  if (running && pauseToggle.checked && !['no-camera', 'error', 'loading'].includes(key)) {
    text = 'paused';
    cls = 'is-warn';
  } else {
    [text, cls] = NAV_STATUS[key] || NAV_STATUS.off;
  }
  statusPill.className = `pill ${cls}`.trim();
  statusText.textContent = text;

  const camText = CAM_PILL_TEXT[key] || 'camera off';
  camPillText.textContent = camText;
  camPill.className = 'pill cam-pill' +
    (key === 'speaking' ? ' is-live' : ['ready', 'idle', 'no-face'].includes(key) ? ' is-good' : '');
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

function showCamEmpty(kind, detail) {
  const copy = EMPTY_COPY[kind] || EMPTY_COPY.off;
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

window.__sotto = { engine };

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
  renderStatus();
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

function openCalibration(mode, label) {
  cal.intent = { mode, label: label || '' };
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
