/*
 * Sotto — ASR module worker (js/asr-worker.js)
 *
 * Runs the vendored Whisper model (transformers.js + ONNX Runtime) off the UI
 * thread. Fully on-device: allowRemoteModels is false and every asset URL is
 * derived from import.meta.url, so the worker resolves vendor/ correctly at
 * both the localhost root and a /sotto/ subpath deploy.
 *
 * Device policy: webgpu when present, wasm otherwise. webgpu is dropped for
 * wasm when it fails to build, throws at first inference, or fails SILENTLY
 * (some GPUs decode garbage with q8 weights and never throw) — the latter is
 * caught by a load-time canary transcription of vendored known speech, plus
 * a runtime decode-loop guard.
 *
 * Owned by js/asr.js (SottoASR). Not a public module — the only export is
 * MODEL_ID, kept here so swapping the model is a one-word change.
 *
 * Protocol (all messages carry a numeric id except worker-initiated status):
 *   main -> worker : { id, type: 'load', payload?: { device?: 'auto' | 'wasm' } }
 *                    { id, type: 'transcribe',
 *                      payload: { buffer, byteOffset, length } }   [buffer transferred]
 *                    { id, type: 'dispose' }
 *   worker -> main : { id, type: 'result', payload: { text?, device, model? } }
 *                      [load results carry model: MODEL_ID]
 *                    { id, type: 'partial', payload: { text } }
 *                      [ACCUMULATED partial transcription for the in-flight
 *                       transcribe job, at most one message per 120 ms, no
 *                       trailing flush — the final result supersedes; never
 *                       emitted for the canary or warm-up runs]
 *                    { id, type: 'error',  payload: { message } }
 *                    {     type: 'status', payload: { status, model, progress?, device?, message? } }
 *                      status: 'loading-model' | 'warming' | 'ready' | 'error'
 */

/**
 * Which vendored Whisper checkpoint to load, as a directory name under
 * vendor/whisper/. One word: 'tiny' or 'base'.
 */
export const MODEL_ID = 'base';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** Warm-up input: 0.5 s of silence at 16 kHz, so the first real job is fast. */
const WARMUP_SAMPLES = 8000;

/**
 * Minimum gap between partial-transcription messages for one job. The
 * decoder emits tokens far faster than anyone reads; one update per 120 ms
 * looks live on screen without flooding the main thread. No trailing flush
 * — the final result message supersedes whatever partial rendered last.
 */
const PARTIAL_MIN_INTERVAL_MS = 120;

/** Inputs quieter than this RMS are treated as silence for hallucination suppression. */
const SILENCE_RMS = 0.004;

/**
 * Classic Whisper silence hallucinations, pre-normalized (lowercased, all
 * punctuation and whitespace stripped). '.' normalizes to '' and is covered
 * by the empty entry.
 */
const SILENCE_HALLUCINATIONS = new Set(['', 'thankyou', 'thanksforwatching', 'you']);

/**
 * Decode budget: Whisper averages roughly 4-5 tokens per second of speech;
 * this cap is ~3x that, so real dictation never hits it while a runaway
 * decode loop (a symptom of a silently broken GPU backend) is cut off in
 * seconds instead of grinding to the model's 448-token ceiling.
 */
const TOKENS_PER_SECOND_CAP = 16;
const MIN_NEW_TOKENS_CAP = 32;

/**
 * Decode-loop detection, a runtime second line of defense against GPUs
 * whose q8 kernels are silently broken: output logits are garbage and the
 * decoder repeats one word or short phrase until it exhausts the token
 * budget. That exhaustion is the tell — no verdict is reached unless the
 * output's word count reached LOOP_NEAR_CAP of the job's max_new_tokens
 * (word count is a floor on token count, so the gate is conservative). A
 * real "yes yes yes yes yes yes yes yes" never gets near its budget; a
 * genuine runaway decode always does. Past that gate the tells are: one
 * word repeated LOOP_RUN_LEN times in a row, one word making up
 * LOOP_DOMINANCE of a LOOP_MIN_WORDS+ output, or a 2-3 word phrase
 * repeated back to back LOOP_NGRAM_REPEATS times. A false positive is NOT
 * benign — it demotes webgpu and tears down a healthy pipeline for a full
 * wasm rebuild — which is why the near-cap gate comes first.
 */
const LOOP_NEAR_CAP = 0.6;
const LOOP_MIN_WORDS = 30;
const LOOP_RUN_LEN = 8;
const LOOP_DOMINANCE = 0.6;
const LOOP_NGRAM_MAX = 3;
const LOOP_NGRAM_REPEATS = 4;

/**
 * Load-time canary validation, the primary defense: some GPUs build the
 * pipeline, survive warm-up, and then decode garbage without ever throwing
 * (observed on Apple GPUs with q8 weights). Known speech is transcribed
 * once at load; if none of these anchor words come back, the webgpu build
 * is silently broken and the worker rebuilds on wasm before any real
 * utterance is touched. The canary file is vendored; if it is missing the
 * check is skipped and only the runtime loop guard applies.
 */
// Public-domain speech (JFK inaugural, 1961 — a US government work), trimmed
// to 4.6s. Verified transcript on this model: "And so, my fellow Americans,
// ask not."
const CANARY_URL = new URL('../vendor/canary.wav', import.meta.url);
const CANARY_ANCHORS = ['fellow', 'americans', 'ask'];
const CANARY_MIN_HITS = 2;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let pipe = null;          // the transformers.js ASR pipeline, once built
let device = null;        // 'webgpu' | 'wasm'
let loaded = false;       // true once load + warm-up completed

// Streamer classes captured from the vendored bundle when the pipeline is
// built. Either may be missing from a future bundle; partials then degrade
// (Whisper -> plain -> none) without touching transcription itself.
let WhisperTextStreamerClass = null;
let TextStreamerClass = null;

// Model files load in parallel; aggregate their progress into one percentage.
const fileProgress = new Map();
let lastPct = -1;

// Jobs are chained so they run strictly FIFO even if several messages arrive
// while an earlier job is still in flight.
let chain = Promise.resolve();

// ---------------------------------------------------------------------------
// Messaging helpers
// ---------------------------------------------------------------------------

function postStatus(status, detail = {}) {
  // model rides on every status so the main thread always knows which
  // checkpoint this worker runs (SottoASR persists device verdicts per model).
  self.postMessage({ type: 'status', payload: { status, model: MODEL_ID, ...detail } });
}

function postResult(id, payload) {
  self.postMessage({ id, type: 'result', payload });
}

function postError(id, err) {
  const message = err && err.message ? String(err.message) : String(err);
  self.postMessage({ id, type: 'error', payload: { message } });
}

// ---------------------------------------------------------------------------
// Pipeline construction
// ---------------------------------------------------------------------------

function onProgress(p) {
  if (!p || p.status !== 'progress' || !p.file || !p.total) return;
  fileProgress.set(p.file, { loaded: p.loaded || 0, total: p.total });
  let loadedBytes = 0;
  let totalBytes = 0;
  for (const f of fileProgress.values()) {
    loadedBytes += f.loaded;
    totalBytes += f.total;
  }
  if (totalBytes <= 0) return;
  const pct = Math.min(100, Math.round((loadedBytes / totalBytes) * 100));
  if (pct !== lastPct) {
    lastPct = pct;
    postStatus('loading-model', { progress: pct });
  }
}

/**
 * Build the ASR pipeline on the given device. All paths are import.meta.url
 * relative so vendor/ resolves under both / and /sotto/ deploys.
 */
async function buildPipeline(dev) {
  const mod = await import(
    new URL('../vendor/transformers/transformers.min.js', import.meta.url).href
  );
  const { pipeline, env } = mod;
  WhisperTextStreamerClass = mod.WhisperTextStreamer || null;
  TextStreamerClass = mod.TextStreamer || null;
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = new URL('../vendor/whisper/', import.meta.url).href;
  env.backends.onnx.wasm.wasmPaths = new URL('../vendor/transformers/', import.meta.url).href;
  return pipeline('automatic-speech-recognition', MODEL_ID, {
    dtype: 'q8',
    device: dev,
    progress_callback: onProgress,
  });
}

/**
 * Parse a 16-bit PCM mono 16 kHz WAV into a Float32Array. Returns null for
 * anything else — the canary check is then simply skipped.
 */
function parseWav16kMono(buf) {
  const dv = new DataView(buf);
  if (dv.byteLength < 44) return null;
  const tag = (o) => String.fromCharCode(
    dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let off = 12;
  let fmtOk = false;
  let dataOff = -1;
  let dataLen = 0;
  while (off + 8 <= dv.byteLength) {
    const id = tag(off);
    const len = dv.getUint32(off + 4, true);
    if (id === 'fmt ' && off + 8 + 16 <= dv.byteLength) {
      const audioFormat = dv.getUint16(off + 8, true);
      const channels = dv.getUint16(off + 10, true);
      const sampleRate = dv.getUint32(off + 12, true);
      const bits = dv.getUint16(off + 22, true);
      fmtOk = audioFormat === 1 && channels === 1 && sampleRate === 16000 && bits === 16;
    } else if (id === 'data') {
      dataOff = off + 8;
      dataLen = Math.min(len, dv.byteLength - dataOff);
    }
    off += 8 + len + (len % 2);
  }
  if (!fmtOk || dataOff < 0 || dataLen < 2) return null;
  const n = dataLen >> 1;
  const pcm = new Float32Array(n);
  for (let i = 0; i < n; i++) pcm[i] = dv.getInt16(dataOff + 2 * i, true) / 32768;
  return pcm;
}

/** Fetch the vendored canary clip, or null when unavailable (e.g. offline). */
async function fetchCanary() {
  try {
    const resp = await fetch(CANARY_URL);
    if (!resp.ok) return null;
    return parseWav16kMono(await resp.arrayBuffer());
  } catch {
    return null;
  }
}

/** True when a canary transcription contains enough known anchor words. */
function canaryLooksSane(text) {
  const lower = (text || '').toLowerCase();
  let hits = 0;
  for (const anchor of CANARY_ANCHORS) {
    if (lower.includes(anchor)) hits += 1;
  }
  return hits >= CANARY_MIN_HITS;
}

/** Tear down whatever pipeline exists and rebuild on wasm. */
async function rebuildOnWasm() {
  if (pipe) {
    try { await pipe.dispose?.(); } catch { /* already broken; ignore */ }
  }
  pipe = null;
  postStatus('loading-model', {});
  pipe = await buildPipeline('wasm');
  device = 'wasm';
}

/**
 * Load the model and run a warm-up inference. Tries webgpu first when
 * available; falls back to wasm if the pipeline fails to build, throws on
 * its first inference, or fails the canary validation (some GPUs only fail
 * late, and some fail silently).
 * @param {'auto'|'wasm'} [preferred] 'wasm' skips webgpu entirely
 */
async function doLoad(preferred = 'auto') {
  if (loaded) return;
  postStatus('loading-model', {});
  try {
    const wantGpu = preferred !== 'wasm'
      && typeof navigator !== 'undefined' && !!navigator.gpu;
    if (wantGpu) {
      try {
        pipe = await buildPipeline('webgpu');
        device = 'webgpu';
      } catch {
        pipe = null; // build-time GPU failure; fall through to wasm
      }
    }
    if (!pipe) {
      pipe = await buildPipeline('wasm');
      device = 'wasm';
    }

    const warmOpts = { max_new_tokens: MIN_NEW_TOKENS_CAP };
    postStatus('warming', { device });
    if (device === 'webgpu') {
      // Warm up AND validate: transcribe the vendored canary clip. A GPU can
      // build the pipeline, then fail at first inference (throw) or fail
      // silently (decode garbage). Either way: rebuild on wasm.
      let gpuOk = false;
      try {
        const canary = await fetchCanary();
        if (canary) {
          const { text } = await pipe(canary, { max_new_tokens: decodeBudget(canary) });
          gpuOk = canaryLooksSane(text);
        } else {
          // Canary unavailable (offline, asset swapped): warm on zeros and
          // rely on the runtime decode-loop guard instead.
          await pipe(new Float32Array(WARMUP_SAMPLES), warmOpts);
          gpuOk = true;
        }
      } catch {
        gpuOk = false;
      }
      if (!gpuOk) {
        await rebuildOnWasm();
        postStatus('warming', { device });
        await pipe(new Float32Array(WARMUP_SAMPLES), warmOpts);
      }
    } else {
      await pipe(new Float32Array(WARMUP_SAMPLES), warmOpts);
    }

    loaded = true;
    postStatus('ready', { device });
  } catch (err) {
    if (pipe) {
      try { await pipe.dispose?.(); } catch { /* ignore */ }
    }
    pipe = null;
    device = null;
    postStatus('error', { message: err && err.message ? String(err.message) : String(err) });
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

function rmsOf(audio) {
  const n = audio.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += audio[i] * audio[i];
  return Math.sqrt(sum / n);
}

/** Lowercase and strip everything that is not a letter or digit. */
function normalizeText(text) {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
}

/**
 * True when the text looks like a runaway decode loop for a job that was
 * allowed maxNewTokens new tokens. See the LOOP_* constants for the
 * rationale; the near-cap gate keeps legitimate repetition ("yes yes yes")
 * from tearing down a healthy webgpu pipeline.
 * @param {string} text decoder output
 * @param {number} maxNewTokens the max_new_tokens the job ran with
 */
function looksLikeDecodeLoop(text, maxNewTokens) {
  const words = text.split(/\s+/).filter(Boolean);
  if (!Number.isFinite(maxNewTokens) || words.length < maxNewTokens * LOOP_NEAR_CAP) {
    return false;
  }
  if (words.length < LOOP_RUN_LEN) return false;

  // Tell 1: one word repeated LOOP_RUN_LEN times in a row.
  let run = 1;
  const counts = new Map();
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    counts.set(w, (counts.get(w) || 0) + 1);
    if (i > 0 && w === words[i - 1]) {
      run += 1;
      if (run >= LOOP_RUN_LEN) return true;
    } else {
      run = 1;
    }
  }

  // Tell 2: one word dominating a long output.
  if (words.length >= LOOP_MIN_WORDS) {
    let max = 0;
    for (const c of counts.values()) if (c > max) max = c;
    if (max / words.length >= LOOP_DOMINANCE) return true;
  }

  // Tell 3: a short phrase looped whole ("Thank you. Thank you. ..."):
  // any 2- or 3-word window repeated back to back LOOP_NGRAM_REPEATS
  // times. Every phase is scanned so the loop is caught wherever it
  // starts in the sentence.
  for (let n = 2; n <= LOOP_NGRAM_MAX; n++) {
    for (let phase = 0; phase < n; phase++) {
      let repeats = 1;
      for (let i = phase; i + 2 * n <= words.length; i += n) {
        let same = true;
        for (let k = 0; k < n; k++) {
          if (words[i + k] !== words[i + n + k]) { same = false; break; }
        }
        repeats = same ? repeats + 1 : 1;
        if (repeats >= LOOP_NGRAM_REPEATS) return true;
      }
    }
  }
  return false;
}

function decodeBudget(audio) {
  const seconds = audio.length / 16000;
  return Math.max(MIN_NEW_TOKENS_CAP, Math.ceil(seconds * TOKENS_PER_SECOND_CAP));
}

/**
 * Build a streamer that posts {id, type: 'partial'} messages carrying the
 * ACCUMULATED decoder text, at most one message per PARTIAL_MIN_INTERVAL_MS
 * (accumulation continues between posts; nothing is lost to the throttle).
 * Prefers WhisperTextStreamer, whose put() consumes Whisper's timestamp
 * tokens so they never leak into the text; falls back to a plain
 * TextStreamer. Returns null when neither class is available or
 * construction throws — partials are an enhancement, never a dependency,
 * and the job then runs exactly as it did before v0.3. Only real
 * transcribe jobs get one: the canary and warm-up runs never stream.
 * @param {number} id job id stamped on each partial message
 * @returns {object|null} a streamer for the pipeline call, or null
 */
function makePartialStreamer(id) {
  let text = '';
  let lastSentAt = 0;
  const callback_function = (delta) => {
    if (typeof delta !== 'string' || delta === '') return;
    text += delta;
    const now = Date.now();
    if (now - lastSentAt < PARTIAL_MIN_INTERVAL_MS) return;
    lastSentAt = now;
    self.postMessage({ id, type: 'partial', payload: { text } });
  };
  try {
    if (WhisperTextStreamerClass) {
      return new WhisperTextStreamerClass(pipe.tokenizer, { callback_function });
    }
    if (TextStreamerClass) {
      return new TextStreamerClass(pipe.tokenizer, { skip_prompt: true, callback_function });
    }
  } catch { /* construction failed: run without partials */ }
  return null;
}

async function doTranscribe(audio, id) {
  if (!loaded || !pipe) throw new Error('model not loaded');
  if (audio.length === 0) return '';

  const rms = rmsOf(audio);
  const opts = { max_new_tokens: decodeBudget(audio) };
  // Partials stream while the decoder runs; the loop guard and silence
  // suppression below judge ONLY the final text.
  const streamer = makePartialStreamer(id);

  let text;
  let gpuFailedLate = false;
  try {
    ({ text } = await pipe(audio, streamer ? { ...opts, streamer } : opts));
    // Some GPUs fail silently: no exception, but broken q8 kernels make the
    // decoder loop garbage. Only judge real speech — silence legitimately
    // produces repetitive hallucinations on healthy hardware.
    if (device === 'webgpu' && rms >= SILENCE_RMS
        && looksLikeDecodeLoop(text || '', opts.max_new_tokens)) {
      gpuFailedLate = true;
    }
  } catch (err) {
    if (device !== 'webgpu') throw err;
    gpuFailedLate = true;
  }

  if (gpuFailedLate) {
    // A GPU that survived warm-up can still fail on a real buffer (thrown
    // error or silent garbage). Rebuild on wasm and retry exactly once,
    // with a FRESH streamer: the new pipeline has a new tokenizer, and the
    // failed attempt's garbage must not prefix the retry's partials.
    await rebuildOnWasm();
    const retryStreamer = makePartialStreamer(id);
    ({ text } = await pipe(audio, retryStreamer ? { ...opts, streamer: retryStreamer } : opts));
    postStatus('ready', { device });
  }

  text = (text || '').trim();
  if (rms < SILENCE_RMS && SILENCE_HALLUCINATIONS.has(normalizeText(text))) {
    return '';
  }
  return text;
}

// ---------------------------------------------------------------------------
// Message handling (strict FIFO via promise chain)
// ---------------------------------------------------------------------------

async function handle(msg) {
  const { id, type, payload } = msg || {};
  try {
    if (type === 'load') {
      await doLoad(payload && payload.device === 'wasm' ? 'wasm' : 'auto');
      postResult(id, { device, model: MODEL_ID });
    } else if (type === 'transcribe') {
      const audio = new Float32Array(payload.buffer, payload.byteOffset, payload.length);
      const text = await doTranscribe(audio, id);
      postResult(id, { text, device });
    } else if (type === 'dispose') {
      if (pipe) {
        try { await pipe.dispose?.(); } catch { /* ignore */ }
      }
      pipe = null;
      loaded = false;
      postResult(id, {});
      self.close();
    } else {
      throw new Error(`unknown message type: ${String(type)}`);
    }
  } catch (err) {
    postError(id, err);
  }
}

self.onmessage = (event) => {
  const msg = event.data;
  chain = chain.then(() => handle(msg));
};
