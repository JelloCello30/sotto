/*
 * Sotto — ASR wrapper (js/asr.js)
 *
 * SottoASR runs the vendored Whisper model in a module worker
 * (js/asr-worker.js) so inference never blocks the UI thread. Fully
 * on-device: no network requests, nothing uploaded. The only thing
 * persisted is the device-verdict record described below.
 *
 * Exports:
 *   SottoASR (default + named) — the wrapper class per SPEC-V2.md/SPEC-V3.md
 *   SottoASRError              — typed error with .code
 *   WASM_TIMEOUT_MS            — default transcribe timeout on the wasm path
 *
 * Usage:
 *   const asr = new SottoASR({ onStatus });
 *   await asr.load();                                   // idempotent
 *   const text = await asr.transcribe(float32, {
 *     timeoutMs: 30000,
 *     onPartial: (text) => { ... },  // accumulated partial transcription
 *   });
 *   asr.dispose();
 *
 * Device selection is automatic: webgpu when available, wasm otherwise,
 * with fallback to wasm when the GPU fails at load, stalls at load (shader
 * compiles that hang for minutes without erroring), fails at first
 * inference, or fails silently (broken q8 kernels that decode garbage
 * loops). Pass
 * { device: 'wasm' } to the constructor to skip webgpu entirely — useful
 * for benchmarking and for machines with known-bad GPU drivers.
 *
 * Device-verdict cache: when an 'auto' load ends demoted to wasm, the
 * verdict is written to localStorage ('sotto.asr.verdict.v1') and later
 * 'auto' loads pass 'wasm' straight to the worker — skipping the doomed
 * webgpu attempt and its canary, which is most of a ~23 s cold load.
 * Honored only for the same model id and for 14 days (GPU drivers change;
 * re-probe occasionally). Where localStorage is unavailable or blocked
 * (private mode), the cache silently does nothing.
 *
 * onStatus(status, detail) statuses (every status carries detail.model,
 * the worker's model id):
 *   'loading-model'  detail.progress is 0-100 when transformers.js reports
 *                    it; detail.message is set when a stalled webgpu load
 *                    is being retried on wasm
 *   'warming'        warm-up inference in progress (detail.device set)
 *   'ready'          model loaded and warm (detail.device: 'webgpu' | 'wasm')
 *   'error'          load or worker failure (detail.message)
 *
 * NOTE for integrators: the Float32Array passed to transcribe() is
 * TRANSFERRED to the worker (zero-copy) and becomes unusable (detached)
 * afterwards. SottoMic.slice() returns a fresh array per call, so this is
 * only a concern if you reuse buffers.
 */

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** Default per-job timeout for transcribe() when inference runs on webgpu. */
const DEFAULT_TIMEOUT_MS = 30000;

/**
 * Default per-job timeout when inference runs on wasm. GitHub Pages sends
 * no COOP/COEP headers, so the page is never crossOriginIsolated, there is
 * no SharedArrayBuffer, and ONNX runs its wasm backend on a single thread —
 * several times slower than webgpu on the same machine. Applied
 * automatically at dispatch time when the loaded device is wasm; an
 * explicit timeoutMs from the caller always wins.
 */
export const WASM_TIMEOUT_MS = 60000;

/**
 * Watchdog for one load attempt. A webgpu shader compile can hang for
 * minutes without ever erroring (observed in the wild); without a timer
 * load() would never settle. On expiry the worker is terminated and, if
 * the attempt allowed webgpu, the load is retried once on wasm.
 */
const LOAD_TIMEOUT_MS = 120000;

// ---------------------------------------------------------------------------
// Device-verdict cache
// ---------------------------------------------------------------------------

/** localStorage key for the persisted device verdict. */
const VERDICT_KEY = 'sotto.asr.verdict.v1';

/**
 * How long a wasm-demotion verdict is trusted. GPU drivers and browser
 * webgpu backends do improve; after this the webgpu path is probed again.
 */
const VERDICT_TTL_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Read the stored verdict: { model, device: 'wasm', at } or null when
 * absent, malformed, or expired (expired and malformed entries are also
 * removed). Every storage failure — no localStorage, private mode,
 * blocked access — reads as null: the cache is an optimization, never a
 * dependency.
 * @returns {{ model: string, device: 'wasm', at: number } | null}
 */
function readVerdict() {
  try {
    const raw = localStorage.getItem(VERDICT_KEY);
    if (!raw) return null;
    let v = null;
    try { v = JSON.parse(raw); } catch { /* malformed: cleared below */ }
    if (v && v.device === 'wasm' && typeof v.model === 'string' && v.model !== ''
        && Number.isFinite(v.at) && Date.now() - v.at <= VERDICT_TTL_MS) {
      return v;
    }
    localStorage.removeItem(VERDICT_KEY);
  } catch { /* no usable storage: behave as uncached */ }
  return null;
}

/** Persist a wasm-demotion verdict for the given model id. Silent no-op without storage. */
function writeVerdict(model) {
  try {
    localStorage.setItem(VERDICT_KEY, JSON.stringify({ model, device: 'wasm', at: Date.now() }));
  } catch { /* no usable storage: skip the cache */ }
}

/** Remove the stored verdict. Silent no-op without storage. */
function clearVerdict() {
  try {
    localStorage.removeItem(VERDICT_KEY);
  } catch { /* no usable storage: nothing to clear */ }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Typed error thrown/rejected by SottoASR.
 * code: 'bad-input' | 'timeout' | 'disposed' | 'load-failed' | 'transcribe-failed'
 *       | 'worker-error'
 */
export class SottoASRError extends Error {
  /**
   * @param {string} code machine-readable error code
   * @param {string} message human-readable detail
   */
  constructor(code, message) {
    super(message);
    this.name = 'SottoASRError';
    this.code = code;
  }
}

// ---------------------------------------------------------------------------
// SottoASR
// ---------------------------------------------------------------------------

export class SottoASR {
  /**
   * @param {{ onStatus?: (status: string, detail?: object) => void,
   *           device?: 'auto' | 'wasm' }} [opts]
   */
  constructor({ onStatus, device } = {}) {
    this._onStatus = typeof onStatus === 'function' ? onStatus : null;
    this._preferredDevice = device === 'wasm' ? 'wasm' : 'auto';

    this._worker = null;
    this._loadPromise = null;
    this._ready = false;
    this._device = null;      // 'webgpu' | 'wasm' | null
    this._model = null;       // model id reported by the worker, e.g. 'base'
    this._disposed = false;

    this._nextId = 1;
    this._pending = new Map(); // id -> {resolve, reject, onPartial?, timer?, stale}
    this._queue = [];          // transcribe jobs waiting to be sent
    this._inflightId = null;   // id of the transcribe job the worker holds
  }

  /** Inference device: 'webgpu' | 'wasm'; null before load and after a stalled worker is torn down. */
  get device() {
    return this._device;
  }

  /** True once load() has resolved (and dispose() has not been called). */
  get ready() {
    return this._ready;
  }

  /** Model id reported by the worker (e.g. 'base'); null until the worker has said. */
  get model() {
    return this._model;
  }

  /**
   * Load the model in the worker and warm it up. Idempotent: repeated calls
   * return the same promise while loading and a resolved one once ready.
   * Each attempt runs under a LOAD_TIMEOUT_MS watchdog — a stalled webgpu
   * load is retried once on wasm — so load() always settles. A failed load
   * clears state so a later call can retry. Rejects with a SottoASRError
   * carrying the failure message.
   * @returns {Promise<void>}
   */
  load() {
    if (this._disposed) {
      // A fresh load after dispose() builds a fresh worker.
      this._disposed = false;
    }
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = this._doLoad().catch((err) => {
      this._loadPromise = null;
      throw err;
    });
    return this._loadPromise;
  }

  /**
   * Transcribe mono 16 kHz PCM. Jobs are queued FIFO; each has its own
   * timeout, defaulting to DEFAULT_TIMEOUT_MS on webgpu and WASM_TIMEOUT_MS
   * on wasm (single-threaded wasm needs the headroom — see the constant).
   * Resolves with the transcription, or '' for silence/no-speech
   * (including suppressed Whisper silence hallucinations).
   * While the worker decodes, onPartial (when given) is called with the
   * ACCUMULATED partial text so far, throttled by the worker to at most
   * one call per 120 ms. Partials are an enhancement: they may never
   * arrive at all (no streamer available), exceptions thrown by the
   * callback are swallowed, and partials for stale (timed-out) jobs are
   * dropped. The resolved final text is the authority.
   * The input array is transferred, not copied — see module header.
   * @param {Float32Array} audio mono PCM at 16000 Hz
   * @param {{ timeoutMs?: number, onPartial?: (text: string) => void }} [opts]
   *   an explicit timeoutMs overrides the per-device default; 0 or a
   *   non-finite value disables the timeout
   * @returns {Promise<string>}
   */
  async transcribe(audio, { timeoutMs, onPartial } = {}) {
    if (!(audio instanceof Float32Array)) {
      throw new SottoASRError('bad-input', 'transcribe() expects a Float32Array of 16 kHz mono PCM');
    }
    if (audio.length === 0) {
      // Covers both genuinely empty input and an already-transferred
      // (detached) buffer, which reports length 0. Nothing to transcribe.
      return '';
    }
    await this.load();
    if (this._disposed) {
      throw new SottoASRError('disposed', 'SottoASR was disposed');
    }
    return new Promise((resolve, reject) => {
      this._queue.push({
        audio,
        timeoutMs,
        onPartial: typeof onPartial === 'function' ? onPartial : null,
        resolve,
        reject,
      });
      this._pump();
    });
  }

  /**
   * Terminate the worker immediately and reject anything queued or in
   * flight. Safe to call repeatedly. load() afterwards starts fresh.
   */
  dispose() {
    if (this._disposed && !this._worker) return;
    this._disposed = true;

    if (this._worker) {
      try {
        this._worker.postMessage({ id: this._nextId++, type: 'dispose' });
      } catch { /* worker may already be dead */ }
      this._worker.terminate();
      this._worker = null;
    }

    const err = new SottoASRError('disposed', 'SottoASR was disposed');
    for (const entry of this._pending.values()) {
      if (entry.timer !== undefined) clearTimeout(entry.timer);
      if (!entry.stale) entry.reject(err);
    }
    this._pending.clear();
    for (const job of this._queue) job.reject(err);
    this._queue.length = 0;
    this._inflightId = null;

    this._loadPromise = null;
    this._ready = false;
    this._device = null;
    this._model = null;
  }

  // -- internals ------------------------------------------------------------

  _emitStatus(status, detail) {
    if (!this._onStatus) return;
    try {
      this._onStatus(status, detail);
    } catch { /* never let a listener break the pipeline */ }
  }

  _ensureWorker() {
    if (this._worker) return;
    this._worker = new Worker(new URL('asr-worker.js', import.meta.url), { type: 'module' });
    this._worker.onmessage = (event) => this._onWorkerMessage(event.data);
    this._worker.onerror = (event) => {
      const message = event && event.message ? event.message : 'worker failed';
      // The worker is unusable; tear it down so the next load() rebuilds it.
      this._worker.terminate();
      this._worker = null;
      this._failEverything(new SottoASRError('worker-error', message));
      this._emitStatus('error', { message });
    };
  }

  async _doLoad() {
    // An 'auto' load honors a fresh stored wasm verdict: the doomed webgpu
    // attempt and its canary are skipped and the worker goes straight to
    // wasm. A verdict-driven wasm load behaves exactly like an explicit
    // one, including no wasm-on-wasm retry after a stall.
    let requested = this._preferredDevice;
    let cached = null;
    if (requested === 'auto') {
      cached = readVerdict();
      if (cached) requested = 'wasm';
    }
    try {
      await this._loadAttempt(requested);
      this._settleVerdict(cached, requested);
      return;
    } catch (err) {
      if (!this._isLoadStall(err) || requested === 'wasm') {
        throw this._terminalLoadError(err);
      }
    }
    // The webgpu attempt stalled. Rebuild a fresh worker and retry exactly
    // once, skipping the GPU entirely.
    this._emitStatus('loading-model', { message: 'GPU path stalled, retrying on CPU' });
    try {
      await this._loadAttempt('wasm');
    } catch (err) {
      throw this._terminalLoadError(err);
    }
    this._settleVerdict(cached, requested);
  }

  /**
   * Post-load verdict bookkeeping (every path is a silent no-op without
   * localStorage). A verdict that was honored is validated against the
   * model the worker just reported: a mismatch means the shipped model
   * changed since it was written, so it is cleared and the NEXT 'auto'
   * load probes webgpu again — this one already ran wasm; one conservative
   * session is the price of not duplicating MODEL_ID outside the worker.
   * A fresh 'auto' probe that ended demoted to wasm (webgpu failed the
   * canary, threw, or stalled) writes the verdict so later sessions skip
   * webgpu — but only when webgpu was actually on the table
   * (navigator.gpu present).
   * @param {{ model: string } | null} cached verdict honored by this load
   * @param {'auto' | 'wasm'} requested device actually asked of the worker
   */
  _settleVerdict(cached, requested) {
    if (cached) {
      if (this._model && cached.model !== this._model) clearVerdict();
      return;
    }
    if (requested !== 'auto') return;    // explicit wasm: nothing was probed
    if (this._device !== 'wasm') return; // webgpu won: no demotion to record
    if (typeof navigator === 'undefined' || !navigator.gpu) return; // webgpu never attempted
    if (this._model) writeVerdict(this._model);
  }

  /**
   * One load attempt against the worker (building it if needed), guarded
   * by the LOAD_TIMEOUT_MS watchdog. On expiry the pending entry is retired
   * and the worker terminated — a stalled shader compile never yields, so
   * the worker cannot be salvaged — and the attempt rejects with the
   * internal 'load-stalled' code so _doLoad can decide whether a wasm
   * retry is due.
   * @param {'auto'|'wasm'} device
   * @returns {Promise<void>} resolves once the model is loaded and warm
   */
  async _loadAttempt(device) {
    this._ensureWorker();
    const id = this._nextId++;
    const result = new Promise((resolve, reject) => {
      const entry = { resolve, reject, stale: false, code: 'load-failed', timer: undefined };
      entry.timer = setTimeout(() => {
        entry.stale = true;
        entry.timer = undefined;
        this._pending.delete(id);
        if (this._worker) {
          this._worker.terminate();
          this._worker = null;
        }
        reject(new SottoASRError('load-stalled', `model load exceeded ${LOAD_TIMEOUT_MS} ms`));
      }, LOAD_TIMEOUT_MS);
      this._pending.set(id, entry);
    });
    this._worker.postMessage({ id, type: 'load', payload: { device } });
    try {
      const payload = await result;
      this._device = payload.device || this._device;
      if (typeof payload.model === 'string' && payload.model !== '') this._model = payload.model;
      this._ready = true;
    } catch (err) {
      if (err instanceof SottoASRError) throw err;
      throw new SottoASRError('load-failed', err && err.message ? err.message : String(err));
    }
  }

  /** True for the internal watchdog rejection produced by _loadAttempt. */
  _isLoadStall(err) {
    return err instanceof SottoASRError && err.code === 'load-stalled';
  }

  /**
   * Normalize a terminal load failure. The internal 'load-stalled' code
   * never escapes: callers see 'load-failed'. Watchdog expiries are also
   * the one path that emits an 'error' status from here — on every other
   * failure the worker posted one itself before its error reply.
   */
  _terminalLoadError(err) {
    if (!this._isLoadStall(err)) {
      if (err instanceof SottoASRError) return err;
      return new SottoASRError('load-failed', err && err.message ? err.message : String(err));
    }
    const out = new SottoASRError('load-failed', err.message);
    this._emitStatus('error', { message: out.message });
    return out;
  }

  _onWorkerMessage(msg) {
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'status') {
      const { status, ...detail } = msg.payload || {};
      if (detail.device) this._device = detail.device;
      if (typeof detail.model === 'string' && detail.model !== '') this._model = detail.model;
      this._emitStatus(status, detail);
      return;
    }

    if (msg.type === 'partial') {
      // Streaming partial for an in-flight transcribe job. Dropped when
      // the job is unknown or stale (timed out), or asked for no partials.
      const entry = this._pending.get(msg.id);
      if (!entry || entry.stale || !entry.onPartial) return;
      const text = msg.payload && typeof msg.payload.text === 'string' ? msg.payload.text : '';
      try {
        entry.onPartial(text);
      } catch { /* never let a listener break the pipeline */ }
      return;
    }

    const entry = this._pending.get(msg.id);
    if (!entry) return;
    this._pending.delete(msg.id);
    if (entry.timer !== undefined) clearTimeout(entry.timer);

    if (!entry.stale) {
      if (msg.type === 'result') {
        entry.resolve(msg.payload || {});
      } else if (msg.type === 'error') {
        const message = (msg.payload && msg.payload.message) || 'inference failed';
        entry.reject(new SottoASRError(entry.code || 'transcribe-failed', message));
      }
    }

    if (msg.id === this._inflightId) {
      this._inflightId = null;
      this._pump();
    }
  }

  /** Send the next queued transcribe job if the worker is free. */
  _pump() {
    if (this._inflightId !== null || this._queue.length === 0) return;
    if (this._disposed || !this._worker) return;

    const job = this._queue.shift();
    const id = this._nextId++;
    this._inflightId = id;

    // Resolve the timeout at dispatch time, once the device is known: wasm
    // is single-threaded here (no COOP/COEP on GitHub Pages) and needs more
    // headroom than webgpu. An explicit caller timeout always wins.
    const timeoutMs = job.timeoutMs !== undefined
      ? job.timeoutMs
      : (this._device === 'wasm' ? WASM_TIMEOUT_MS : DEFAULT_TIMEOUT_MS);

    const entry = {
      resolve: (payload) => job.resolve(typeof payload.text === 'string' ? payload.text : ''),
      reject: job.reject,
      onPartial: job.onPartial,
      stale: false,
      timer: undefined,
    };
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      entry.timer = setTimeout(() => {
        // The worker cannot abort mid-inference, and a job that outlives
        // its timeout has in practice wedged the backend (observed: webgpu
        // hangs that never return). Queued jobs' timers only start at
        // dispatch, so leaving the worker alive would strand them behind
        // this one forever. Reject the caller, kill the worker, and fail
        // everything else fast — the app retries per job, and the next
        // transcribe() (or load()) rebuilds a fresh worker.
        entry.stale = true;
        entry.timer = undefined;
        job.reject(new SottoASRError('timeout', `transcription exceeded ${timeoutMs} ms`));
        if (this._worker) {
          this._worker.terminate();
          this._worker = null;
        }
        this._device = null;
        const message = 'transcriber stalled and will reload on the next request';
        this._failEverything(new SottoASRError('worker-error', message));
        this._emitStatus('error', { message });
      }, timeoutMs);
    }
    this._pending.set(id, entry);

    const { audio } = job;
    this._worker.postMessage(
      {
        id,
        type: 'transcribe',
        payload: { buffer: audio.buffer, byteOffset: audio.byteOffset, length: audio.length },
      },
      [audio.buffer],
    );
  }

  _failEverything(err) {
    for (const entry of this._pending.values()) {
      if (entry.timer !== undefined) clearTimeout(entry.timer);
      if (!entry.stale) entry.reject(err);
    }
    this._pending.clear();
    for (const job of this._queue) job.reject(err);
    this._queue.length = 0;
    this._inflightId = null;
    this._loadPromise = null;
    this._ready = false;
  }
}

export default SottoASR;
