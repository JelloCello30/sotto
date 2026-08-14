/*
 * Sotto — microphone capture (js/audio.js)
 *
 * Raw-PCM microphone capture for Whisper mode, per SPEC-V2.md. An
 * AudioWorklet (js/audio-worklet.js) streams Float32 samples to the main
 * thread, which writes them into a preallocated 30-second ring buffer
 * indexed by AudioContext frame number. The frame timeline is aligned to
 * performance.now() so slice(t0Ms, t1Ms) can map the engine's lip-motion
 * segment times (also performance.now()-based) onto the matching audio
 * samples — correctly even after the ring wraps.
 *
 * Alignment approach: every chunk arrives tagged with the frame index of its
 * first sample. Each arrival yields a candidate epoch ("wall time of context
 * frame 0"); message latency only ever makes candidates later, so the running
 * minimum converges on the tightest estimate within a few chunks. A slow
 * upward nudge tracks clock drift, and a large persistent jump (context was
 * suspended, so frames stalled while the wall clock ran) snaps the epoch
 * forward; transient main-thread jank self-heals through the minimum.
 * Residual error is a few milliseconds — small next to the ±250 ms padding
 * the fusion layer adds around each segment.
 *
 * Fully on-device: no MediaRecorder, no compression, nothing persisted,
 * nothing transmitted. stop() releases the microphone tracks and closes the
 * AudioContext. Constraints per spec: audio only (video: false),
 * echoCancellation: false, noiseSuppression: false, autoGainControl: true.
 *
 * Exports:
 *   SottoMic (default + named)  — the capture class per SPEC-V2.md
 *   SottoMicError               — typed error with .code
 *   TARGET_RATE, RING_SECONDS   — fixed capture parameters, for UI display
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Sample rate of everything slice() returns, per SPEC-V2.md. */
export const TARGET_RATE = 16000;

/** Ring buffer capacity in seconds. */
export const RING_SECONDS = 30;

/** Time constant (seconds) for the smoothed RMS level meter. */
const LEVEL_TAU_S = 0.15;

/** Per-chunk pull rate for tracking slow clock drift between audio and wall clock. */
const EPOCH_DRIFT_ALPHA = 0.001;

/**
 * A candidate epoch this far above the estimate means the context was
 * suspended (frames stalled while the wall clock ran): snap forward. If it
 * was only main-thread jank, later candidates pull the minimum back down.
 */
const EPOCH_JUMP_MS = 1500;

/**
 * Worklet module URL, resolved against this module so the site works at both
 * http://localhost:4173/ and a /sotto/ subpath. Never a root-absolute path.
 */
const WORKLET_URL = new URL('audio-worklet.js', import.meta.url);

/** Sentinel thrown internally when stop() interrupts a pending start(). */
const ABORTED = Symbol('sotto-mic-start-aborted');

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Typed error thrown by SottoMic. `code` is machine-readable:
 *   'unsupported' — no getUserMedia (insecure context) or no AudioWorklet
 *   'mic-denied'  — the user or a policy denied microphone permission
 *   'mic-none'    — no microphone exists on this device
 *   'mic-failed'  — the device or audio graph failed for another reason
 */
export class SottoMicError extends Error {
  /**
   * @param {string} code machine-readable error code
   * @param {string} message human-readable detail
   * @param {unknown} [cause] the underlying error, if any
   */
  constructor(code, message, cause) {
    super(message);
    this.name = 'SottoMicError';
    this.code = code;
    if (cause !== undefined) this.cause = cause;
  }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/**
 * Linear resample of a mono Float32 buffer between sample rates. Output
 * sample j is interpolated at input position j * fromRate / toRate, so time
 * spacing stays uniform; the tail clamps to the last input sample.
 * @param {Float32Array} input source samples
 * @param {number} fromRate source sample rate in Hz
 * @param {number} toRate target sample rate in Hz
 * @returns {Float32Array} resampled copy
 */
function resampleLinear(input, fromRate, toRate) {
  const outLen = Math.round((input.length * toRate) / fromRate);
  if (outLen <= 0) return new Float32Array(0);
  const out = new Float32Array(outLen);
  const step = fromRate / toRate;
  const last = input.length - 1;
  for (let j = 0; j < outLen; j++) {
    const x = j * step;
    const i0 = Math.floor(x);
    if (i0 >= last) {
      out[j] = input[last];
    } else {
      const frac = x - i0;
      out[j] = input[i0] + (input[i0 + 1] - input[i0]) * frac;
    }
  }
  return out;
}

/** Throw SottoMicError 'unsupported' unless this environment can capture. */
function assertSupported() {
  if (
    typeof navigator === 'undefined' ||
    !navigator.mediaDevices ||
    typeof navigator.mediaDevices.getUserMedia !== 'function'
  ) {
    throw new SottoMicError(
      'unsupported',
      'Microphone capture needs navigator.mediaDevices.getUserMedia, which browsers only provide in a secure context (https, or localhost).',
    );
  }
  if (
    typeof AudioContext !== 'function' ||
    typeof AudioWorkletNode !== 'function' ||
    !('audioWorklet' in AudioContext.prototype)
  ) {
    throw new SottoMicError(
      'unsupported',
      'This browser does not support the AudioWorklet API, which Whisper mode needs for raw microphone capture.',
    );
  }
}

/**
 * Map a getUserMedia rejection onto a SottoMicError per SPEC-V2.md codes.
 * @param {unknown} err the DOMException (or other value) getUserMedia threw
 * @returns {SottoMicError}
 */
function mapGetUserMediaError(err) {
  const name = (err && err.name) || '';
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return new SottoMicError(
      'mic-denied',
      'Microphone access was denied. Whisper mode needs the microphone; allow it in the browser prompt or in site settings, then try again.',
      err,
    );
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
    return new SottoMicError('mic-none', 'No microphone was found on this device.', err);
  }
  return new SottoMicError(
    'mic-failed',
    'Could not open the microphone: ' + (err && err.message ? err.message : String(err)),
    err,
  );
}

// ---------------------------------------------------------------------------
// SottoMic
// ---------------------------------------------------------------------------

/**
 * Microphone capture with a wall-clock-aligned 30 s Float32 ring buffer.
 *
 * ```js
 * const mic = new SottoMic();
 * await mic.start();                    // may throw SottoMicError
 * const pcm = mic.slice(t0Ms, t1Ms);    // Float32Array @ 16 kHz
 * const v = mic.level();                // 0..1 smoothed RMS
 * mic.stop();                           // release tracks, close context
 * ```
 *
 * Lifecycle notes:
 * - start() is idempotent; concurrent calls share one attempt.
 * - stop() is safe before start() and during a pending start(); a start()
 *   interrupted by stop() cleans up and resolves with running === false.
 * - If the OS ends the audio track (device unplugged, permission revoked),
 *   the mic stops itself and `running` becomes false.
 * - Captured audio survives stop(): slice() still serves the final 30 s so a
 *   segment that ended just before stopping can still be transcribed. The
 *   buffer is cleared on the next start(). Nothing is ever persisted.
 */
export class SottoMic {
  constructor() {
    /** @type {boolean} */
    this._running = false;
    /** Generation counter; stop() bumps it to cancel a pending start(). */
    this._gen = 0;
    /** @type {Promise<void>|null} in-flight start attempt, if any */
    this._startPromise = null;

    /** @type {MediaStream|null} */
    this._stream = null;
    /** @type {AudioContext|null} */
    this._context = null;
    /** @type {MediaStreamAudioSourceNode|null} */
    this._source = null;
    /** @type {AudioWorkletNode|null} */
    this._node = null;
    /** @type {GainNode|null} zero-gain sink that keeps the graph pulled */
    this._sink = null;

    /** @type {Float32Array|null} ring buffer, RING_SECONDS at context rate */
    this._ring = null;
    /** Context sample rate the ring is filled at (16000 when granted). */
    this._rate = TARGET_RATE;
    /** Context frame index one past the newest sample in the ring. */
    this._headFrame = -1;
    /** Context frame index of the earliest sample ever captured. */
    this._firstFrame = -1;
    /** @type {number|null} performance.now() of context frame 0 */
    this._epochMs = null;
    /** Smoothed RMS, 0..1. */
    this._level = 0;

    this._onTrackEnded = () => {
      this.stop();
    };
  }

  /**
   * True while the microphone is capturing.
   * @returns {boolean}
   */
  get running() {
    return this._running;
  }

  /**
   * Request the microphone and begin filling the ring buffer.
   *
   * Constraints: audio only (video: false) with echoCancellation: false,
   * noiseSuppression: false, autoGainControl: true. A 16 kHz AudioContext is
   * requested; when the platform refuses (either at construction or when
   * bridging the hardware-rate stream, as Firefox does), capture falls back
   * to the hardware rate and slice() resamples instead.
   *
   * Idempotent: resolves immediately if already running, and concurrent
   * calls await the same attempt.
   *
   * @returns {Promise<void>}
   * @throws {SottoMicError} code 'mic-denied' | 'mic-none' | 'unsupported' | 'mic-failed'
   */
  async start() {
    if (this._running) return;
    if (this._startPromise) return this._startPromise;
    const attempt = this._doStart(++this._gen);
    this._startPromise = attempt;
    try {
      await attempt;
    } finally {
      if (this._startPromise === attempt) this._startPromise = null;
    }
  }

  /**
   * Stop capturing: release all microphone tracks (the browser mic indicator
   * turns off), tear down the audio graph, and close the AudioContext.
   * Safe to call before start(), repeatedly, or while start() is pending
   * (the pending start() is cancelled and resolves without running).
   * The ring buffer keeps its last 30 s for slice() until the next start().
   */
  stop() {
    this._gen++;
    this._teardown();
  }

  /**
   * Extract the audio captured during the wall-clock window [t0Ms, t1Ms]
   * (performance.now() milliseconds — the same clock the engine stamps
   * segment t0/t1 with), resampled to 16 kHz when the context runs at
   * another rate. The window is clamped to what the 30 s ring still holds;
   * a window entirely outside the buffer yields an empty array. The result
   * is a fresh copy the caller owns (safe to transfer to a worker).
   *
   * @param {number} t0Ms window start, performance.now() ms
   * @param {number} t1Ms window end, performance.now() ms
   * @returns {Float32Array} mono samples at 16 kHz (possibly empty)
   */
  slice(t0Ms, t1Ms) {
    if (!Number.isFinite(t0Ms) || !Number.isFinite(t1Ms)) {
      throw new TypeError('SottoMic.slice(t0Ms, t1Ms) needs finite millisecond timestamps');
    }
    const ring = this._ring;
    if (ring === null || this._headFrame < 0 || this._epochMs === null) {
      return new Float32Array(0);
    }
    const rate = this._rate;
    const N = ring.length;
    let f0 = Math.round(((t0Ms - this._epochMs) / 1000) * rate);
    let f1 = Math.round(((t1Ms - this._epochMs) / 1000) * rate);
    const lo = Math.max(this._firstFrame, this._headFrame - N);
    if (f0 < lo) f0 = lo;
    if (f1 > this._headFrame) f1 = this._headFrame;
    if (f1 <= f0) return new Float32Array(0);
    const len = f1 - f0;
    const raw = new Float32Array(len);
    const pos = f0 % N;
    const first = Math.min(len, N - pos);
    raw.set(ring.subarray(pos, pos + first), 0);
    if (first < len) raw.set(ring.subarray(0, len - first), first);
    if (rate === TARGET_RATE) return raw;
    return resampleLinear(raw, rate, TARGET_RATE);
  }

  /**
   * Smoothed RMS of the incoming audio for a live meter, 0..1 (EMA, ~150 ms
   * time constant). 0 when not running. Note the raw scale: with AGC on,
   * normal speech sits around 0.05-0.3 and whispers around 0.01-0.1, so a
   * meter will want its own display scaling.
   * @returns {number}
   */
  level() {
    if (!this._running) return 0;
    return Math.min(1, this._level);
  }

  // -- internals ------------------------------------------------------------

  /**
   * The actual start sequence for one generation. Throws the ABORTED
   * sentinel (caught below, resolving quietly) if stop() bumps the
   * generation mid-flight.
   * @param {number} gen generation this attempt belongs to
   */
  async _doStart(gen) {
    assertSupported();
    const checkGen = () => {
      if (gen !== this._gen) throw ABORTED;
    };

    let stream = null;
    let context = null;
    let committed = false;
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: true,
            channelCount: { ideal: 1 },
          },
          video: false,
        });
      } catch (err) {
        throw mapGetUserMediaError(err);
      }
      checkGen();

      context = await this._makeContext(true);
      checkGen();

      let source;
      try {
        source = context.createMediaStreamSource(stream);
      } catch (err) {
        if (context.sampleRate !== TARGET_RATE) {
          throw new SottoMicError(
            'mic-failed',
            'Could not attach the microphone stream to the audio graph: ' +
              (err && err.message ? err.message : String(err)),
            err,
          );
        }
        // Firefox refuses to bridge a hardware-rate stream into a 16 kHz
        // context. Rebuild at the hardware rate; slice() resamples instead.
        await context.close().catch(() => {});
        context = await this._makeContext(false);
        checkGen();
        source = context.createMediaStreamSource(stream);
      }

      const node = new AudioWorkletNode(context, 'sotto-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: 'explicit',
        channelInterpretation: 'speakers',
      });
      // Zero-gain sink: keeps the graph pulled by the destination without
      // ever making the capture audible.
      const sink = context.createGain();
      sink.gain.value = 0;

      // Commit fields and reset ring state before any chunk can arrive.
      committed = true;
      this._stream = stream;
      this._context = context;
      this._source = source;
      this._node = node;
      this._sink = sink;
      this._rate = context.sampleRate;
      const ringLen = Math.round(RING_SECONDS * context.sampleRate);
      if (this._ring !== null && this._ring.length === ringLen) {
        this._ring.fill(0);
      } else {
        this._ring = new Float32Array(ringLen);
      }
      this._headFrame = -1;
      this._firstFrame = -1;
      this._epochMs = null;
      this._level = 0;

      node.port.onmessage = (event) => {
        if (gen === this._gen) this._ingest(event.data);
      };
      for (const track of stream.getAudioTracks()) {
        track.addEventListener('ended', this._onTrackEnded);
      }
      context.addEventListener('statechange', () => {
        // Recover from interruptions (mobile Safari) while this graph is live.
        if (this._context === context && context.state === 'suspended') {
          context.resume().catch(() => {});
        }
      });

      source.connect(node);
      node.connect(sink);
      sink.connect(context.destination);

      if (context.state !== 'running') {
        try {
          await context.resume();
        } catch (err) {
          throw new SottoMicError(
            'mic-failed',
            'The audio context could not start: ' + (err && err.message ? err.message : String(err)),
            err,
          );
        }
      }
      checkGen();

      this._running = true;
    } catch (err) {
      if (committed && this._context === context) {
        // Failure after commit: tear down through the normal path.
        this._teardown();
      } else {
        // Failure before commit (or stop() already tore the fields down):
        // clean up whatever this attempt created locally.
        if (stream) {
          for (const track of stream.getTracks()) {
            try {
              track.stop();
            } catch (_) {
              /* already stopped */
            }
          }
        }
        if (context && context.state !== 'closed') {
          context.close().catch(() => {});
        }
      }
      if (err === ABORTED) return;
      if (err instanceof SottoMicError) throw err;
      throw new SottoMicError(
        'mic-failed',
        'Microphone start failed: ' + (err && err.message ? err.message : String(err)),
        err,
      );
    }
  }

  /**
   * Create an AudioContext (preferring 16 kHz when preferTarget), verify
   * AudioWorklet, and load the capture worklet module.
   * @param {boolean} preferTarget try {sampleRate: 16000} first
   * @returns {Promise<AudioContext>}
   */
  async _makeContext(preferTarget) {
    let ctx = null;
    if (preferTarget) {
      try {
        ctx = new AudioContext({ sampleRate: TARGET_RATE });
      } catch (_) {
        ctx = null; // this platform refuses 16 kHz contexts; use hardware rate
      }
    }
    if (ctx === null) ctx = new AudioContext();
    if (!ctx.audioWorklet) {
      await ctx.close().catch(() => {});
      throw new SottoMicError(
        'unsupported',
        'This browser does not support the AudioWorklet API, which Whisper mode needs for raw microphone capture.',
      );
    }
    try {
      await ctx.audioWorklet.addModule(WORKLET_URL);
    } catch (err) {
      await ctx.close().catch(() => {});
      throw new SottoMicError(
        'mic-failed',
        'Could not load the audio capture worklet (' +
          WORKLET_URL.href +
          '): ' +
          (err && err.message ? err.message : String(err)),
        err,
      );
    }
    return ctx;
  }

  /**
   * Write one worklet chunk { f, d, n } into the ring at the position its
   * frame index dictates, zero-filling any timeline gap, and update the
   * wall-clock epoch and the level meter.
   * @param {{f: number, d: ArrayBuffer, n: number}} msg
   */
  _ingest(msg) {
    const ring = this._ring;
    if (
      ring === null ||
      !msg ||
      !(msg.d instanceof ArrayBuffer) ||
      typeof msg.f !== 'number' ||
      !Number.isFinite(msg.f) ||
      msg.f < 0
    ) {
      return;
    }
    const n = Math.min(msg.n >>> 0, msg.d.byteLength >> 2);
    if (n === 0) return;
    const samples = new Float32Array(msg.d, 0, n);
    const startFrame = Math.floor(msg.f);
    const endFrame = startFrame + n;
    const rate = this._rate;
    const N = ring.length;

    // Wall-clock alignment: this chunk's last sample was processed just
    // before now, so each arrival bounds the epoch from above; keep the
    // minimum, drift-track slowly, and snap forward after a suspension.
    const cand = performance.now() - (endFrame / rate) * 1000;
    if (this._epochMs === null || cand < this._epochMs) {
      this._epochMs = cand;
    } else if (cand - this._epochMs > EPOCH_JUMP_MS) {
      this._epochMs = cand;
    } else {
      this._epochMs += EPOCH_DRIFT_ALPHA * (cand - this._epochMs);
    }

    // Zero-fill any gap in the frame timeline (input dropouts) so stale ring
    // content is never mistaken for audio from the gap.
    if (this._headFrame >= 0 && startFrame > this._headFrame) {
      const gap = Math.min(startFrame - this._headFrame, N);
      this._fillZeros(startFrame - gap, gap);
    }
    if (this._firstFrame < 0) this._firstFrame = startFrame;

    // Write the chunk (n is far smaller than N, so at most one wrap).
    const pos = startFrame % N;
    const first = Math.min(n, N - pos);
    ring.set(samples.subarray(0, first), pos);
    if (first < n) ring.set(samples.subarray(first), 0);
    if (endFrame > this._headFrame) this._headFrame = endFrame;

    // Level meter: chunk RMS, EMA-smoothed with a fixed time constant.
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const s = samples[i];
      sum += s * s;
    }
    const rms = Math.sqrt(sum / n);
    const alpha = 1 - Math.exp(-(n / rate) / LEVEL_TAU_S);
    this._level += alpha * (rms - this._level);
  }

  /**
   * Zero count ring samples starting at fromFrame's ring position.
   * @param {number} fromFrame first frame to clear
   * @param {number} count frames to clear (at most the ring length)
   */
  _fillZeros(fromFrame, count) {
    const ring = this._ring;
    const N = ring.length;
    if (count >= N) {
      ring.fill(0);
      return;
    }
    const pos = fromFrame % N;
    const first = Math.min(count, N - pos);
    ring.fill(0, pos, pos + first);
    if (first < count) ring.fill(0, 0, count - first);
  }

  /** Release every live resource. Idempotent; never throws. */
  _teardown() {
    const stream = this._stream;
    const context = this._context;
    const source = this._source;
    const node = this._node;
    const sink = this._sink;
    this._running = false;
    this._level = 0;
    this._stream = null;
    this._context = null;
    this._source = null;
    this._node = null;
    this._sink = null;

    if (node) {
      try {
        node.port.postMessage('stop');
      } catch (_) {
        /* port already closed */
      }
      try {
        node.port.onmessage = null;
        node.port.close();
      } catch (_) {
        /* port already closed */
      }
      try {
        node.disconnect();
      } catch (_) {
        /* already disconnected */
      }
    }
    if (source) {
      try {
        source.disconnect();
      } catch (_) {
        /* already disconnected */
      }
    }
    if (sink) {
      try {
        sink.disconnect();
      } catch (_) {
        /* already disconnected */
      }
    }
    if (stream) {
      for (const track of stream.getTracks()) {
        track.removeEventListener('ended', this._onTrackEnded);
        try {
          track.stop();
        } catch (_) {
          /* already stopped */
        }
      }
    }
    if (context && context.state !== 'closed') {
      context.close().catch(() => {});
    }
  }
}

export default SottoMic;
