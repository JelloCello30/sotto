/*
 * Sotto — microphone capture worklet (js/audio-worklet.js)
 *
 * Runs inside AudioWorkletGlobalScope, loaded by js/audio.js via
 * new URL('audio-worklet.js', import.meta.url) so subpath deploys work.
 *
 * Copies input samples into fixed-size chunks and posts each chunk to the
 * main thread as { f, d, n }:
 *   f — AudioContext frame index of the chunk's first sample
 *   d — ArrayBuffer of Float32 samples (transferred, not copied)
 *   n — number of valid samples in d
 *
 * The frame index is what lets the main thread keep the ring buffer aligned
 * to wall-clock time even across input gaps. Nothing is stored here, the
 * node's output stays silent, and a 'stop' message from the main thread
 * flushes the partial chunk and retires the processor.
 */

/** Samples per posted chunk (~64 ms at 16 kHz, ~21 ms at 48 kHz). */
const CHUNK_SAMPLES = 1024;

class SottoCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    /** Accumulation buffer; its ArrayBuffer is transferred when full. */
    this._chunk = new Float32Array(CHUNK_SAMPLES);
    this._fill = 0;
    this._chunkStartFrame = 0;
    this._stopped = false;
    /** Scratch buffer for downmixing, allocated only if input is multichannel. */
    this._mix = null;
    this.port.onmessage = (event) => {
      if (event.data === 'stop') {
        this._flush();
        this._stopped = true;
      }
    };
  }

  /** Post whatever is accumulated (used on input gaps and on stop). */
  _flush() {
    if (this._fill === 0) return;
    const partial = this._chunk.slice(0, this._fill);
    this.port.postMessage(
      { f: this._chunkStartFrame, d: partial.buffer, n: partial.length },
      [partial.buffer],
    );
    this._fill = 0;
  }

  process(inputs) {
    if (this._stopped) return false;
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      // Input gap (track muted, device settling). Flush so the partial chunk
      // keeps its correct start frame instead of absorbing later samples.
      this._flush();
      return true;
    }
    const n = input[0].length;
    let mono = input[0];
    if (input.length > 1) {
      // The node is configured mono upstream, but downmix defensively by
      // averaging in case a browser delivers multiple channels anyway.
      if (this._mix === null || this._mix.length !== n) {
        this._mix = new Float32Array(n);
      }
      const mix = this._mix;
      mix.set(input[0]);
      for (let c = 1; c < input.length; c++) {
        const ch = input[c];
        for (let i = 0; i < n; i++) mix[i] += ch[i];
      }
      const inv = 1 / input.length;
      for (let i = 0; i < n; i++) mix[i] *= inv;
      mono = mix;
    }
    let i = 0;
    while (i < n) {
      if (this._fill === 0) this._chunkStartFrame = currentFrame + i;
      const take = Math.min(n - i, CHUNK_SAMPLES - this._fill);
      this._chunk.set(mono.subarray(i, i + take), this._fill);
      this._fill += take;
      i += take;
      if (this._fill === CHUNK_SAMPLES) {
        this.port.postMessage(
          { f: this._chunkStartFrame, d: this._chunk.buffer, n: CHUNK_SAMPLES },
          [this._chunk.buffer],
        );
        this._chunk = new Float32Array(CHUNK_SAMPLES);
        this._fill = 0;
      }
    }
    return true;
  }
}

registerProcessor('sotto-capture', SottoCaptureProcessor);
