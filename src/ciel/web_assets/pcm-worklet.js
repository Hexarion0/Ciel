// Continuous interpolation preserves resampling phase across hardware audio blocks.
class CielPCMProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.index = 0; this.next = 0; this.previous = 0; this.step = sampleRate / 16000; }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    const samples = [];
    for (const value of channel) {
      if (this.index === 0) this.previous = value;
      while (this.next <= this.index) {
        const fraction = Math.max(0, Math.min(1, this.next - (this.index - 1)));
        samples.push(this.previous + (value - this.previous) * fraction);
        this.next += this.step;
      }
      this.previous = value; this.index++;
    }
    if (samples.length) this.port.postMessage(new Float32Array(samples));
    return true;
  }
}
registerProcessor('ciel-pcm', CielPCMProcessor);
