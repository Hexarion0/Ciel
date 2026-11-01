'use strict';
// Live captions: on the device that plays a spoken reply, its text appears in step with the
// voice, like the terminal's captions. Each audio chunk says which caption characters it
// speaks; the reveal moves through them as the chunk plays. When playback ends, stops, or
// never starts, the full formatted reply is shown instead.

const CAPTION_NO_JOB_MS = 1000;    // no audio job offered to this device: someone else's reply
const CAPTION_MEMORY = 100;        // finished captions kept before the oldest are forgotten
const CAPTION_START_LIMIT_MS = 30000;  // voice never started: stop hiding the answer

class LiveCaptions {
  constructor(now = () => Date.now()) {
    this.now = now;
    this.items = new Map();
  }
  /** Decide, the first time a spoken reply is seen, whether to wait for its voice.
   *  ``fresh`` is false for replies already on screen when the page loaded (history), so
   *  no clock comparison between the PC and this device is needed. */
  observe(message, {audible, fresh = true}) {
    if (!message || message.role !== 'assistant' || !message.spoken || this.items.has(message.id)) return;
    this.track(message.id, fresh && audible ? 'waiting' : 'done');
  }
  /** Create state for a reply; playback may report before the chat poll draws the message. */
  track(id, phase) {
    const item = {phase, since: this.now(), caption: '', from: 0, to: 0, segmentStart: 0, segmentMs: 1};
    this.items.set(id, item);
    if (this.items.size > CAPTION_MEMORY)
      for (const [old, entry] of this.items)
        if (entry.phase === 'done' && this.items.size > CAPTION_MEMORY) this.items.delete(old);
    return item;
  }
  /** Every poll lists this device's audio jobs; a reply without one is not ours to voice. */
  listed(jobs) {
    const voiced = new Set((jobs || []).map(job => job.message));
    for (const [id, item] of this.items)
      if (item.phase === 'waiting' && !voiced.has(id) && this.now() - item.since > CAPTION_NO_JOB_MS)
        item.phase = 'done';
  }
  begin(id, caption) {
    if (id === null || id === undefined) return false;
    const item = this.items.get(id) || this.track(id, 'waiting');
    if (item.phase === 'done') return false;
    Object.assign(item, {phase: 'playing', caption: String(caption || ''), from: 0, to: 0,
      segmentStart: this.now(), segmentMs: 1});
    return true;
  }
  /** A chunk started playing: reveal characters from..to over its duration. */
  segment(id, from, to, seconds) {
    const item = this.items.get(id);
    if (!item || item.phase !== 'playing') return;
    const clamp = value => Math.max(0, Math.min(item.caption.length, Number(value) || 0));
    item.from = Math.max(clamp(from), this.revealed(item));
    item.to = Math.max(item.from, clamp(to));
    item.segmentStart = this.now();
    item.segmentMs = Math.max(1, (Number(seconds) || 0) * 1000);
  }
  finish(id) {
    if (id === null || id === undefined) return;
    (this.items.get(id) || this.track(id, 'done')).phase = 'done';
  }
  finishAll() { for (const item of this.items.values()) item.phase = 'done'; }
  revealed(item) {
    const progress = Math.min(1, Math.max(0, (this.now() - item.segmentStart) / item.segmentMs));
    return Math.round(item.from + (item.to - item.from) * progress);
  }
  /** Caption text to show for a message, '' while its voice starts, or null for full text. */
  text(message) {
    const item = message && this.items.get(message.id);
    if (!item) return null;
    if (item.phase === 'waiting' && this.now() - item.since > CAPTION_START_LIMIT_MS) item.phase = 'done';
    if (item.phase === 'done') return null;
    return item.phase === 'waiting' ? '' : item.caption.slice(0, this.revealed(item));
  }
  active() { for (const item of this.items.values()) if (item.phase !== 'done') return true; return false; }
}
