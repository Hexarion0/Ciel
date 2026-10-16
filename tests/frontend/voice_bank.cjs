// Verify the audition page's blind gate, single-player behavior, notes, and filters.
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const html = fs.readFileSync('scripts/voice_bank/audition.html', 'utf8');
const code = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(dataset = {}) { this.dataset = dataset; this.listeners = {}; this.value = ''; this.hidden = false; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  pause() { this.paused = true; }
  querySelector() { return new Element(); }
}
const a = new Element({candidate: 'candidate-neutral-01', blind: 'A'});
const b = new Element({candidate: 'candidate-neutral-01', blind: 'B'});
a.parentElement = b.parentElement = new Element();
const reveal = new Element({candidate: 'candidate-neutral-01'}); reveal.disabled = true;
const candidate = new Element({style: 'neutral'});
const field = new Element({field: 'decision'}); field.value = 'review';
const feedback = new Element({candidate: 'candidate-neutral-01'});
feedback.querySelectorAll = () => [field];
const nodes = new Map(['settings','save-status','filter','empty','export','reveal-candidate-neutral-01']
  .map(id => [id, new Element()]));
nodes.get('settings').textContent = JSON.stringify({run_id: 'test', blind_key: {
  'candidate-neutral-01': {A: 'FP32', B: 'INT8'}
}});
const stored = new Map();
let downloaded = null;
const document = {
  getElementById: id => nodes.get(id),
  querySelectorAll: selector => ({'.feedback': [feedback], audio: [a,b], '.reveal': [reveal], '.candidate': [candidate]})[selector],
  querySelector: () => reveal,
  body: {appendChild() {}},
  createElement: () => ({click() {}, remove() {}})
};
vm.runInNewContext(code, {
  document, console, setTimeout: fn => fn(),
  localStorage: {getItem: key => stored.get(key), setItem: (key,value) => stored.set(key,value)},
  Blob: class {constructor(chunks) {this.text = chunks.join('');}},
  URL: {createObjectURL(blob) {downloaded = JSON.parse(blob.text); return 'blob:test';}, revokeObjectURL() {}}
});
reveal.listeners.click();
assert.equal(nodes.get('reveal-candidate-neutral-01').textContent, undefined);
a.listeners.play(); assert.equal(b.paused, true);
a.listeners.ended(); assert.equal(reveal.disabled, true);
b.listeners.ended(); assert.equal(reveal.disabled, false);
reveal.listeners.click();
assert.equal(nodes.get('reveal-candidate-neutral-01').textContent, 'A = FP32 · B = INT8');
field.value = 'keep'; field.listeners.input();
nodes.get('export').listeners.click();
assert.equal(downloaded.decisions['candidate-neutral-01'].decision, 'keep');
assert.equal(downloaded.decisions['candidate-neutral-01'].revealed_after_listening, true);
nodes.get('filter').listeners.change({target: {value: 'firm'}});
assert.equal(candidate.hidden, true); assert.equal(nodes.get('empty').hidden, false);
nodes.get('filter').listeners.change({target: {value: 'all'}});
assert.equal(candidate.hidden, false); assert.equal(nodes.get('empty').hidden, true);
console.log('Voice audition interaction checks passed.');
