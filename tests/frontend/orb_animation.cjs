// Exercise the real canvas path and animation lifecycle without browser dependencies.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');

const html = fs.readFileSync('src/ciel/web_assets/index.html', 'utf8');
const source = html.slice(html.indexOf('function createOrb() {'), html.indexOf('const orb = createOrb();'));

function harness({reduced = false, canvasAvailable = true} = {}) {
  const callbacks = new Map(), listeners = {}, pointerListeners = {};
  const draws = [];
  const work = {frames: 0, strokes: 0, vertices: 0};
  let nextFrame = 0, trace = [];
  const context = {
    createRadialGradient() { return {addColorStop() {}}; },
    clearRect() { trace = []; work.frames++; work.strokes = 0; work.vertices = 0; },
    beginPath() {},
    moveTo(...args) { trace.push(args); work.vertices++; },
    lineTo(...args) { trace.push(args); work.vertices++; },
    arc(...args) {
      assert(args.every(Number.isFinite), 'Canvas geometry must remain finite');
      assert(args[2] >= 0, 'Canvas arc radii must be positive');
      trace.push(args);
    },
    stroke() { work.strokes++; },
    fill() { draws.push(JSON.stringify(trace)); }
  };
  const canvas = {clientWidth: 300, width: 400, height: 400,
    getContext: () => canvasAvailable ? context : null};
  const core = {
    disabled: false,
    classList: {toggle(name, value) { core[name] = value; }},
    addEventListener(name, callback) { pointerListeners[name] = callback; },
    getBoundingClientRect: () => ({left: 0, top: 0, width: 300, height: 300})
  };
  const document = {hidden: false, addEventListener(name, callback) { listeners[name] = callback; }};
  const motion = {matches: reduced, addEventListener(name, callback) { listeners.motion = callback; }};
  const sandbox = {
    element: id => id === 'core' ? core : canvas,
    devicePixelRatio: 3,
    document,
    window: {addEventListener(name, callback) { listeners[name] = callback; }},
    matchMedia: () => motion,
    requestAnimationFrame(callback) { const id = ++nextFrame; callbacks.set(id, callback); return id; },
    cancelAnimationFrame(id) { callbacks.delete(id); }
  };
  vm.createContext(sandbox);
  const orb = vm.runInContext(source + '\ncreateOrb()', sandbox);
  function tick(timestamp) {
    const pending = [...callbacks.values()];
    callbacks.clear();
    for (const callback of pending) callback(timestamp);
    assert(callbacks.size <= 1, 'Only one animation loop may run');
  }
  return {orb, callbacks, listeners, pointerListeners, document, motion, canvas, draws, work, tick};
}

const fallback = harness({canvasAvailable: false});
fallback.orb.update('speaking', false);
assert.equal(fallback.callbacks.size, 0, 'Missing canvas support must leave controls usable');

const page = harness();
assert.equal(page.callbacks.size, 0, 'Connecting must start with a still orb');
assert(page.canvas.width <= 256, 'High DPI rendering must be capped to limit work');
for (const state of ['idle', 'listening_wake', 'recording', 'processing', 'speaking']) {
  page.orb.update(state, false);
  assert.equal(page.callbacks.size, 1, state + ' must animate');
  const initial = page.draws.at(-1);
  for (let step = 0; step < 12; step++) page.tick(step * 50);
  assert.notEqual(page.draws.at(-1), initial, state + ' must visibly move');
  assert(page.work.strokes <= 60, 'Batch canvas work to keep per-frame draw calls bounded');
  assert(page.work.vertices <= 800, 'Keep geometry light enough for shared audio/graphics load');
  const scheduled = [...page.callbacks.keys()];
  page.orb.update(state, false);
  assert.deepEqual([...page.callbacks.keys()], scheduled, 'Status polling must not restart animations');
}
page.orb.update('listening_wake', true);
assert.equal(page.callbacks.size, 0, 'Muting the waiting microphone must pause motion');
page.orb.update('speaking', true);
assert.equal(page.callbacks.size, 1, 'Muted microphone must still show a speaking reply');
page.document.hidden = true;
page.listeners.visibilitychange();
assert.equal(page.callbacks.size, 0, 'Hidden tabs must release their animation loop');
page.document.hidden = false;
page.listeners.visibilitychange();
assert.equal(page.callbacks.size, 1, 'Returning to the tab must resume motion');
page.motion.matches = true;
page.listeners.motion();
assert.equal(page.callbacks.size, 0, 'Changing reduced motion must immediately stop animation');
for (const state of ['idle', 'recording', 'processing', 'speaking']) {
  page.orb.update(state, false);
  assert.equal(page.callbacks.size, 0, 'Reduced motion must keep every state still');
}
page.motion.matches = false;
page.listeners.motion();
assert.equal(page.callbacks.size, 1);
page.pointerListeners.pointermove({clientX: 290, clientY: 10, pointerType: 'mouse'});
page.tick(1000);
page.pointerListeners.pointerleave();
page.canvas.clientWidth = 64;
page.listeners.resize();
assert.equal(page.canvas.width, 64, 'Compact layouts must resize the rendering surface');
page.tick(1100);
page.orb.update('offline', true);
assert.equal(page.callbacks.size, 0, 'Disconnecting must stop animation');

const reducedPage = harness({reduced: true});
reducedPage.orb.update('speaking', false);
assert.equal(reducedPage.callbacks.size, 0, 'Initial reduced-motion preference must be respected');
for (const [state, limit] of [['listening_wake', 13], ['speaking', 17], ['processing', 21]]) {
  const paced = harness();
  paced.orb.update(state, false);
  const before = paced.work.frames;
  for (let step = 0; step <= 60; step++) paced.tick(step * 1000 / 60);
  assert(paced.work.frames - before <= limit, state + ' must stay within its rendering budget');
  assert(paced.work.frames - before >= 8, state + ' must retain visible animation');
}
console.log('Orb checks passed: motion, polling, mute, visibility, reduced motion, resize, fallback.');
