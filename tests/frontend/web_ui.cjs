// Exercise the page's event handlers and async boundaries without a DOM dependency.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.textContent = '';
    this.dataset = {};
    this.style = {};
    this.value = '';
    this.attributes = {};
    this.scrollHeight = 200;
    this.scrollTop = 0;
    this.clientHeight = 400;
    this.classes = new Set();
    this.classList = {
      toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name)
    };
  }
  append(...nodes) { this.children.push(...nodes); }
  get firstElementChild() { return this.children[0]; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  remove() {}
  getContext() { return null; }
  focus() {}
  requestSubmit() { return this.listeners.submit({preventDefault() {}}); }
}

const html = fs.readFileSync('src/ciel/web_assets/index.html', 'utf8');
const elements = new Map(
  [...html.matchAll(/<[a-z][^>]*\bid="([^"]+)"[^>]*>/gi)].map(match => {
    const node = new Element();
    for (const attribute of match[0].matchAll(/([\w-]+)="([^"]*)"/g)) {
      node.setAttribute(attribute[1], attribute[2]);
    }
    return [match[1], node];
  })
);
elements.get('speech').hidden = true;
const timers = new Map();
let nextTimer = 0;
const starters = [new Element(), new Element()];
const state = {
  owner: 'a different owner', wake_phrase: 'hey ciel', state: 'listening_wake',
  muted: false, voice_enabled: true, activity: [], messages: [
    {id: 1, role: 'user', text: '<script>literal text</script>', time: 1},
    {id: 2, role: 'assistant', text: 'Hello.', time: 2}
  ]
};
const requests = [];
let resolvePost = null, holdPost = false, rejectPost = false;
let holdGet = false, rejectGet = false;
function waitForRelease(signal, onReady = () => {}) {
  assert(signal, 'Requests must support cancellation');
  return new Promise((resolve, reject) => {
    const abort = () => {
      const error = Error('Request aborted');
      error.name = 'AbortError';
      reject(error);
    };
    signal.addEventListener('abort', abort, {once: true});
    onReady(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    });
  });
}
function expireDeadline() {
  const deadline = [...timers.values()].find(timer => timer.delay === 8000);
  assert(deadline, 'A stalled request must have a deadline');
  deadline.callback();
}
const context = {
  document: {
    body: new Element(),
    getElementById: id => {
      assert(elements.has(id), 'missing DOM id ' + id);
      return elements.get(id);
    },
    querySelectorAll: () => starters,
    createElement: () => new Element()
  },
  AbortController,
  setInterval() {},
  setTimeout(callback, delay) {
    const id = ++nextTimer;
    timers.set(id, {callback, delay});
    return id;
  },
  clearTimeout(id) { timers.delete(id); },
  fetch: async (url, options) => {
    if (url === '/api/message') {
      const text = JSON.parse(options.body).text;
      requests.push(text);
      if (holdPost) await waitForRelease(options.signal, resolve => resolvePost = resolve);
      if (rejectPost) throw Error('Connection lost');
      if (text === '/mute') state.muted = !state.muted;
      return {ok: true, json: async () => ({accepted: true})};
    }
    if (holdGet) await waitForRelease(options.signal);
    if (rejectGet) throw Error('Connection lost');
    return {ok: true, json: async () => state};
  }
};
vm.createContext(context);
vm.runInContext(html.split('<script>')[1].split('</script>')[0], context);
const settle = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  await settle();
  assert.equal(elements.get('input-label').textContent, '[a different owner]');
  assert(elements.get('speech').hidden, 'Opening the page must not replay history');
  assert.equal(elements.get('core-state').textContent, 'Awaiting your command.');
  assert(!html.includes('id="messages"'), 'The voice page must not contain a conversation feed');
  assert.equal(elements.get('speech-text').attributes.tabindex, '0');
  assert.equal(elements.get('speech-text').attributes['aria-live'], 'off');
  const announcements = [];
  let announcement = '';
  Object.defineProperty(elements.get('reply-announcement'), 'textContent', {
    get: () => announcement,
    set: value => {
      announcement = value;
      if (value) announcements.push(value);
    }
  });
  const reply = {id: 4, role: 'assistant', text: 'Reply <script>literal</script>', time: 4, live: true};
  state.messages.push({id: 3, role: 'user', text: 'Speak', time: 3}, reply);
  state.state = 'speaking';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert(!elements.get('speech').hidden);
  assert.equal(elements.get('speech-text').textContent, reply.text);
  assert.equal(announcements.length, 0, 'Do not repeatedly announce streaming text');
  reply.text += ' with another sentence.';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(elements.get('speech-text').textContent, reply.text);
  reply.live = false;
  state.state = 'listening_wake';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert(!elements.get('speech').hidden, 'Keep completed speech available to read');
  assert.equal(announcements.length, 1);
  const completedAnnouncement = announcements[0];
  assert(completedAnnouncement.endsWith(reply.text), 'Announce the full completed reply');
  assert.equal(timers.size, 0, 'Completed speech must not expire on a timer');
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.deepEqual(announcements, [completedAnnouncement], 'Announce a completed reply only once');
  elements.get('dismiss-speech').listeners.click();
  assert(elements.get('speech').hidden);
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert(elements.get('speech').hidden, 'Polling must not reopen a dismissed reply');

  // Distinct replies with identical wording must each reach assistive technology.
  const repeatedReply = {...reply, id: 6, time: 6, live: true};
  state.messages.push({id: 5, role: 'user', text: 'Repeat that', time: 5}, repeatedReply);
  state.state = 'speaking';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(announcements.length, 1, 'Do not announce the next reply while streaming');
  repeatedReply.live = false;
  state.state = 'listening_wake';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.deepEqual(announcements, [completedAnnouncement, completedAnnouncement]);

  await vm.runInContext("submit('/mute')", context);
  assert.equal(elements.get('core').attributes['aria-pressed'], 'true');
  assert.equal(elements.get('mute-label').textContent, 'Unmute mic');
  elements.get('stop').listeners.click();
  await settle();
  assert.deepEqual(requests, ['/mute', '/stop']);
  state.state = 'speaking';
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(elements.get('core-state').textContent, 'Speaking.');

  // Preserve a new draft typed before the previous submission completes.
  holdPost = true;
  elements.get('input').value = 'first message';
  const pending = elements.get('composer').requestSubmit();
  assert(elements.get('send').disabled);
  assert(elements.get('mute').disabled);
  assert.equal(elements.get('send').attributes['aria-busy'], 'true');
  elements.get('input').value = 'new draft';
  resolvePost();
  await pending;
  holdPost = false;
  assert.equal(elements.get('input').value, 'new draft');
  assert(!elements.get('send').disabled);

  // A failed request retains text and permits retry.
  rejectPost = true;
  await elements.get('composer').requestSubmit();
  assert.equal(elements.get('input').value, 'new draft');
  assert.equal(elements.get('notice').textContent, 'Connection lost');
  assert(elements.get('notice').classes.has('error'));
  await vm.runInContext('poll()', context);
  assert.equal(elements.get('notice').textContent, 'Connection lost');
  assert(!elements.get('send').disabled);
  rejectPost = false;

  // A stalled submission aborts, keeps the draft, and permits another command.
  holdPost = true;
  elements.get('input').value = 'draft retained after timeout';
  const stalledSubmission = elements.get('composer').requestSubmit();
  assert.equal(elements.get('send').attributes['aria-busy'], 'true');
  expireDeadline();
  await stalledSubmission;
  holdPost = false;
  assert.equal(elements.get('input').value, 'draft retained after timeout');
  assert.equal(elements.get('send').attributes['aria-busy'], 'false');
  assert(elements.get('notice').classes.has('error'));
  assert.equal(timers.size, 0, 'The timed-out submission must release its deadline');
  await vm.runInContext('poll()', context);
  assert(!elements.get('send').disabled);
  assert(!elements.get('mute').disabled);
  assert(await vm.runInContext("submit('/help')", context), 'Commands must recover after timeout');

  // A lost connection disables runtime actions while preserving editable drafts.
  rejectGet = true;
  await vm.runInContext('poll()', context);
  assert(!elements.get('input').disabled, 'Offline users must still be able to edit a draft');
  for (const id of ['send', 'core', 'mute', 'stop', 'help', 'exit']) {
    assert(elements.get(id).disabled, id + ' must be disabled while offline');
  }
  elements.get('input').value = 'prepared while disconnected';
  const previousRequests = requests.length;
  await elements.get('composer').requestSubmit();
  assert.equal(requests.length, previousRequests, 'Do not submit a message while offline');
  assert.equal(elements.get('input').value, 'prepared while disconnected');
  rejectGet = false;
  await vm.runInContext('poll()', context);
  assert(!elements.get('send').disabled);
  assert.equal(elements.get('input').value, 'prepared while disconnected');

  // A stalled status request must release the polling guard and reconnect later.
  holdGet = true;
  const stalledPoll = vm.runInContext('poll()', context);
  expireDeadline();
  await stalledPoll;
  holdGet = false;
  assert(elements.get('send').disabled);
  assert(!elements.get('input').disabled);
  assert.equal(timers.size, 0);
  await vm.runInContext('poll()', context);
  assert(!elements.get('send').disabled, 'Polling must recover after a status timeout');
  assert.equal(elements.get('input').value, 'prepared while disconnected');

  // A late status response must not re-enable controls after shutdown.
  await vm.runInContext("submit('/exit')", context);
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(elements.get('core-state').textContent, 'Session ended.');
  assert(elements.get('input').disabled);
  assert(elements.get('core').disabled);
  console.log('UI checks passed: speech, announcements, controls, drafts, timeout, reconnect, shutdown.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
