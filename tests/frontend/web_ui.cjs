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
  append(...nodes) { for (const n of nodes) { n.parent = this; this.children.push(n); } }
  get parentElement() { return this.parent; }
  get firstElementChild() { return this.children[0]; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
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
  URL, crypto: require('node:crypto').webcrypto,
  window: {},
  AbortController,
  setInterval() {},
  setTimeout(callback, delay) {
    const id = ++nextTimer;
    timers.set(id, {callback, delay});
    return id;
  },
  clearTimeout(id) { timers.delete(id); },
  fetch: async (url, options) => {
    if (url === '/api/conversations/activate') { state.active={handle:'next-chat',revision:2}; return {ok:true,json:async()=>state.active}; }
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
vm.runInContext(['orb.js', 'render.js', 'audio.js', 'attachments.js', 'settings.js', 'chat.js'].map(name => fs.readFileSync('src/ciel/web_assets/' + name, 'utf8')).join('\n'), context);
const settle = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  await settle();
  assert.equal(elements.get('input-label').textContent, '[a different owner]');
  assert.equal(elements.get('messages').children.length, 2, 'Opening chat shows saved history');
  assert.equal(elements.get('core-state').textContent, 'Awaiting your command.');
  const feed = elements.get('messages');
  const first = feed.children[0];
  const reply = {id: 4, role: 'assistant', text: 'Reply <script>literal</script>', time: 4, live: true};
  state.messages.push({id: 3, role: 'user', text: 'Speak', time: 3}, reply);
  feed.scrollHeight = 2000; feed.scrollTop = 100;
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(feed.children.length, 4);
  assert.equal(feed.children[0], first, 'Polling keeps old message nodes');
  assert.equal(feed.scrollTop, 100, 'Reading earlier messages must not force scroll');
  assert(!elements.get('new-messages').hidden);
  elements.get('new-messages').listeners.click();
  assert.equal(feed.scrollTop, feed.scrollHeight);
  reply.text += ' with another sentence.'; reply.live = false;
  vm.runInContext('render(' + JSON.stringify(state) + ')', context);
  assert.equal(feed.children.length, 4, 'Stream updates the same message');
  elements.get('history-toggle').listeners.click();
  assert.equal(elements.get('history-toggle').attributes['aria-expanded'], 'true');
  elements.get('history-close').listeners.click();
  assert.equal(elements.get('history-toggle').attributes['aria-expanded'], 'false');

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

  vm.runInContext('attachments.busy=true',context);
  assert(await vm.runInContext("submit('/stop')",context), 'Stop works while an image is uploading');
  vm.runInContext('attachments.busy=false; attachments.items=[{error:true}]',context);
  assert(await vm.runInContext("submit('/help')",context), 'Session controls work with failed images');
  vm.runInContext('attachments.items=[]',context);

  // Explicit switching clears temporary local audio and attachments as well as the server queue.
  state.active={handle:'current-chat',revision:1};
  vm.runInContext('render('+JSON.stringify(state)+'); attachments.items=[{handle:"uploaded",url:"blob:preview"}]; attachments.render()',context);
  await vm.runInContext('activateConversation(null)',context);
  assert.equal(vm.runInContext('attachments.admittedIds().length',context),0,'New chat clears old attachments');

  context.matchMedia=()=>({matches:true});
  vm.runInContext('setDrawer(false)',context);
  assert.equal(elements.get('sidebar').inert,true,'Closed mobile history is not keyboard focusable');
  assert.equal(elements.get('presence').parentElement,elements.get('topbar'),'Phone orb stays outside the inert drawer');
  vm.runInContext('setDrawer(true)',context);
  assert.equal(elements.get('sidebar').inert,false);
  delete context.matchMedia;

  vm.runInContext('renderTimeline([{id:1,role:"assistant",text:"An archived reply"}])',context);
  assert.equal(elements.get('messages').children[0].className,'message assistant','A different archive must not inherit a user label from a colliding message ID');

  // Either exit command ends the session; late status cannot re-enable controls.
  for (const command of ['/exit', '/QuIt']) {
    vm.runInContext('runtime.ended = false; render(' + JSON.stringify(state) + ')', context);
    assert(await vm.runInContext('submit(' + JSON.stringify(command) + ')', context));
    vm.runInContext('render(' + JSON.stringify(state) + ')', context);
    assert.equal(elements.get('core-state').textContent, 'Session ended.');
    assert(elements.get('input').disabled);
    assert(elements.get('core').disabled);
  }
  console.log('UI checks passed: speech, announcements, controls, drafts, timeout, reconnect, shutdown.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
