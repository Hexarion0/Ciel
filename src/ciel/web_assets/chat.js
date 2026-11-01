
'use strict';
const element = id => document.getElementById(id);
const runtime = {
  online: false, pending: false, polling: false, ended: false, initialized: false,
  cursor: null, liveMessages: [], activityTime: 0, baseline: 0, replyId: null, replyLive: false, dismissed: null, announced: null
};
const controls = ['core', 'mute', 'stop', 'exit', 'help', 'send', 'settings-open'];
function text(node, value) { if (node.textContent !== value) node.textContent = value; }
function feedback(value, error = false) {
  text(element('notice'), value);
  element('notice').classList.toggle('error', error);
}
function syncControls() {
  element('input').disabled = runtime.ended;
  for (const id of controls) element(id).disabled = !runtime.online || runtime.pending;
  if (typeof browsingHandle !== 'undefined' && browsingHandle) element('send').disabled = true;
  element('attach').disabled = runtime.pending || attachments.busy || !runtime.online;
  element('record').disabled = !runtime.online || Boolean(browsingHandle);
  element('send').setAttribute('aria-busy', String(runtime.pending));
  element('composer').setAttribute('aria-busy', String(runtime.pending));
}
const messageNodes = new Map();
function hideSpeech() {} // Stopping playback leaves the transcript available.
/** Show a live caption while this device speaks the reply; otherwise the full message. */
function renderCaption(body, message) {
  // Replies already present at the first render are history; only later ones are captioned.
  if (liveCaptions) liveCaptions.observe(message, {audible: browserAudio.available && !browserAudio.muted,
    fresh: runtime.initialized});
  const caption = liveCaptions ? liveCaptions.text(message) : null;
  body.classList.toggle('captioning', caption !== null);
  if (caption === null) {
    const changed = body.dataset.content !== message.text;
    body.dataset.caption = '';
    renderMessageContent(body, message.text);
    return changed;
  }
  if (body.dataset.caption === caption && body.dataset.content === '') return false;
  clearNode(body); body.textContent = caption;
  body.dataset.caption = caption; body.dataset.content = '';
  return true;
}
let captionTimer = null;
/** Repaint only captioned replies while any reply is being spoken. */
function paintCaptions() {
  if (!liveCaptions) return;
  if (!browsingHandle) {
    const feed = element('messages');
    let following = null, changed = false;
    for (const message of runtime.liveMessages) {
      const row = messageNodes.get(message.id), item = liveCaptions.items.get(message.id);
      if (!row || !item || (item.phase === 'done' && !row.body.classList.contains('captioning'))) continue;
      if (following === null) following = feed.scrollHeight - feed.clientHeight - feed.scrollTop < 80;
      changed = renderCaption(row.body, message) || changed;
    }
    if (changed && following) feed.scrollTop = feed.scrollHeight;
  }
  if (mobileUI) mobileUI.syncTranscript();
  if (!liveCaptions.active() && captionTimer !== null) { clearInterval(captionTimer); captionTimer = null; }
}
function watchCaptions() {
  if (liveCaptions && liveCaptions.active() && captionTimer === null) captionTimer = setInterval(paintCaptions, 80);
}
function renderTimeline(messages, {followLatest = true} = {}) {
  const feed = element('messages');
  const following = feed.scrollHeight - feed.clientHeight - feed.scrollTop < 80;
  let changed = false;
  const ids = new Set(messages.map(message => message.id));
  for (const [id, row] of messageNodes) {
    if (!ids.has(id)) { row.root.remove(); messageNodes.delete(id); }
  }
  for (const message of messages) {
    let row = messageNodes.get(message.id);
    if (!row) {
      const root = document.createElement('article'); root.className = 'message ' + message.role;
      const author = document.createElement('div'); author.className = 'message-author';
      author.textContent = message.role === 'user' ? 'You' : 'Ciel';
      const body = document.createElement('div'); body.className = 'message-content';
      const sources = document.createElement('nav'); sources.className = 'message-sources';
      sources.setAttribute('aria-label', "Sources for Ciel's reply");
      const note = document.createElement('div'); note.className = 'message-note';
      root.append(author, body, sources, note); feed.append(root);
      row = {root, author, body, sources, note}; messageNodes.set(message.id, row); changed = true;
    }
    row.root.className = 'message ' + message.role;
    text(row.author, message.role === 'user' ? 'You' : 'Ciel');
    if (renderCaption(row.body, message)) changed = true;
    renderMessageSources(row.sources, message.sources);
    text(row.note, message.interrupted ? 'Reply stopped' : message.live ? 'Speaking…' : '');
  }
  element('welcome').hidden = messages.length > 0;
  if (changed && following && followLatest) { feed.scrollTop = feed.scrollHeight; element('new-messages').hidden = true; }
  else if (changed) element('new-messages').hidden = false;
}
let drawerOpen = false;
function syncDrawer() {
  const mobile = typeof matchMedia === 'function' && matchMedia('(max-width: 759px)').matches;
  const presence = element('presence');
  const slot = mobile ? (mobileUI ? mobileUI.orbSlot() : element('topbar')) : element('sidebar');
  let moved = false;
  if (mobile && presence.parentElement !== slot) { slot.append(presence); moved = true; }
  else if (!mobile && presence.parentElement && presence.parentElement !== slot) {
    element('sidebar').insertBefore(presence, element('connect-phone')); moved = true;
  }
  // The orb sizes its canvas on resize; a new slot (sidebar, header, voice stage) changes its size.
  if (moved && typeof window !== 'undefined' && window.dispatchEvent && typeof Event === 'function')
    window.dispatchEvent(new Event('resize'));
  element('sidebar').inert = mobile && !drawerOpen;
}
function setDrawer(open) {
  drawerOpen = open;
  document.body.classList.toggle('history-open', open);
  element('history-toggle').setAttribute('aria-expanded', String(open));
  syncDrawer();
  if (!open) element('history-toggle').focus();
  else element('history-close').focus();
}
const orb = createOrb();
function disconnect() {
  runtime.online = false;
  browserAudio.resync = true; browserAudio.stop();
  document.body.classList.toggle('offline', true);
  text(element('connection'), runtime.ended ? 'Ended' : 'Reconnecting');
  text(element('core-state'), runtime.ended ? 'Session ended.' : 'Disconnected.');
  text(element('state-hint'), 'Start ciel web to connect.');
  orb.update('offline', true);
  if (runtime.ended) hideSpeech();
  syncControls();
}
function render(snapshot) {
  if (runtime.ended) return;
  if (snapshot.cursor !== undefined) {
    const retained = new Set(snapshot.retained_ids || []);
    const merged = new Map((snapshot.reset ? [] : runtime.liveMessages).filter(message => retained.has(message.id)).map(message => [message.id, message]));
    for (const message of snapshot.messages) merged.set(message.id, message);
    runtime.liveMessages = Array.from(merged.values()).sort((a,b) => a.id-b.id);
    runtime.cursor = snapshot.cursor;
    snapshot = {...snapshot, messages:runtime.liveMessages};
  }
  const reconnected = !runtime.online;
  runtime.online = true;
  document.body.classList.toggle('offline', false);
  document.body.classList.toggle('muted', snapshot.muted);
  document.body.dataset.state = snapshot.state;
  text(element('connection'), 'Connected');
  text(element('owner-label'), snapshot.owner);
  const labels = {
    listening_wake: snapshot.muted ? 'Microphone muted.' : 'Awaiting your command.',
    recording: 'Listening to you.', processing: 'Thinking.', speaking: 'Speaking.', idle: 'Ready.'
  };
  text(element('core-state'), labels[snapshot.state] || 'Ready.');
  const hint = snapshot.state === 'speaking' ? 'Use Stop reply to interrupt.'
    : snapshot.state === 'processing' ? 'Use Stop reply to cancel.'
    : snapshot.state === 'recording' ? 'Listening through your PC microphone.'
    : snapshot.muted ? 'You can still type' + (snapshot.voice_enabled ? ' and hear replies.' : '.')
    : 'Say “' + snapshot.wake_phrase + '”, or type below.';
  text(element('state-hint'), hint);
  const observation = snapshot.observation || 'Screen observation: off.';
  text(element('observation-state'), observation.replace(/\. Use .*$/, ''));
  element('observation-state').setAttribute('title', observation);
  text(element('input-label'), '[' + snapshot.owner + ']');
  text(element('mute-label'), snapshot.muted ? 'Unmute mic' : 'Mute mic');
  text(element('orb-action'), snapshot.muted ? 'Unmute mic' : 'Mute mic');
  for (const id of ['core', 'mute']) {
    element(id).setAttribute('aria-pressed', String(snapshot.muted));
    element(id).setAttribute('aria-label', snapshot.muted ? 'Unmute microphone' : 'Mute microphone');
  }
  orb.update(snapshot.state, snapshot.muted);
  syncControls();
  if (snapshot.active) {
    if (activeSelection && snapshot.active.revision !== activeSelection.revision && element('input').value.trim() && !browsingHandle) {
      browsingHandle = activeSelection.handle;
      element('archive-banner').hidden = false;
      feedback('The active conversation changed. Choose a chat before sending your draft.');
    }
    if (activeSelection && snapshot.active.revision !== activeSelection.revision) {
      browserAudio.stop(); attachments.clear(); element('read-full').hidden = true;
    }
    activeSelection = snapshot.active;
    if (Date.now() - catalogTime > 3000) { catalogTime = Date.now(); loadConversations(); }
  }
  browserAudio.syncState(snapshot);
  if (typeof matchMedia === 'function' && matchMedia('(max-width:759px)').matches) text(element('chat-subtitle'), element('core-state').textContent);
  if (!browsingHandle) {
    renderTimeline(snapshot.messages);
    text(element('chat-title'), snapshot.title || 'A little space to think');  // Ciel names chats
  }
  watchCaptions();
  if (commandPicker && !commandPicker.catalog) commandPicker.load();
  const latest = snapshot.messages.at(-1);
  if (runtime.initialized && latest?.role === 'assistant' && !latest.live && runtime.announced !== latest.id) {
    text(element('reply-announcement'), 'Ciel: ' + latest.text); runtime.announced = latest.id;
  }
  if (!runtime.initialized) { runtime.initialized = true; runtime.announced = latest?.id; }
  const activity = snapshot.activity.at(-1);
  if (activity && activity.time > runtime.activityTime) {
    runtime.activityTime = activity.time;
    feedback(activity.text, activity.error);
  } else if (reconnected && !runtime.pending) feedback('');
}
async function requestJSON(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const data = await response.json();
    return {response, data};
  } finally { clearTimeout(timeout); }
}
async function poll() {
  if (runtime.polling || runtime.ended) return;
  runtime.polling = true;
  try {
    const {response, data: snapshot} = await requestJSON('/api/state' + (runtime.cursor === null ? '' : '?after=' + runtime.cursor), {cache: 'no-store'});
    if (!response.ok) {
      if (response.status === 401) {
        element('pairing').hidden = false; cielSettings.close(); attachments.clear(); browserAudio.stop();
        runtime.liveMessages=[];runtime.cursor=null;activeSelection=null;browsingHandle=null;
        clearNode(element('messages')); messageNodes.clear(); clearNode(element('conversation-list'));
      }
      throw Object.assign(Error(), {unpaired: response.status === 401});
    }
    if (!runtime.ended) render(snapshot);
  } catch (error) {
    if (!runtime.ended) {
      disconnect();
      // An unpaired device is refused on purpose; the pairing form already explains it.
      feedback(error.unpaired ? '' : 'Cannot reach Ciel. Check that ciel web is running.', !error.unpaired);
    }
  } finally { runtime.polling = false; }
}
async function submit(value) {
  const message = value.trim() || (attachments.admittedIds().length ? 'What can you tell me about this image?' : '');
  if (!message || !runtime.online || runtime.pending || runtime.ended || (browsingHandle && (!message.startsWith('/') || /^\/(?:fast|strong|deep)(?:\s|$)/i.test(message)))) return false;
  const command = message.startsWith('/') && !/^\/(?:fast|strong|deep)(?:\s|$)/i.test(message);
  // /help opens the command panel right here instead of a one-line server reply.
  const help = /^\/help(?:\s+(.*))?$/i.exec(message);
  if (help && helpPanel && helpPanel.open(help[1] || '')) return true;
  attachments.checkExpiry();
  if (!command && (attachments.busy || attachments.items.some(item => !item.handle))) { feedback('Finish uploading or remove the failed image before sending.', true); return false; }
  if (!command) browserAudio.enable();
  const attachmentEpoch = attachments.epoch;
  runtime.pending = true;
  syncControls();
  feedback(message.startsWith('/') ? 'Sending command…' : 'Sending message…');
  try {
    const {response, data: result} = await requestJSON('/api/message', {
      method: 'POST', headers: browserAudio.headers({'Content-Type': 'application/json'}), body: JSON.stringify({text: message,
        request_id: attachments.requestId, attachment_ids: attachments.admittedIds(),
        revision: activeSelection?.revision, conversation_handle: activeSelection?.handle})
    });
    if (!response.ok || !result.accepted) throw Error(result.error || 'Ciel could not accept this request. Try again.');
    if (attachments.epoch === attachmentEpoch && (!command || message.toLowerCase() === '/stop')) attachments.clear();
    if (['/exit', '/quit'].includes(message.toLowerCase())) {
      runtime.ended = true; disconnect(); feedback('Session ended. Start ciel web to connect again.');
    } else {
      if (!message.startsWith('/') || message.toLowerCase() === '/stop') hideSpeech();
      feedback(message.startsWith('/') ? 'Command sent.' : 'Message sent.');
      await poll();
    }
    return true;
  } catch (error) {
    feedback(error.name === 'AbortError'
      ? 'Ciel did not respond in time. Check the connection before retrying.'
      : error.message || 'Could not send. Try again.', true);
    return false;
  } finally { runtime.pending = false; syncControls(); }
}
element('new-messages').addEventListener('click', () => {
  const feed = element('messages'); feed.scrollTop = feed.scrollHeight; element('new-messages').hidden = true;
});
element('history-toggle').addEventListener('click', () => setDrawer(true));
element('history-close').addEventListener('click', () => setDrawer(false));
element('drawer-shade').addEventListener('click', () => setDrawer(false));
// Escape closes the history drawer, unless it is closing the command list first.
element('input').addEventListener('keydown', event => { if (event.key === 'Escape' && element('command-menu').hidden) setDrawer(false); });
element('composer').addEventListener('submit', async event => {
  event.preventDefault();
  const input = element('input'), draft = input.value;
  if (await submit(draft)) {
    if (input.value === draft) { input.value = ''; input.style.height = 'auto'; }
    if (runtime.online) input.focus();
  }
});
element('input').addEventListener('input', () => {
  const input = element('input'); input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 100) + 'px';
  if (commandPicker) commandPicker.render();
});
element('input').addEventListener('focus', () => { if (commandPicker) commandPicker.setOpen(true); });
element('input').addEventListener('blur', () => { if (commandPicker) commandPicker.setOpen(false); });
element('input').addEventListener('keydown', event => {
  if (commandPicker && commandPicker.key(event)) return;
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); element('composer').requestSubmit();
  }
});
element('mute').addEventListener('click', () => submit('/mute'));
element('core').addEventListener('click', () => browserAudio.available ? browserAudio.toggleRecording() : submit('/mute'));
for (const id of ['stop', 'exit', 'help']) element(id).addEventListener('click', () => { if(id !== 'help') browserAudio.stop({microphone:id==='exit'}); submit('/' + id); });


let activeSelection = null, browsingHandle = null, viewEpoch = 0;
let catalogCursor = null, historyCursor = null, historyMessages = [], catalogTime = 0;
let catalogBusy = false, showingHidden = false;
async function loadConversations(append = false) {
  if (catalogBusy) return;
  catalogBusy = true;
  try {
    const params = [append && catalogCursor ? 'before=' + catalogCursor : '', showingHidden ? 'hidden=1' : ''].filter(Boolean);
    const {response, data} = await requestJSON('/api/conversations' + (params.length ? '?' + params.join('&') : ''));
    if (!response.ok || !Array.isArray(data.conversations)) return;
    const nav = element('conversation-list');
    if (!append) clearNode(nav);
    for (const item of data.conversations) {
      const row = document.createElement('div'); row.className = 'conversation-row';
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = item.title; button.title = item.title;
      button.setAttribute('aria-current', String(item.handle === (browsingHandle || activeSelection?.handle)));
      button.addEventListener('click', () => browseConversation(item.handle, item.title));
      const remove = document.createElement('button'); remove.type = 'button';
      if (showingHidden) {
        remove.className = 'conversation-restore'; remove.textContent = 'Restore';
        remove.setAttribute('aria-label', 'Restore ' + item.title + ' to your list');
        remove.addEventListener('click', () => restoreConversation(item.handle));
      } else {
        remove.className = 'conversation-remove'; remove.textContent = '×';
        remove.setAttribute('aria-label', 'Remove ' + item.title + ' from the list');
        remove.addEventListener('click', () => hideConversation(item.handle, item.title));
      }
      row.append(button, remove); nav.append(row);
    }
    catalogCursor = data.next_before; element('history-more').hidden = !catalogCursor;
    syncHiddenToggle(data.hidden_count || 0);
  } catch { /* Status polling supplies the connection warning. */ }
  finally { catalogBusy = false; }
}
/** Offer the hidden chats view only when something is hidden (or while viewing it). */
function syncHiddenToggle(count) {
  const toggle = element('history-hidden');
  if (showingHidden && !count) { showingHidden = false; loadConversations(); return; }
  toggle.hidden = !showingHidden && !count;
  text(toggle, showingHidden ? 'Back to your chats' : 'Show hidden chats (' + count + ')');
  text(element('history-heading'), showingHidden ? 'Hidden chats' : 'Your conversations');
}
async function restoreConversation(handle) {
  try {
    const {response, data} = await requestJSON('/api/conversations/unhide', {
      method: 'POST', headers: browserAudio.headers({'Content-Type': 'application/json'}),
      body: JSON.stringify({handle})
    });
    if (!response.ok) throw Error(data.error || 'Could not restore this conversation.');
    await loadConversations();
    feedback('Restored to your list.');
  } catch (error) { feedback(error.message, true); }
}
/** Remove a conversation from the list; Ciel keeps its history and can still look it up. */
async function hideConversation(handle, title) {
  if (!window.confirm('Remove “' + title + '” from your list? Ciel keeps it and can still look it up.')) return;
  try {
    const {response, data} = await requestJSON('/api/conversations/hide', {
      method: 'POST', headers: browserAudio.headers({'Content-Type': 'application/json'}),
      body: JSON.stringify({handle})
    });
    if (!response.ok) throw Error(data.error || 'Could not remove this conversation.');
    if (browsingHandle === handle) element('return-active').click();
    await loadConversations();
    feedback('Removed from your list. Ciel can still look it up.');
  } catch (error) { feedback(error.message, true); }
}
async function browseConversation(handle, title) {
  const epoch = ++viewEpoch;
  try {
    const {response, data} = await requestJSON('/api/conversations/' + handle);
    if (epoch !== viewEpoch) return;
    if (!response.ok) throw Error(data.error || 'Could not open this conversation.');
    browsingHandle = handle; historyMessages = data.messages; historyCursor = data.next_before;
    renderTimeline(historyMessages);
    element('archive-banner').hidden = false;
    element('history-earlier').hidden = !historyCursor;
    element('send').disabled = true;
    text(element('chat-title'), title || 'Earlier conversation');
    setDrawer(false);
  } catch (error) { if (epoch === viewEpoch) feedback(error.message, true); }
}
async function activateConversation(handle) {
  if (!activeSelection || runtime.pending) return;
  runtime.pending = true; syncControls();
  try {
    const {response, data} = await requestJSON('/api/conversations/activate', {
      method: 'POST', headers: browserAudio.headers({'Content-Type':'application/json'}),
      body: JSON.stringify({handle, expected_revision:activeSelection.revision})
    });
    if (!response.ok) throw Error(data.error || 'Could not change conversations.');
    browserAudio.stop(); attachments.clear(); element('read-full').hidden = true;
    activeSelection = data; browsingHandle = null; viewEpoch++;
    element('archive-banner').hidden = true; element('history-earlier').hidden = true;
    text(element('chat-title'), 'A little space to think');
    await poll(); await loadConversations(); setDrawer(false);
  } catch (error) { feedback(error.message, true); }
  finally { runtime.pending = false; syncControls(); }
}
element('new-chat').addEventListener('click', () => activateConversation(null));
element('continue-chat').addEventListener('click', () => activateConversation(browsingHandle));
element('return-active').addEventListener('click', () => {
  browsingHandle = null; viewEpoch++; element('archive-banner').hidden = true;
  element('history-earlier').hidden = true; text(element('chat-title'), 'A little space to think');
  poll(); syncControls();
});
element('history-more').addEventListener('click', () => loadConversations(true));
element('history-hidden').addEventListener('click', () => {
  showingHidden = !showingHidden; catalogCursor = null; loadConversations();
});
element('history-earlier').addEventListener('click', async () => {
  if (!browsingHandle || !historyCursor) return;
  const handle = browsingHandle, epoch = viewEpoch, feed = element('messages'), height = feed.scrollHeight;
  try {
    const {response, data} = await requestJSON('/api/conversations/' + handle + '?before=' + historyCursor);
    if (epoch !== viewEpoch || !response.ok) return;
    historyMessages = [...data.messages, ...historyMessages]; historyCursor = data.next_before;
    // Rebuild only for explicit paging; ordinary polling preserves all focused nodes.
    clearNode(feed); messageNodes.clear(); renderTimeline(historyMessages, {followLatest:false});
    feed.scrollTop += feed.scrollHeight - height;
    element('history-earlier').hidden = !historyCursor;
  } catch { feedback('Could not load earlier messages. Try again.', true); }
});
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => {
  element('input').value = button.dataset.prompt || ''; element('input').focus();
});

const browserAudio = new BrowserAudio();
const attachments = new AttachmentComposer();
const cielSettings = new CielSettings();
const mobileUI = typeof MobileUI === 'function' ? new MobileUI() : null;
const liveCaptions = typeof LiveCaptions === 'function' ? new LiveCaptions() : null;
const commandPicker = typeof CommandPicker === 'function'
  ? new CommandPicker(element('input'), element('command-menu')) : null;
const helpPanel = typeof HelpPanel === 'function' && commandPicker
  ? new HelpPanel(element('help-panel'), commandPicker) : null;
element('help-close').addEventListener('click', () => { if (helpPanel) helpPanel.close(); });
element('attach').addEventListener('click', () => element('image-input').click());
element('image-input').addEventListener('change', async event => { await attachments.add(event.target.files); event.target.value=''; });
element('input').addEventListener('paste', event => {
  const files = Array.from(event.clipboardData?.files || []);
  if (files.length) { event.preventDefault(); attachments.add(files); }
});
element('connect-phone').addEventListener('click', () => { setDrawer(false); cielSettings.open('connection'); });
element('settings-open').addEventListener('click', () => { setDrawer(false); cielSettings.open(); });
element('settings-close').addEventListener('click', () => cielSettings.close());
element('settings-form').addEventListener('submit', event => { event.preventDefault(); cielSettings.save(); });
element('network-enable').addEventListener('click', () => cielSettings.enableNetwork());
element('network-code').addEventListener('click', () => cielSettings.pairCode().catch(() => text(element('network-pair-code'),'Could not create a code. Try again.')));
element('network-disconnect').addEventListener('click', () => cielSettings.disconnectPhones().catch(() => text(element('network-pair-code'),'Phones could not be disconnected. Try again.')));
element('memory-refresh').addEventListener('click', () => cielSettings.memories().catch(() => text(element('saved-memory'),'Could not load saved memory.')));
element('server-restart').addEventListener('click', () => cielSettings.restart().catch(() => text(element('settings-notice'),'Server restart could not be requested.')));
if (document.addEventListener) document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {browserAudio.maybeStart();poll();}
});
if(window.addEventListener) {
  window.addEventListener('pagehide',()=>{browserAudio.pageActive=false;browserAudio.stop();});
  window.addEventListener('pageshow',()=>{browserAudio.pageActive=true;browserAudio.maybeStart();});
}
element('record').addEventListener('click', () => browserAudio.toggleRecording());
element('enable-sound').addEventListener('click', () => browserAudio.enable());
element('sound').addEventListener('click', () => {
  browserAudio.muted = !browserAudio.muted;
  if (browserAudio.muted) browserAudio.stop({microphone:false}); else browserAudio.enable();
  text(element('sound'), browserAudio.muted ? 'Sound off' : 'Sound on');
  element('sound').setAttribute('aria-pressed', String(browserAudio.muted));
});
element('pairing').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const {response,data}=await requestJSON('/api/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:element('pair-code').value})});
    if (!response.ok) throw Error(data.error || 'Pairing failed. Check the code on your PC.');
    element('pair-code').value=''; await browserAudio.initialize(); await poll();
  } catch(error) { feedback(error.message,true); }
});
if (document.addEventListener) document.addEventListener('keydown', event => {
  if (helpPanel && helpPanel.opened) {
    if (event.key === 'Escape') { event.preventDefault(); helpPanel.close(); }
    if (event.key === 'Tab') {  // keep focus inside the commands dialog
      const nodes = Array.from(element('help-panel').querySelectorAll('button')).filter(node => !node.hidden && node.getClientRects().length);
      if (nodes.length && event.shiftKey && document.activeElement === nodes[0]) { event.preventDefault(); nodes.at(-1).focus(); }
      else if (nodes.length && !event.shiftKey && document.activeElement === nodes.at(-1)) { event.preventDefault(); nodes[0].focus(); }
    }
    return;
  }
  if (cielSettings.opened) {
    if(event.key==='Escape') {event.preventDefault();cielSettings.close();}
    if(event.key==='Tab') {
      const nodes=Array.from(element('settings-panel').querySelectorAll('button:not([disabled]),input:not([disabled]),select,textarea,a[href]')).filter(node=>!node.hidden && node.getClientRects().length);
      if(nodes.length && event.shiftKey && document.activeElement===nodes[0]) {event.preventDefault();nodes.at(-1).focus();}
      else if(nodes.length && !event.shiftKey && document.activeElement===nodes.at(-1)) {event.preventDefault();nodes[0].focus();}
    }
    return;
  }
  if (!drawerOpen) return;
  if (event.key === 'Escape') { event.preventDefault(); setDrawer(false); }
  if (event.key === 'Tab') {
    const nodes = Array.from(element('sidebar').querySelectorAll('a[href],button:not([disabled])')).filter(node => !node.hidden && node.getClientRects().length);
    if (!nodes.length) return;
    if (event.shiftKey && document.activeElement === nodes[0]) { event.preventDefault(); nodes.at(-1).focus(); }
    else if (!event.shiftKey && document.activeElement === nodes.at(-1)) { event.preventDefault(); nodes[0].focus(); }
  }
});
if (typeof matchMedia === 'function') matchMedia('(max-width:759px)').addEventListener('change',syncDrawer);
syncDrawer();
if (mobileUI) mobileUI.bind();
if (commandPicker && typeof document !== 'undefined' && document.activeElement === element('input')) commandPicker.setOpen(true);
// Installable app with an offline screen; browsers only allow this over HTTPS or on the PC.
if (typeof navigator !== 'undefined' && navigator.serviceWorker && globalThis.isSecureContext)
  navigator.serviceWorker.register('/sw.js').catch(() => {});
poll(); browserAudio.initialize();
setInterval(poll, 350); setInterval(() => attachments.checkExpiry(),1000); setInterval(() => browserAudio.poll(), 500);
