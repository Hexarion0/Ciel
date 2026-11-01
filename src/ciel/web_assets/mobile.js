'use strict';
// Phone layout (max-width 759px): switchable voice and chat modes, and a status that always
// says what Ciel is doing. The new buttons forward to the existing controls, so all audio,
// pairing, and request logic stays in audio.js and chat.js. Phone styles apply only once
// bind() marks the page with .mobile-ui, so a missing script leaves the plain layout working.

const MOBILE_QUERY = '(max-width:759px)';
const SLOW_THINKING_SECONDS = 20;

/** Describe Ciel's current activity for the phone status pill and voice screen. */
function mobileStatus(s) {
  const show = (key, label, detail, time = '') => ({key, label, detail, time});
  if (s.pairing) return show('pairing', 'Not paired', 'Enter the pairing code from your PC.');
  if (!s.online) return s.connecting
    ? show('connecting', 'Connecting', "Reaching Ciel's PC…")
    : show('offline', 'Offline', "Can't reach Ciel's PC. Check that ciel web is running.");
  if (s.browsing) return show('browsing', 'Earlier chat', 'Continue this chat to talk about it.');
  if (s.playing || s.server === 'speaking') return show('speaking', 'Speaking', 'Tap stop to interrupt.');
  if (s.waiting || s.server === 'processing') {
    const seconds = Math.floor(s.elapsed || 0);
    return show('thinking', 'Thinking',
      seconds >= SLOW_THINKING_SECONDS
        ? 'This is taking longer than usual. Tap stop to cancel.' : 'Working on a reply.',
      seconds >= 3 ? seconds + 's' : '');
  }
  if (!s.available) {
    if (s.server === 'recording') return show('hearing', 'Hearing you', 'Ciel replies when you pause.');
    if (s.server === 'listening_wake') return s.muted
      ? show('paused', 'Mic paused', 'Tap the mic to listen again, or type.')
      : show('listening', 'Listening', 'Say the wake phrase, or type.');
    return show('ready', 'Ready', 'Type a message.');
  }
  if (s.starting) return show('starting', 'Starting mic…', 'Allow the microphone if asked.');
  if (s.blocked) return show('blocked', 'Mic unavailable', 'Allow the microphone, or type instead.');
  if (s.hearing) return show('hearing', 'Hearing you', 'Ciel replies when you pause.');
  if (s.recording) return show('listening', 'Listening', 'Just talk. Ciel replies when you pause.');
  return show('paused', 'Mic paused', 'Tap the mic to talk, or type.');
}

/** Write an attribute only when it changes; aria-pressed also drives the icon CSS. */
function setAttr(node, name, value) {
  if (node.getAttribute(name) !== value) node.setAttribute(name, value);
}
function setFlag(node, name, value) { if (node[name] !== value) node[name] = value; }

class MobileUI {
  constructor() {
    this.thinkingSince = null;
    try { this.mode = localStorage.getItem('ciel-mobile-mode') === 'chat' ? 'chat' : 'voice'; }
    catch { this.mode = 'voice'; }
  }
  get phone() { return typeof matchMedia === 'function' && matchMedia(MOBILE_QUERY).matches; }
  /** Where the orb lives on a phone: center stage in voice mode, the header in chat mode. */
  orbSlot() { return element(this.mode === 'voice' ? 'voice-orb' : 'header-orb'); }
  setMode(mode) {
    this.mode = mode === 'chat' ? 'chat' : 'voice';
    try { localStorage.setItem('ciel-mobile-mode', this.mode); } catch { /* per-device only */ }
    this.applyMode();
    if (this.mode === 'chat' && this.phone) element('input').focus();
  }
  applyMode() {
    const voice = this.mode === 'voice';
    document.body.classList.toggle('mode-voice', voice);
    document.body.classList.toggle('mode-chat', !voice);
    setAttr(element('m-mode'), 'aria-pressed', String(!voice));
    if (typeof syncDrawer === 'function') syncDrawer();
    this.sync();
  }
  toggleMic() {
    if (browserAudio.available) browserAudio.toggleRecording(); else submit('/mute');
  }
  sync() {
    if (!this.phone || document.hidden) return;
    const audio = browserAudio, body = document.body;
    const server = body.dataset.state;
    const playing = Boolean(audio.source || audio.playbackActive);
    const state = mobileStatus({
      online: runtime.online, connecting: !runtime.online && !body.classList.contains('offline'),
      pairing: !element('pairing').hidden, browsing: Boolean(browsingHandle),
      available: audio.available, starting: audio.starting,
      blocked: audio.permissionBlocked && !audio.recording, recording: Boolean(audio.recording),
      hearing: Boolean(audio.recording && audio.recording.vad && audio.recording.vad.active),
      playing, waiting: Boolean(audio.waiting), server,
      muted: body.classList.contains('muted'), elapsed: this.elapsed(),
    });
    this.track(state.key);
    element('m-status').dataset.status = state.key;
    element('voice-stage').dataset.status = state.key;
    text(element('m-status-label'), state.label);
    text(element('m-status-time'), state.time ? ' · ' + state.time : '');
    text(element('voice-status'), state.label);
    text(element('voice-time'), state.time);
    text(element('voice-detail'), state.detail);
    this.syncTranscript();
    // Stop follows the work itself, so a running reply can be stopped even from an archive.
    this.syncButtons(playing || audio.waiting || ['processing', 'speaking'].includes(server));
  }
  elapsed() { return this.thinkingSince === null ? 0 : (Date.now() - this.thinkingSince) / 1000; }
  track(key) {
    if (key !== 'thinking') this.thinkingSince = null;
    else if (this.thinkingSince === null) this.thinkingSince = Date.now();
  }
  syncTranscript() {
    const messages = runtime.liveMessages || [];
    const last = role => [...messages].reverse().find(message => message.role === role);
    const heard = last('user'), reply = last('assistant');
    const fresh = reply && (!heard || reply.id > heard.id);
    text(element('voice-heard'), heard ? '“' + heard.text + '”' : '');
    text(element('voice-reply'), fresh ? reply.text : '');
  }
  syncButtons(busy) {
    const audio = browserAudio, mute = element('mute');
    const listening = audio.available ? Boolean(audio.recording || audio.starting)
      : !document.body.classList.contains('muted');
    // Without browser audio the mic button controls the PC microphone, like #mute.
    const micOff = !runtime.online || Boolean(browsingHandle)
      || (!audio.available && (mute.hidden || mute.disabled));
    for (const id of ['m-mic', 'm-chat-mic']) {
      setFlag(element(id), 'disabled', micOff);
      setAttr(element(id), 'aria-pressed', String(listening));
    }
    const stop = element('stop');
    setFlag(element('m-stop'), 'disabled', !busy || stop.disabled);
    setFlag(element('m-chat-stop'), 'hidden', !busy);
    setFlag(element('m-chat-stop'), 'disabled', stop.disabled);
    setFlag(element('m-attach'), 'disabled', element('attach').disabled);
    const sound = element('sound');
    setFlag(element('m-sound'), 'hidden', sound.hidden);
    setAttr(element('m-sound'), 'aria-pressed', sound.getAttribute('aria-pressed') || 'false');
  }
  bind() {
    element('m-mode').addEventListener('click', () => this.setMode(this.mode === 'voice' ? 'chat' : 'voice'));
    element('m-to-chat').addEventListener('click', () => this.setMode('chat'));
    for (const id of ['m-mic', 'm-chat-mic']) element(id).addEventListener('click', () => this.toggleMic());
    for (const id of ['m-stop', 'm-chat-stop']) element(id).addEventListener('click', () => element('stop').click());
    element('m-attach').addEventListener('click', () => element('attach').click());
    element('m-sound').addEventListener('click', () => { element('sound').click(); this.sync(); });
    this.applyMode();
    document.documentElement.classList.add('mobile-ui');
    setInterval(() => this.sync(), 250);
  }
}
