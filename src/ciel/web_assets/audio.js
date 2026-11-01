'use strict';
// Sample-count timing stays stable across microphone block sizes and hardware rates.
class SpeechSegmenter {
  constructor() { this.noise=.002; this.reset(); }
  reset() { this.lead=[];this.leadSamples=0;this.run=0;this.active=false;this.chunks=[];this.samples=0;this.voiced=0;this.quiet=0; }
  push(input) {
    let energy=0;for(const sample of input) energy+=sample*sample;
    const rms=Math.sqrt(energy/Math.max(1,input.length)), voice=rms>Math.max(.01,this.noise*3.5);
    if(!this.active) {
      if(!voice) this.noise=.99*this.noise+.01*Math.min(rms,.015);
      this.lead.push(input);this.leadSamples+=input.length;
      while(this.leadSamples>4000) {
        const excess=this.leadSamples-4000, first=this.lead[0];
        if(first.length<=excess) {this.lead.shift();this.leadSamples-=first.length;}
        else {this.lead[0]=first.slice(excess);this.leadSamples-=excess;}
      }
      this.run=voice?this.run+input.length:0;
      if(this.run>=960) {
        this.active=true;this.chunks=this.lead;this.samples=this.leadSamples;
        this.voiced=this.run;this.lead=[];this.leadSamples=0;
      }
      return null;
    }
    const chunk=input.slice(0,960000-this.samples);
    this.chunks.push(chunk);this.samples+=chunk.length;
    if(voice) {this.voiced+=chunk.length;this.quiet=0;} else this.quiet+=chunk.length;
    if(this.quiet<14400 && this.samples<960000) return null;
    const result=this.voiced>=2240?{chunks:this.chunks,samples:this.samples}:null;
    this.reset();return result;
  }
}
class BrowserAudio {
  constructor() {
    this.context = null; this.recording = null; this.generation = 0; this.playing = false;
    this.available = false; this.muted = false; this.csrf = ''; this.device = '';
    this.client = crypto.randomUUID().replaceAll('-', '');
    this.seen = new Set(); this.ready = false; this.source = null; this.pendingChunk = null;
    this.micGeneration=0;this.starting=false;this.sending=false;this.waiting=null;
    this.playbackActive=false;this.playbackAbort=null;
    this.serverState='idle';this.pauseUntil=0;this.permissionBlocked=false;this.workletReady=false;
    this.pageActive=true;
    try {this.listeningWanted=localStorage.getItem('ciel-listening')!=='off';} catch {this.listeningWanted=true;}
  }
  headers(extra = {}) { return {'X-CSRF-Token':this.csrf, ...extra}; }
  async initialize() {
    try {
      const {data} = await requestJSON('/api/session');
      element('pairing').hidden = data.paired;
      this.csrf = data.csrf || ''; this.device = data.device_id || '';
      this.available = Boolean(data.browser_audio && data.paired);
      element('connect-phone').hidden = data.device_id !== 'pc';
      element('mute').hidden = data.pc_capture === false;
      element('exit').hidden = data.pc_capture === false;
      element('record').hidden = !this.available;
      element('sound').hidden = !this.available;
      if (this.available) {
        // Mark existing playback on page load so reconnecting never replays an old reply.
        const {data:existing} = await requestJSON('/api/playback');
        for (const job of existing.jobs || []) this.seen.add(job.id);
        await this.maybeStart();
      }
    } catch { /* Main poll shows connection errors. */ }
  }
  async enable({automatic=false}={}) {
    try {
      if (!this.context) this.context = new (window.AudioContext || window.webkitAudioContext)();
      this.ready=false;
      if(automatic && this.context.state!=='running') {
        let deadline;
        try {await Promise.race([this.context.resume(),new Promise(resolve=>deadline=setTimeout(resolve,250))]);}
        finally {clearTimeout(deadline);}
      } else await this.context.resume();
      this.ready = this.context.state === 'running';
      if (this.ready) { element('enable-sound').hidden = true; if (this.pendingChunk) { const chunk=this.pendingChunk; this.pendingChunk=null; await this.playChunk(chunk); } }
    } catch {if(!automatic) feedback('Sound is blocked. Use Enable sound or read the reply.', true);}
  }
  async toggleRecording() {
    if(this.recording || this.starting) {
      this.listeningWanted=false;this.stopCapture();
      try {localStorage.setItem('ciel-listening','off');} catch {}
      feedback('Microphone paused.');return;
    }
    this.listeningWanted=true;this.permissionBlocked=false;
    try {localStorage.setItem('ciel-listening','on');} catch {}
    await this.startListening(false);
  }
  async maybeStart() {
    if(this.recording && this.context?.state!=='running') this.stopCapture();
    if(this.listeningWanted && !this.permissionBlocked) await this.startListening(true);
  }
  async startListening(automatic) {
    if(this.recording || this.starting || !this.pageActive || !this.available || !activeSelection || runtime.pending || browsingHandle || runtime.online===false) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      this.permissionBlocked=true;feedback('Microphone access needs local HTTPS. You can still type.', true); return;
    }
    const generation=++this.micGeneration, selection={...activeSelection};
    this.starting=true;this.showMicStatus();
    let stream;
    try {
      if(!automatic) await this.enable();
      if(generation!==this.micGeneration || selection.revision!==activeSelection?.revision) return;
      stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      if(generation!==this.micGeneration || selection.revision!==activeSelection?.revision) {stream.getTracks().forEach(track=>track.stop());return;}
      if(automatic) await this.enable({automatic:true});
      const context=this.context;
      if(!context || context.state!=='running') {
        stream.getTracks().forEach(track=>track.stop());this.permissionBlocked=true;
        feedback('Enable microphone once, then Ciel keeps listening.');return;
      }
      if(!this.workletReady) {await context.audioWorklet.addModule('/assets/pcm-worklet.js');this.workletReady=true;}
      if(generation!==this.micGeneration || selection.revision!==activeSelection?.revision) {stream.getTracks().forEach(track=>track.stop());return;}
      const worklet = new AudioWorkletNode(context, 'ciel-pcm');
      const source = context.createMediaStreamSource(stream), silent = context.createGain(); silent.gain.value=0;
      source.connect(worklet); worklet.connect(silent); silent.connect(context.destination);
      const recording={stream,context,worklet,source,silent,selection,vad:new SpeechSegmenter()};
      worklet.port.onmessage=event=>this.processSamples(recording,event.data);
      this.recording = recording;
      for(const track of stream.getTracks()) track.addEventListener('ended',()=>{
        if(this.recording!==recording) return;
        this.permissionBlocked=true;this.stopCapture();feedback('Microphone disconnected. Enable it again or type.',true);
      });
      feedback('Listening continuously. Speak naturally; pause to send.');
    } catch {
      if (stream) stream.getTracks().forEach(track=>track.stop());
      if(generation===this.micGeneration) {
        this.permissionBlocked=true;
        feedback('Microphone permission is unavailable. Allow it in your browser or type a message.',true);
      }
    } finally {
      if(generation===this.micGeneration) this.starting=false;
      this.showMicStatus();
    }
  }
  stopCapture() {
    this.micGeneration++;this.starting=false;
    const recording=this.recording;this.recording=null;
    this.showMicStatus();
    if (!recording) return;
    recording.stream.getTracks().forEach(track=>track.stop());
    recording.source.disconnect(); recording.worklet.disconnect(); recording.silent.disconnect();
  }
  captureBlocked() {
    return !this.pageActive || runtime.online===false || runtime.pending || browsingHandle || this.sending || this.waiting || this.playbackActive || this.source || this.pendingChunk || ['processing','speaking'].includes(this.serverState) || Date.now()<this.pauseUntil;
  }
  processSamples(recording, chunk) {
    if(this.recording!==recording) return;
    if(recording.selection.revision!==activeSelection?.revision || this.captureBlocked()) {recording.vad.reset();return;}
    const utterance=recording.vad.push(chunk);
    if(utterance) this.sendVoice(recording,utterance);
  }
  async sendVoice(recording, utterance) {
    if(this.recording!==recording || this.captureBlocked()) return;
    this.sending=true;
    const waiting={since:Date.now(),baseline:runtime.liveMessages?.at(-1)?.id || 0,activity:runtime.activityTime || 0,busy:false};
    this.waiting=waiting;this.showMicStatus();
    const buffer = new ArrayBuffer(44+utterance.samples*2), view = new DataView(buffer);
    const word = (offset,value)=>{ for(let i=0;i<value.length;i++) view.setUint8(offset+i,value.charCodeAt(i)); };
    word(0,'RIFF'); view.setUint32(4,buffer.byteLength-8,true); word(8,'WAVE'); word(12,'fmt ');
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
    view.setUint32(24,16000,true); view.setUint32(28,32000,true); view.setUint16(32,2,true); view.setUint16(34,16,true);
    word(36,'data'); view.setUint32(40,utterance.samples*2,true);
    let offset=44;
    for(const chunk of utterance.chunks) for(const value of chunk) { view.setInt16(offset,Math.round(Math.max(-1,Math.min(1,value))*32767),true); offset+=2; }
    try {
      const {response,data}=await requestJSON('/api/voice',{method:'POST',headers:this.headers({
        'Content-Type':'audio/wav','X-Request-ID':crypto.randomUUID().replaceAll('-',''),
        'X-Revision':String(recording.selection.revision),'X-Conversation':recording.selection.handle
      }),body:buffer});
      if (!response.ok || !data.accepted) throw Error(data.error || 'Could not send the recording. Try again.');
      feedback('Heard you.');
    } catch(error) {if(this.waiting===waiting) this.waiting=null;feedback(error.message,true);this.pauseUntil=Date.now()+1000;}
    finally {this.sending=false;this.showMicStatus();}
  }
  syncState(snapshot) {
    this.serverState=snapshot.state;
    if(browsingHandle && this.recording) this.stopCapture();
    if(this.waiting) {
      const busy=['processing','speaking'].includes(snapshot.state);
      if(busy) this.waiting.busy=true;
      const answered=(snapshot.messages || []).some(message=>message.id>this.waiting.baseline && message.role==='assistant' && !message.live);
      const activity=(snapshot.activity || []).at(-1);
      if(!busy && (this.waiting.busy || answered || activity?.time>this.waiting.activity || Date.now()-this.waiting.since>90000)) {this.waiting=null;this.pauseUntil=Date.now()+600;}
    }
    this.maybeStart();this.showMicStatus();
  }
  showMicStatus() {
    const on=Boolean(this.recording || this.starting);
    element('record').textContent=on?'Pause mic':'Enable microphone';
    element('record').setAttribute('aria-pressed',String(on));
    if(!this.available) return;
    element('core').setAttribute('aria-label',on?'Pause microphone':'Enable microphone');
    element('core').setAttribute('aria-pressed',String(on));
    text(element('orb-action'),on?'Pause mic':'Enable microphone');
    if(this.starting) {text(element('core-state'),'Enabling microphone…');orb.update('processing',false);}
    else if(this.source) {text(element('core-state'),'Speaking on this device.');orb.update('speaking',false);}
    else if(this.waiting || this.serverState==='processing') {text(element('core-state'),'Thinking.');orb.update('processing',false);}
    else if(browsingHandle) {text(element('core-state'),'Microphone paused.');orb.update('idle',true);}
    else if(this.recording && !this.captureBlocked()) {text(element('core-state'),this.recording.vad.active?'Listening to you.':'Listening.');orb.update(this.recording.vad.active?'recording':'listening_wake',false);}
    else if(!on) {text(element('core-state'),'Microphone paused.');orb.update('idle',true);}
    text(element('state-hint'),browsingHandle?'Continue this chat or return to the current chat to speak.':on?'Speak naturally. Ciel sends when you pause; the mic pauses during replies.':'Enable the microphone once for continuous listening. You can also type.');
  }
  stop({microphone=true}={}) {
    this.generation++;if(microphone) this.stopCapture();
    if(this.recording) this.recording.vad.reset();
    this.waiting=null;this.pauseUntil=Date.now()+600;
    if(this.playbackAbort) this.playbackAbort.abort();
    if (this.source) { try { this.source.stop(); } catch {} this.source=null; }
    this.pendingChunk=null;
  }
  async playChunk(chunk, {decoded=null,onStart=null}={}) {
    if (!this.ready || !this.context || this.context.state !== 'running') {
      this.pendingChunk=chunk; element('enable-sound').hidden=false; return false;
    }
    const generation=this.generation;
    const audio=decoded || await this.context.decodeAudioData(chunk);
    if (this.muted || generation!==this.generation) return true;
    await new Promise(resolve=>{
      const source=this.context.createBufferSource(); this.source=source;
      source.buffer=audio; source.connect(this.context.destination); source.onended=resolve;
      source.start();if(onStart) onStart();
      orb.update('speaking',false); text(element('core-state'),'Speaking on this device.');
    });
    this.source=null;this.pauseUntil=Date.now()+600;return true;
  }
  async poll() {
    if (!runtime.online) { this.resync=true; return; }
    if (!this.available || this.playing || this.pendingChunk) return;
    this.playing=true;
    try {
      const {response,data}=await requestJSON('/api/playback');
      if (!response.ok) return;
      if (this.resync) {
        for (const job of data.jobs || []) this.seen.add(job.id);
        this.resync=false; return;
      }
      for(const job of data.jobs || []) {
        if (this.seen.has(job.id)) continue;
        this.seen.add(job.id);
        if (job.error) { feedback(job.error,true); continue; }
        if (job.excerpt) {
          element('read-full').hidden=false;
          element('read-full').onclick=async()=>{
            await this.enable();
            await requestJSON('/api/replay',{method:'POST',headers:this.headers({'Content-Type':'application/json'}),body:JSON.stringify({job:job.id})});
          };
        }
        if (this.muted) continue;
        this.playbackActive=true;
        if(this.recording) this.recording.vad.reset();
        const generation=this.generation;
        let next=null;
        const readChunk=async()=>{
          while(generation===this.generation && !this.muted) {
            const controller=new AbortController(), deadline=setTimeout(()=>controller.abort(),8000);
            this.playbackAbort=controller;
            let reply,chunk;
            try {
              reply=await fetch('/api/audio/'+job.id+'?client='+this.client,{signal:controller.signal});
              if(reply.ok && reply.status!==204) chunk=await reply.arrayBuffer();
            } finally {
              clearTimeout(deadline);if(this.playbackAbort===controller) this.playbackAbort=null;
            }
            if(generation!==this.generation || this.muted) return null;
            if(reply.status===204) {await new Promise(resolve=>setTimeout(resolve,150));continue;}
            if(!reply.ok) return null;
            const decoded=this.ready && this.context?.state==='running'
              ? await this.context.decodeAudioData(chunk) : null;
            return generation===this.generation && !this.muted?{chunk,decoded}:null;
          }
          return null;
        };
        // Keep one fetch/decode ahead; handle its rejection even during playback or Stop.
        const prefetch=()=>readChunk().then(value=>({value}),error=>({error}));
        try {
          next=prefetch();
          while(generation===this.generation) {
            const {value,error}=await next;next=null;
            if(error) throw error;
            if(!value || generation!==this.generation) break;
            const played=await this.playChunk(value.chunk,{
              decoded:value.decoded,onStart:()=>{next=prefetch();}
            });
            if(!played) {this.seen.delete(job.id);break;}
            if(!next) break;
          }
        } finally {
          if(next && this.playbackAbort) this.playbackAbort.abort();
        }
      }
    } catch { /* Leave transcript available when playback loses its connection. */ }
    finally {if(this.playbackActive) this.pauseUntil=Date.now()+600;this.playbackActive=false;this.playing=false;}
  }
}
