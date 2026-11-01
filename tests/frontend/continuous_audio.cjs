// Continuous microphone sessions segment speech locally and never record Ciel's reply.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
let stopped=0,requests=[],now=1000;
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{hidden:true,textContent:'',setAttribute(){}});return elements.get(id)};
class AudioContext {
  constructor(){this.state='running';this.audioWorklet={addModule:async()=>{}};this.destination={};}
  async resume(){this.state='running';}
  createMediaStreamSource(){return {connect(){},disconnect(){}};}
  createGain(){return {gain:{value:1},connect(){},disconnect(){}};}
}
const stream=()=>({getTracks:()=>[{stop(){stopped++},addEventListener(){}}]});
const context={crypto:require('node:crypto').webcrypto,Float32Array,ArrayBuffer,DataView,
  Date:class extends Date{static now(){return now}},element,text:(node,value)=>node.textContent=value,
  feedback(){},orb:{update(){}},document:{hidden:false},window:{isSecureContext:true,AudioContext},
  navigator:{mediaDevices:{getUserMedia:async()=>stream()}},localStorage:{getItem(){return null},setItem(){}},
  activeSelection:{handle:'chat',revision:1},runtime:{online:true,pending:false,liveMessages:[],activityTime:0},
  browsingHandle:null,setTimeout,clearTimeout,AbortController,
  AudioWorkletNode:class{constructor(){this.port={};}connect(){}disconnect(){}},
  requestJSON:async(path,options)=>{requests.push({path,options});return {response:{ok:true},data:{accepted:true,jobs:[]}}}
};
vm.createContext(context);vm.runInContext(fs.readFileSync('src/ciel/web_assets/audio.js','utf8'),context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const samples=(level,count=320)=>new Float32Array(count).map((_,i)=>level*Math.sin(i*.2));
(async()=>{
  const vad=vm.runInContext('new SpeechSegmenter()',context);
  const feed=(frames,level)=>{const results=[];for(let i=0;i<frames;i++){const r=vad.push(samples(level));if(r)results.push(r);}return results};
  assert.equal(feed(200,.001).length,0,'Room noise must never upload by itself');
  feed(3,.1);assert.equal(feed(60,0).length,0,'A short click must not become a turn');
  feed(25,.08);assert.equal(feed(25,0).length,0,'A natural short pause stays inside the utterance');
  feed(20,.08);const clips=feed(50,0);
  assert.equal(clips.length,1,'Speech is sent automatically after the end pause');
  assert(clips[0].samples>16000,'Keep both phrases and the leading audio');
  assert.equal(feed(100,0).length,0,'Silence after speech cannot send repeated empty turns');
  feed(10,.08);assert.equal(feed(50,0).length,1,'A short yes or no is still an utterance');
  feed(10,.08);const long=feed(3100,.08);
  assert.equal(long.length,1);assert.equal(long[0].samples,960000,'Bound each upload at 60 seconds');

  const audio=vm.runInContext('new BrowserAudio()',context);audio.available=true;
  await audio.toggleRecording();
  const recording=audio.recording;assert(recording,'Enable the microphone once');
  const talk=()=>{for(let i=0;i<25;i++)recording.worklet.port.onmessage({data:samples(.08)});for(let i=0;i<50;i++)recording.worklet.port.onmessage({data:samples(0)});};
  talk();await settle();
  assert.equal(requests.filter(r=>r.path==='/api/voice').length,1,'First utterance auto-sends');
  assert.equal(audio.recording,recording,'Sending must retain the microphone stream');
  assert.equal(stopped,0);
  const upload=requests.find(r=>r.path==='/api/voice').options;
  assert.equal(new DataView(upload.body).getUint32(24,true),16000);
  assert.equal(upload.headers['X-Conversation'],'chat');
  talk();await settle();assert.equal(requests.filter(r=>r.path==='/api/voice').length,1,'Do not overlap turns while a reply is pending');
  audio.syncState({state:'processing',messages:[],activity:[]});
  audio.syncState({state:'idle',messages:[{id:2,role:'assistant',text:'Hello',live:false}],activity:[]});
  now+=1000;
  audio.playbackActive=true;talk();await settle();
  assert.equal(requests.filter(r=>r.path==='/api/voice').length,1,'Never transcribe Ciel playback');
  audio.playbackActive=false;
  const originalRequest=context.requestJSON;let releasePlayback;
  context.requestJSON=(path,options)=>path==='/api/playback'?new Promise(resolve=>releasePlayback=resolve):originalRequest(path,options);
  context.document.hidden=true;
  const metadataPoll=audio.poll();talk();await settle();
  assert.equal(requests.filter(r=>r.path==='/api/voice').length,2,'A second utterance needs no click');
  assert(releasePlayback,'Switching tabs must not deliberately stop playback polling');
  const previousPause=audio.pauseUntil;
  releasePlayback({response:{ok:true},data:{jobs:[]}});await metadataPoll;
  assert.equal(audio.pauseUntil,previousPause,'Empty playback polls must not continually pause the mic');
  context.requestJSON=originalRequest;
  context.document.hidden=false;
  audio.waiting=null;await audio.poll();
  assert(requests.some(r=>r.path==='/api/playback'),'An open mic must not block reply audio polling');
  audio.stop({microphone:false});assert.equal(audio.recording,recording,'Stop reply leaves continuous listening enabled');
  await audio.toggleRecording();assert.equal(audio.recording,null);assert.equal(stopped,1);
  await audio.maybeStart();assert.equal(audio.recording,null,'Explicit microphone pause remains paused');
  talk();await settle();assert.equal(requests.filter(r=>r.path==='/api/voice').length,2,'Late worklet messages cannot send after pausing');

  let release;
  context.navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>release=resolve);
  const pending=audio.toggleRecording();await settle();audio.stop();release(stream());await pending;
  assert.equal(audio.recording,null,'Stop while permission is pending cancels capture');
  assert.equal(stopped,2,'A late microphone stream is immediately released');
  context.navigator.mediaDevices.getUserMedia=async()=>stream();
  await audio.maybeStart();assert(audio.recording,'Reconnecting resumes enabled listening automatically');
  audio.stop();assert.equal(stopped,3);
  const suspended=vm.runInContext('new BrowserAudio()',context);suspended.available=true;
  suspended.context={state:'suspended',resume:()=>new Promise(()=>{})};
  await suspended.maybeStart();assert.equal(suspended.recording,null);
  assert(suspended.permissionBlocked,'A blocked audio context exposes Enable microphone');
  assert.equal(stopped,4,'Blocked automatic capture immediately releases the microphone');
  let prompts=0;
  context.navigator.mediaDevices.getUserMedia=async()=>{prompts++;throw Error('denied')};
  const denied=vm.runInContext('new BrowserAudio()',context);denied.available=true;
  await denied.maybeStart();await denied.maybeStart();
  assert.equal(prompts,1,'A denied permission cannot trigger an automatic prompt loop');
  denied.permissionBlocked=false;denied.pageActive=false;await denied.maybeStart();
  assert.equal(prompts,1,'An unloading page must not reopen its microphone');
  console.log('Automatic speech boundaries, repeated turns, echo suppression, pause, and permission-race checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
