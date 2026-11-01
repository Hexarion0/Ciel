const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{hidden:true,textContent:'',setAttribute(){}});return elements.get(id)};
let warnings=[], starts=0, releaseDecode;
const context={crypto:require('node:crypto').webcrypto,element,feedback:(text)=>warnings.push(text),
 text:(node,value)=>node.textContent=value,orb:{update(){}},
 document:{hidden:false},window:{isSecureContext:true},navigator:{mediaDevices:{getUserMedia:async()=>{throw Error('denied')}}},
 activeSelection:{handle:'chat',revision:1},runtime:{pending:false},browsingHandle:null,
 clearTimeout,setTimeout,AbortController,AudioWorkletNode:class{}};
vm.createContext(context);vm.runInContext(fs.readFileSync('src/ciel/web_assets/audio.js','utf8'),context);
const audio=vm.runInContext('new BrowserAudio()',context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const wave=id=>{const chunk=new ArrayBuffer(1);new Uint8Array(chunk)[0]=id;return chunk;};
function playbackPlayer(id,decode=async chunk=>({id:new Uint8Array(chunk)[0]})) {
 const player=vm.runInContext('new BrowserAudio()',context),sources=[];
 player.available=true;player.ready=true;
 player.context={state:'running',resume:async()=>{player.context.state='running';},
  decodeAudioData:decode,createBufferSource:()=>({
   connect(){},start(){sources.push(this);},stop(){this.onended();}
  })};
 context.requestJSON=async()=>({response:{ok:true},data:{jobs:[{id}]}});
 return {player,sources};
}
async function testPlaybackPrefetch() {
 let fetched=0,releaseNextDecode;
 const decoded=[];
 const {player,sources}=playbackPlayer('prefetch-job',chunk=>{
   const id=new Uint8Array(chunk)[0];decoded.push(id);
   return id===2?new Promise(resolve=>releaseNextDecode=resolve):Promise.resolve({id});
 });
 context.fetch=async()=>{
  fetched++;assert(fetched<=2,'Prefetch must stay one chunk ahead of playback');
  return {status:200,ok:true,arrayBuffer:async()=>wave(fetched)};
 };
 const polling=player.poll();
 await settle();
 assert.equal(sources.length,1,'The first chunk starts without waiting for the next decode');
 assert.equal(fetched,2,'Fetch the next chunk before the first playback ends');
 assert.deepEqual(decoded,[1,2],'Decode the next chunk while the first is still playing');
 player.stop({microphone:false});releaseNextDecode({id:2});await polling;
 await settle();
 assert.equal(sources.length,1,'Stop must discard a late prefetched decode');
}
async function testOrderedPlaybackAndAutoplayRecovery() {
 let fetched=0,decoded=0;
 const {player,sources}=playbackPlayer('ordered-job',async chunk=>{
  decoded++;return {id:new Uint8Array(chunk)[0]};
 });
 context.fetch=async()=>{
  const id=++fetched;
  return id<=2?{status:200,ok:true,arrayBuffer:async()=>wave(id)}:{status:410,ok:false};
 };
 const polling=player.poll();await settle();
 assert.equal(fetched,2,'Only one future chunk is fetched while first playback is held');
 assert.equal(decoded,2,'Prefetched audio is ready before the previous chunk ends');
 assert.equal(sources.length,1,'Prefetch must not overlap audible chunks');
 sources[0].onended();await settle();
 assert.equal(sources.length,2);assert.equal(sources[1].buffer.id,2);
 assert.equal(decoded,2,'Play prefetched audio without decoding it again');
 sources[1].onended();await polling;
 assert.equal(fetched,3,'Stop fetching after the server reports the end');
 assert.equal(player.playing,false);assert.equal(player.playbackActive,false);

 fetched=0;decoded=0;
 const blocked=playbackPlayer('blocked-job',async chunk=>{
  decoded++;return {id:new Uint8Array(chunk)[0]};
 });
 blocked.player.ready=false;blocked.player.context.state='suspended';
 context.fetch=async()=>++fetched===1
  ?{status:200,ok:true,arrayBuffer:async()=>wave(1)}:{status:410,ok:false};
 await blocked.player.poll();
 assert.equal(fetched,1,'Blocked autoplay must not consume a prefetched chunk');
 assert.equal(decoded,0);assert(blocked.player.pendingChunk);
 assert(!blocked.player.seen.has('blocked-job'),'Unplayed jobs remain resumable');
 const enabled=blocked.player.enable();await settle();
 assert.equal(blocked.sources.length,1);blocked.sources[0].onended();await enabled;
 assert.equal(blocked.player.pendingChunk,null);
 await blocked.player.poll();assert.equal(fetched,2);
}
async function testPrefetchFailuresAndAbort() {
 const unhandled=[],onUnhandled=error=>unhandled.push(error);
 process.on('unhandledRejection',onUnhandled);
 try {
  for(const failure of ['fetch','decode','abort']) {
   let fetched=0,aborted=false;
   const {player,sources}=playbackPlayer('failure-'+failure,async chunk=>{
    const id=new Uint8Array(chunk)[0];
    if(id===2 && failure==='decode') throw Error('decode failed');
    return {id};
   });
   context.fetch=async(_url,{signal})=>{
    const id=++fetched;
    if(id===2 && failure==='fetch') throw Error('connection lost');
    if(id===2 && failure==='abort') return new Promise((_resolve,reject)=>{
     signal.addEventListener('abort',()=>{aborted=true;reject(Error('aborted'));});
    });
    return {status:200,ok:true,arrayBuffer:async()=>wave(id)};
   };
   const polling=player.poll();await settle();
   assert.equal(fetched,2);assert.equal(sources.length,1);
   assert.deepEqual(unhandled,[],'Prefetch failures are handled while audio is still playing');
   if(failure==='abort') player.stop({microphone:false});else sources[0].onended();
   await polling;await settle();
   assert.deepEqual(unhandled,[]);assert.equal(sources.length,1);
   if(failure==='abort') assert(aborted,'Stop aborts an outstanding prefetch');
  }
 } finally {process.off('unhandledRejection',onUnhandled);}
}
(async()=>{
 audio.available=true;audio.enable=async()=>{};
 await audio.toggleRecording();
 assert.equal(audio.recording,null);assert(warnings.at(-1).includes('permission'));
 assert.equal(await audio.playChunk(new ArrayBuffer(4)),false);
 assert.equal(element('enable-sound').hidden,false,'Blocked autoplay exposes recovery');
 audio.pendingChunk=null;audio.ready=true;
 audio.context={state:'running',decodeAudioData:()=>new Promise(resolve=>releaseDecode=resolve),
 createBufferSource:()=>({connect(){},start(){starts++},stop(){}})};
 const pending=audio.playChunk(new ArrayBuffer(4));
 audio.stop();releaseDecode({});await pending;
 assert.equal(starts,0,'Stop during decode must prevent late audio');
 let requests=0;
 context.runtime.online=false;audio.available=true;audio.ready=false;audio.pendingChunk=null;
 context.requestJSON=async()=>{requests++;return {response:{ok:true},data:{jobs:[{id:'missed-job'}]}}};
 context.fetch=async()=>{throw Error('Missed audio must not be downloaded automatically')};
 await audio.poll();assert.equal(requests,0,'Disconnected clients never poll for playback');
 context.runtime.online=true;audio.resync=true;
 await audio.poll();assert(audio.seen.has('missed-job'),'Reconnect establishes a fresh baseline');
 await testPlaybackPrefetch();
 await testOrderedPlaybackAndAutoplayRecovery();
 await testPrefetchFailuresAndAbort();
 console.log('Browser playback prefetch, ordered audio, autoplay recovery, cancellation, and failures passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
