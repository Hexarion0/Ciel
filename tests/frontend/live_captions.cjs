// Spoken replies are revealed in step with the voice on the playing device; others see full text.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context={};
vm.createContext(context);
vm.runInContext(fs.readFileSync('src/ciel/web_assets/captions.js','utf8')+';globalThis.LiveCaptions=LiveCaptions;',context);
let now=100000;
const captions=new context.LiveCaptions(()=>now);
const reply=(id,extra={})=>({id,role:'assistant',spoken:true,time:now/1000,text:'Full **formatted** reply.',...extra});
const CAPTION='Hello there, my friend.';

// Waiting for the voice: the reply is held back instead of appearing all at once.
const first=reply(1);
captions.observe(first,{audible:true});
assert.equal(captions.text(first),'');
assert.equal(captions.active(),true);

// Each chunk reveals its characters as it plays, never rewinding.
assert.equal(captions.begin(1,CAPTION),true);
captions.segment(1,0,10,1);
assert.equal(captions.text(first),'');
now+=500; assert.equal(captions.text(first),CAPTION.slice(0,5));
now+=500; assert.equal(captions.text(first),CAPTION.slice(0,10));
now+=200; assert.equal(captions.text(first),CAPTION.slice(0,10),'Holds between chunks');
captions.segment(1,8,CAPTION.length,2);
assert.equal(captions.text(first),CAPTION.slice(0,10),'A later chunk never hides revealed words');
now+=2000; assert.equal(captions.text(first),CAPTION);

// When playback ends, the full formatted reply returns.
captions.finish(1);
assert.equal(captions.text(first),null);
assert.equal(captions.active(),false);

// Not playing sound on this device, or an old message from history: show it all at once.
const muted=reply(2); captions.observe(muted,{audible:false}); assert.equal(captions.text(muted),null);
const old=reply(3); captions.observe(old,{audible:true,fresh:false}); assert.equal(captions.text(old),null,'History at load is shown in full');
// No clock comparison: a reply stamped by a PC clock far ahead or behind still captions.
const skewed=reply(8,{time:(now+600000)/1000}); captions.observe(skewed,{audible:true}); assert.equal(captions.text(skewed),'');
captions.finish(8);
const typed={id:4,role:'assistant',time:now/1000,text:'Typed'}; captions.observe(typed,{audible:true}); assert.equal(captions.text(typed),null);

// Another device's reply: no audio job arrives here, so it appears after a short check.
const elsewhere=reply(5); captions.observe(elsewhere,{audible:true});
captions.listed([]); assert.equal(captions.text(elsewhere),'','Too early to tell');
now+=1100; captions.listed([]); assert.equal(captions.text(elsewhere),null);

// A job listed for this device keeps waiting, but never hides the answer for over 30 seconds.
const slow=reply(6); captions.observe(slow,{audible:true});
now+=2000; captions.listed([{message:6}]); assert.equal(captions.text(slow),'');
now+=30000; assert.equal(captions.text(slow),null);

// Stop or Sound off finishes every caption.
const stopped=reply(7); captions.observe(stopped,{audible:true}); captions.begin(7,CAPTION);
captions.finishAll(); assert.equal(captions.text(stopped),null);
// The audio poll can see a job before the chat poll draws its message: keep that state.
const early=reply(9); captions.begin(9,CAPTION); captions.segment(9,0,10,1); now+=1000;
captions.observe(early,{audible:true}); assert.equal(captions.text(early),CAPTION.slice(0,10),'Early playback keeps captioning');
captions.finish(9);
const finishedEarly=reply(10); captions.begin(10,CAPTION); captions.finish(10);
captions.observe(finishedEarly,{audible:true}); assert.equal(captions.text(finishedEarly),null,'Finished before drawn shows full text');
// Jobs without a message (replays of untracked text) never create captions.
assert.equal(captions.begin(null,CAPTION),false);
// Old finished captions are pruned so the map stays small.
for(let id=100;id<300;id++){captions.observe(reply(id),{audible:false});}
assert.ok(captions.items.size<=120,'Caption memory is bounded');
console.log('Live captions passed.');
