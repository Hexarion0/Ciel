// The phone layout always names what Ciel is doing, including how long it has been thinking.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context={console};
vm.createContext(context);
vm.runInContext(fs.readFileSync('src/ciel/web_assets/mobile.js','utf8')+';globalThis.mobileStatus=mobileStatus;',context);
const status=input=>context.mobileStatus({online:true,available:true,recording:true,server:'idle',elapsed:0,...input});

assert.equal(status({online:false}).key,'offline');
// Before the first poll answers, the page is connecting, not offline.
assert.equal(status({online:false,connecting:true}).key,'connecting');
assert.equal(status({online:false,connecting:true,pairing:true}).key,'pairing');
assert.match(status({online:false}).detail,/ciel web/);
assert.equal(status({pairing:true}).key,'pairing');
// An unpaired phone is refused by the server, which looks offline; pairing is the real story.
assert.equal(status({online:false,pairing:true}).key,'pairing');
assert.equal(status({browsing:true}).key,'browsing');
assert.equal(status({starting:true,recording:false}).key,'starting');
assert.equal(status({blocked:true,recording:false}).key,'blocked');
assert.equal(status({}).key,'listening');
assert.equal(status({hearing:true}).key,'hearing');
assert.equal(status({recording:false}).key,'paused');
assert.equal(status({playing:true}).key,'speaking');
assert.equal(status({server:'speaking'}).key,'speaking');
assert.equal(status({waiting:true}).key,'thinking');
assert.equal(status({server:'processing'}).key,'thinking');
assert.equal(status({server:'processing',elapsed:1}).label,'Thinking');
assert.equal(status({server:'processing',elapsed:1}).time,'');
// Seconds stay out of the label so screen readers don't announce every tick.
assert.equal(status({server:'processing',elapsed:12.4}).label,'Thinking');
assert.equal(status({server:'processing',elapsed:12.4}).time,'12s');
assert.equal(status({server:'speaking'}).time,'');
assert.doesNotMatch(status({server:'processing',elapsed:12}).detail,/longer/);
assert.match(status({server:'processing',elapsed:25}).detail,/longer than usual.*stop/i);
// Speaking and thinking win over the microphone state, so the user knows a reply is coming.
assert.equal(status({server:'processing',recording:false}).key,'thinking');
// Without browser audio (PC page), the server's own listening state is shown.
const pc=input=>status({available:false,recording:false,...input});
assert.equal(pc({server:'listening_wake'}).key,'listening');
assert.equal(pc({server:'listening_wake',muted:true}).key,'paused');
assert.equal(pc({server:'recording'}).key,'hearing');
assert.equal(pc({server:'idle'}).key,'ready');
for(const key of ['offline','pairing','browsing','starting','blocked','listening','hearing','paused','speaking','thinking'])
  assert.ok(status({online:key!=='offline',pairing:key==='pairing',browsing:key==='browsing',starting:key==='starting',
    blocked:key==='blocked',recording:['listening','hearing'].includes(key),hearing:key==='hearing',
    playing:key==='speaking',waiting:key==='thinking'}).label,'Every status has a label: '+key);
console.log('Mobile status labels passed.');
