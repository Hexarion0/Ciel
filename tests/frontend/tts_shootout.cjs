const fs=require('fs');
const vm=require('vm');
const assert=require('assert/strict');
class Control {
 constructor(dataset={}){this.dataset=dataset;this.value='';this.disabled=false;this.textContent='';this.events={};this.paused=false;}
 addEventListener(name,fn){this.events[name]=fn;}
 fire(name){this.events[name]?.();}
 pause(){this.paused=true;}
}
const choice=new Control(),reveal=new Control(),field=new Control();
const samples=['A','B','C'].map(label=>{
 const sample=new Control({label});sample.identity=new Control();sample.rating=new Control({rating:'naturalness'});
 sample.querySelector=selector=>selector==='.identity'?sample.identity:null;
 sample.querySelectorAll=()=>[sample.rating];return sample;
});
const round=new Control({round:'main-short'});
round.querySelector=selector=>({'[data-choice]':choice,'[data-reveal]':reveal,'[data-notes]':field})[selector];
round.querySelectorAll=()=>samples;
const players=[new Control(),new Control()];
for(const player of players){player.error={hidden:true};player.parentElement={querySelector:()=>player.error};}
const exportButton=new Control(),status=new Control(),saved=new Map();
let download;
const document={
 querySelectorAll:selector=>selector==='[data-round]'?[round]:players,
 getElementById:id=>id==='export'?exportButton:status,
 createElement:()=>({click(){download=this.download;}}),
};
const settings={run_id:'test',identities:{'main-short':{A:'engine one',B:'engine two',C:'engine three'}}};
const template=fs.readFileSync('scripts/tts_shootout/listening.html','utf8');
const script=template.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__SETTINGS__',JSON.stringify(settings));
vm.runInNewContext(script,{document,localStorage:{getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)},
 Blob,URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},setTimeout:fn=>fn()});
assert.equal(reveal.disabled,true);
reveal.fire('click');assert.equal(samples[0].identity.textContent,'');
samples[0].rating.value='5';samples[0].rating.fire('change');
choice.value='B';choice.fire('change');assert.equal(reveal.disabled,false);
reveal.fire('click');assert.equal(samples[0].identity.textContent,'engine one');
const snapshot=JSON.parse(saved.get('ciel-quality-shootout-test'))['main-short'];
assert.equal(snapshot.choice_before_reveal,'B');assert.equal(snapshot.ratings_before_reveal.A.naturalness,'5');
samples[0].rating.value='2';samples[0].rating.fire('change');
assert.equal(JSON.parse(saved.get('ciel-quality-shootout-test'))['main-short'].ratings_before_reveal.A.naturalness,'5');
players[0].fire('play');assert.equal(players[1].paused,true);
players[0].fire('error');assert.equal(players[0].error.hidden,false);
exportButton.fire('click');assert.equal(download,'ciel-quality-notes-test.json');
console.log('Blind shootout interaction checks passed.');
