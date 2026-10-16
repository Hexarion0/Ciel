// State-transition checks only: no model inference or simulated human dataset acceptance.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
class Control{
 constructor(dataset={}){this.dataset=dataset;this.value='';this.checked=false;this.disabled=false;
  this.textContent='';this.hidden=false;this.events={};this.children=[];}
 addEventListener(name,fn){this.events[name]=fn;}
 fire(name){this.events[name]?.();}
 pause(){this.paused=true;}
 append(item){this.children.push(item);}
}
function environment(document,saved=new Map()){
 return {document,localStorage:{getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)},
  Blob,URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},setTimeout:fn=>fn()};
}
{
 const ids={};for(const id of ['source','filter','download','progress','save-status','previous',
  'position','next','empty','clip','clip-title','provenance','flags','baseline','raw','prepared',
  'audio-error','baseline-error','caption','asr','transcript','notes','decision','approval-hint',
  'transcript_verified','same_speaker_verified','audio_quality_verified'])ids[id]=new Control();
 ids.filter.value='all';const approve=new Control();ids.decision.querySelector=()=>approve;
 const players=[ids.baseline,ids.raw,ids.prepared];let downloaded='';
 const document={getElementById:id=>ids[id],querySelectorAll:()=>players,
  createElement:type=>type==='a'?{click(){downloaded=this.download;}}:new Control()};
 const settings={manifest_sha256:'test',reviews:{},sources:[{id:'source-one'}],counts:{'source-one':1},
  clips:[{id:'clip-one',source_id:'source-one',source_url:'https://example.test/video',start:1,end:5,
   prepared_seconds:4,audio_sha256:'audio-hash',flags:[],caption:'A candidate transcript.',
   proposal:'A candidate transcript.',asr:'A candidate transcript.',raw:'raw.wav',audio:'clip.wav'}]};
 const script=fs.readFileSync('scripts/tts_finetune/review.html','utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1].replace('__SETTINGS__',JSON.stringify(settings));
 const saved=new Map();vm.runInNewContext(script,environment(document,saved));
 assert.equal(approve.disabled,true);assert.match(ids.progress.textContent,/0 approved/);
 ids.decision.value='approve';ids.decision.fire('change');assert.equal(ids.decision.value,'pending');
 for(const name of ['transcript_verified','same_speaker_verified','audio_quality_verified']){
  ids[name].checked=true;ids[name].fire('change');
 }
 assert.equal(approve.disabled,false);ids.decision.value='approve';ids.decision.fire('change');
 assert.match(ids.progress.textContent,/1 approved/);
 ids.transcript.value='Corrected spoken words.';ids.transcript.fire('input');
 assert.equal(ids.transcript_verified.checked,false);assert.equal(ids.decision.value,'pending');
 assert.match(ids.progress.textContent,/0 approved/);
 ids.filter.value='reject';ids.filter.fire('change');assert.equal(ids.empty.hidden,false);
 ids.filter.value='all';ids.filter.fire('change');assert.equal(ids.empty.hidden,true);
 ids.raw.fire('play');assert.equal(ids.prepared.paused,true);
 ids.prepared.fire('error');assert.equal(ids['audio-error'].hidden,false);
 ids.download.fire('click');assert.equal(downloaded,'ciel-dataset-reviews.json');
 assert.equal(JSON.parse(saved.get('ciel-dataset-review-test'))['clip-one'].decision,'pending');
}
{
 const choice=new Control(),reveal=new Control(),notes=new Control(),status=new Control();
 const samples=['A','B'].map(label=>{
  const sample=new Control({label});sample.identity=new Control();
  sample.ratings=Array.from({length:9},(_,i)=>new Control({rating:`rating${i}`}));
  sample.querySelector=()=>sample.identity;sample.querySelectorAll=()=>sample.ratings;return sample;
 });
 const round=new Control({round:'comparison'});
 round.querySelector=selector=>({'[data-choice]':choice,'[data-reveal]':reveal,'[data-notes]':notes})[selector];
 round.querySelectorAll=selector=>selector==='[data-label]'?samples:samples.flatMap(s=>s.ratings);
 const document={querySelectorAll:selector=>selector==='[data-round]'?[round]:[],
  getElementById:()=>status};
 const settings={run_id:'ratings-test',identities:{comparison:{A:'first engine',B:'second engine'}}};
 const script=fs.readFileSync('scripts/tts_finetune/listening.html','utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1].replace('__SETTINGS__',JSON.stringify(settings));
 const saved=new Map();vm.runInNewContext(script,environment(document,saved));
 choice.value='A';choice.fire('change');assert.equal(reveal.disabled,true);
 reveal.fire('click');assert.equal(samples[0].identity.textContent,'');
 for(const control of samples[0].ratings){control.value='5';control.fire('change');}
 assert.equal(reveal.disabled,true);
 for(const control of samples[1].ratings){control.value='uncertain';control.fire('change');}
 assert.equal(reveal.disabled,false);reveal.fire('click');
 assert.equal(samples[0].identity.textContent,'first engine');
 const snapshot=JSON.parse(saved.get('ciel-quality-shootout-ratings-test')).comparison;
 assert.equal(snapshot.choice_before_reveal,'A');assert.equal(snapshot.ratings_before_reveal.B.rating8,'uncertain');
 samples[0].ratings[0].value='1';samples[0].ratings[0].fire('change');
 assert.equal(JSON.parse(saved.get('ciel-quality-shootout-ratings-test')).comparison.ratings_before_reveal.A.rating0,'5');
}
console.log('Dataset verification and blind-rating gate checks passed.');
