const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map(),app={inert:false};
function element(id){if(!elements.has(id))elements.set(id,{hidden:false,value:'',textContent:'',focus(){this.focused=true},scrollIntoView(){},children:[]});return elements.get(id)}
const fields=[{dataset:{setting:'nim_model'},type:'text',value:''},{dataset:{setting:'tts_enabled'},type:'checkbox',checked:false},{dataset:{setting:'nim_api_key'},type:'password',value:''}];
const snapshot={values:{nim_model:'chosen-model',tts_enabled:true},configured:{nim_api_key:true,groq_api_key:false,openrouter_api_key:false},persona:'',needs_restart:false};
let saved=null,fail=false;
const context={element,text:(node,value)=>node.textContent=value,browserAudio:{device:'pc',headers:v=>v},
 document:{querySelector:()=>app,querySelectorAll:selector=>selector==='.setting-field'?fields:[]},
 requestJSON:async(path,options)=>{
  if(path==='/api/network')return {response:{ok:true},data:{host:'192.168.1.100',enabled:false}};
  if(options){saved=JSON.parse(options.body);return {response:{ok:!fail},data:fail?{error:'Try again'}:{...snapshot,values:{...snapshot.values,...saved}}};}
  return {response:{ok:true},data:snapshot};
 }};
vm.createContext(context);vm.runInContext(fs.readFileSync('src/ciel/web_assets/settings.js','utf8'),context);
const settings=vm.runInContext('new CielSettings()',context);
(async()=>{
 await settings.open();assert(app.inert);assert.equal(fields[0].value,'chosen-model');assert.equal(fields[2].value,'','Saved credentials are never populated');
 fields[0].value='new-model';await settings.save();assert.equal(saved.nim_model,'new-model');assert(!('nim_api_key' in saved),'Blank credentials keep saved keys');
 fail=true;fields[0].value='draft-model';await settings.save();assert.equal(fields[0].value,'draft-model');assert.equal(element('settings-notice').textContent,'Try again');
 settings.close();assert.equal(app.inert,false);assert(element('settings-open').focused);
 console.log('Browser settings load, safe credentials, save, failure draft and focus passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
