// The service worker only adds an offline screen for app navigations; it never caches or
// intercepts API traffic, so chats and settings always come from the PC.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');

const listeners={};
class FakeResponse { constructor(body,init={}){this.body=body;this.status=init.status??200;this.headers=init.headers||{};} }
let network=async request=>new FakeResponse('live:'+request.url);
const context={
  self:{addEventListener:(type,handler)=>{listeners[type]=handler;},location:{origin:'https://192.168.1.100:8765'},skipWaiting:()=>Promise.resolve(),clients:{claim:()=>Promise.resolve()}},
  fetch:request=>network(request), Response:FakeResponse, URL, caches:undefined,
};
context.self.self=context.self;
vm.createContext(context);
vm.runInContext(fs.readFileSync('src/ciel/web_assets/sw.js','utf8'),context);
assert(listeners.fetch,'A fetch handler is required for the offline screen');
assert(!/caches\./.test(fs.readFileSync('src/ciel/web_assets/sw.js','utf8')),'The service worker must not use Cache Storage');

function dispatch(url,{mode='cors',method='GET'}={}) {
  let responded=null;
  listeners.fetch({request:{url,mode,method},respondWith:value=>{responded=Promise.resolve(value);}});
  return responded;
}

(async()=>{
  const origin='https://192.168.1.100:8765';
  for(const path of ['/api/state','/api/session','/api/message','/assets/chat.js','/sw.js'])
    assert.equal(dispatch(origin+path),null,path+' must go straight to the network');
  assert.equal(dispatch(origin+'/',{mode:'navigate',method:'POST'}),null,'Only GET navigations are handled');
  assert.equal(dispatch('https://example.org/',{mode:'navigate'}),null,'Other sites are never handled');

  const live=await dispatch(origin+'/',{mode:'navigate'});
  assert.equal(live.body,'live:'+origin+'/','Reachable PC serves the live app');

  network=async()=>{throw new TypeError('Failed to fetch');};
  const offline=await dispatch(origin+'/',{mode:'navigate'});
  assert.equal(offline.status,503);
  assert.match(offline.headers['Content-Type'],/text\/html/);
  assert.match(offline.body,/Ciel.s PC isn.t reachable/);
  assert.match(offline.body,/href="\/"/,'The offline screen offers a retry link');
  console.log('Service worker offline screen and pass-through rules passed.');
})().catch(error=>{console.error(error);process.exit(1);});
