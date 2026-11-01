const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
let fail=false, revision=1, uploads=0, now=0;
const previews={children:[],textContent:'',get firstElementChild(){return this.children[0]},append(node){this.children.push(node)}};
const elements={'attachments':previews,'attach':{disabled:false}};
const context={Date:{now:()=>now},URL:{createObjectURL:()=> 'blob:preview',revokeObjectURL:()=>{}},crypto:{randomUUID:()=> '1234567890123456'},
 document:{createElement:tag=>({tagName:tag,children:[],append(...nodes){this.children.push(...nodes)},addEventListener(){},remove(){},setAttribute(){}})},
 element:id=>elements[id],clearNode:n=>{n.children=[];n.textContent=''},feedback(){},
 fetch:async()=>{uploads++;return {ok:!fail,json:async()=>fail?{error:'Failed'}:{handle:'uploaded-'+uploads,expires_in:900}}},
 browserAudio:{headers:value=>value},get activeSelection(){return {handle:'chat',revision}},
};
vm.createContext(context);vm.runInContext(fs.readFileSync('src/ciel/web_assets/attachments.js','utf8'),context);
const composer=vm.runInContext('new AttachmentComposer()',context);
(async()=>{
 const image={type:'image/png',size:42};
 await composer.add([image]);assert.deepEqual(Array.from(composer.admittedIds()),['uploaded-1']);
 fail=true;await composer.add([image]);assert.equal(composer.admittedIds().length,1,'Failed upload is not admitted');
 composer.clear();assert.equal(composer.admittedIds().length,0);
 fail=false;await composer.add([image,image,image]);assert.equal(composer.admittedIds().length,0,'More than two images are rejected before uploads');
 await composer.add([{type:'application/pdf',size:42}]);assert.equal(composer.admittedIds().length,0);
 await composer.add([{type:'image/png',size:11*1024*1024}]);assert.equal(composer.admittedIds().length,0);
 await composer.add([image]);now=901000;composer.checkExpiry();
 assert.equal(composer.admittedIds().length,0,'Expired images cannot be sent');
 assert.equal(composer.items[0].expired,true,'Expired preview explains reupload');
 console.log('Attachment admission, limits, errors and cleanup passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
