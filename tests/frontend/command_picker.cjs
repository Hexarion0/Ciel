// The composer's / picker behaves like the terminal's: arrows, Tab, Enter, Escape, and taps.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');

class Node {
  constructor(){this.children=[];this.attributes={};this.listeners={};this.hidden=false;this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  setAttribute(k,v){this.attributes[k]=String(v);}
  removeAttribute(k){delete this.attributes[k];}
  addEventListener(name,fn){this.listeners[name]=fn;}
  focus(){}
}
const context={
  document:{createElement:()=>new Node()},
  clearNode:node=>{node.children=[];},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('src/ciel/web_assets/commands.js','utf8')+';globalThis.CommandPicker=CommandPicker;',context);

const catalog={
  commands:[{name:'/help',description:'Get help with a command'},{name:'/status',description:'Show your current session'},
    {name:'/stop',description:'Stop the current recording and reply'},{name:'/mute',description:'Toggle or set microphone mute'},
    {name:'/mode',description:'Choose how CIEL listens'},{name:'/memory',description:'Manage saved-memory recall'}],
  arguments:{'/mute':['on','off'],'/mode':['ambient','wake'],'/memory':['status','pause','resume'],'/help':['help','status','stop','mute','mode','memory']},
  spaced:['/help','/memory','/mode','/mute'],
};
const input=new Node(); input.value='';
const menu=new Node();
const picker=new context.CommandPicker(input,menu);
picker.catalog=catalog;
const press=(key,extra={})=>{let prevented=false;const used=picker.key({key,preventDefault:()=>{prevented=true;},...extra});return {used,prevented};};
const type=value=>{input.value=value;picker.render();};
const names=()=>menu.children.filter(c=>c.className==='command-option').map(c=>c.children[0].textContent);

picker.setOpen(true);
type('/m');
assert.deepEqual(names(),['/mute','/mode','/memory']);
assert.equal(menu.hidden,false);
assert.equal(input.attributes['aria-expanded'],'true');
assert.equal(menu.children[0].attributes['aria-selected'],'true');

// Arrows move the selection and wrap around.
press('ArrowDown'); assert.equal(menu.children[1].attributes['aria-selected'],'true');
press('ArrowUp'); press('ArrowUp'); assert.equal(menu.children[2].attributes['aria-selected'],'true');

// Tab completes the selected command and adds a space when it takes options.
assert.deepEqual(press('Tab'),{used:true,prevented:true});
assert.equal(input.value,'/memory ');
assert.deepEqual(names(),['/memory status','/memory pause','/memory resume']);

// Enter completes a partial command instead of sending, then sends it once complete.
type('/sta');
assert.deepEqual(press('Enter'),{used:true,prevented:true});
assert.equal(input.value,'/status');
assert.equal(press('Enter').used,false,'A complete command is sent normally');

// Shift+Enter and ordinary text never trigger the picker.
type('/st'); assert.equal(press('Enter',{shiftKey:true}).used,false);
type('hello /m'); assert.equal(menu.hidden,true); assert.equal(press('Tab').used,false);

// Escape hides the list until the draft changes.
type('/m'); press('Escape'); assert.equal(menu.hidden,true);
type('/mu'); assert.deepEqual(names(),['/mute']);

// A tap completes like Tab.
type('/mo'); menu.children[0].listeners.click(); assert.equal(input.value,'/mode ');

// The list closes when the composer loses focus, and long lists show a count like the terminal.
picker.setOpen(false); assert.equal(menu.hidden,true);
picker.setOpen(true); type('/');
assert.equal(menu.children.at(-1).className,'command-option','Six fit without a count');
catalog.commands.push({name:'/exit',description:'Close CIEL; /quit also works'});
type('/ '); type('/');
assert.match(menu.children.at(-1).textContent,/^1 \/ 7 commands$/);
console.log('Command picker passed.');
