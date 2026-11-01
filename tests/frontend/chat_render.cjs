// Supported Markdown: paragraphs, lists, fenced code, inline code, emphasis, safe links.
// Raw HTML stays literal. Nested block structures and tables intentionally stay plain text.
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
class Node {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.dataset = {}; this.listeners = {}; this._text = ''; }
  get textContent() { return this._text + this.children.map(n => n.textContent).join(''); }
  set textContent(value) { this._text = value; this.children = []; }
  set innerHTML(_) { throw Error('HTML injection'); }
  append(...nodes) { for (const n of nodes) { n.parent = this; this.children.push(n); } }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  get firstElementChild() { return this.children[0]; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  setAttribute(key, value) { this[key] = value; }
}
let copied;
const context = { URL, document: {createElement: tag => new Node(tag)}, navigator: {clipboard: {writeText: async text => {copied = text;}}} };
vm.createContext(context);
vm.runInContext(fs.readFileSync('src/ciel/web_assets/render.js', 'utf8'), context);
const body = new Node(), sources = new Node();
context.body = body; context.sources = sources;
vm.runInContext('renderMessageContent(body, "Hello <script>alert(1)</script>\\n\\n- One\\n- Two\\n\\n```js\\nconst n = 1;\\nnext();\\n```\\n[Good](https://example.com) [Bad](javascript:alert(1))")', context);
assert(body.textContent.includes('<script>alert(1)</script>'));
assert(body.children.some(n => n.tagName === 'ul'));
const pre = body.children.find(n => n.tagName === 'div');
assert(pre, 'Code block has a Copy action');
(async () => {
  await pre.children.find(n => n.tagName === 'button').listeners.click();
  assert.equal(copied, 'const n = 1;\nnext();');
  function all(n) { return [n, ...n.children.flatMap(all)]; }
  const links = all(body).filter(n => n.tagName === 'a');
  assert.equal(links.length, 1); assert.equal(links[0].href, 'https://example.com');
  const value = [{title: 'Reference', url: 'https://example.com/'}];
  context.value = value;
  vm.runInContext('renderMessageSources(sources, value)', context);
  const link = sources.children[0];
  vm.runInContext('renderMessageSources(sources, value)', context);
  assert.equal(sources.children[0], link, 'Unchanged source keeps keyboard focus');
  vm.runInContext('renderMessageContent(body, "```py\\nprint(1)\\n")', context);
  assert(body.textContent.includes('print(1)'));
  console.log('Safe rendering checks passed.');
})().catch(e => { console.error(e); process.exitCode = 1; });
