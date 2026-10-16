"""Run frontend interaction regressions without introducing browser dependencies."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["web_ui.cjs", "orb_animation.cjs"])
def test_web_frontend_regressions(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the optional frontend interaction checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(root / "tests/frontend" / script)], cwd=root,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reply_sources_render_as_safe_links_and_preserve_keyboard_focus():
    """Run citation rendering against a DOM stub that forbids HTML injection."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the optional frontend interaction checks")
    root = Path(__file__).resolve().parents[1]
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const html = fs.readFileSync('src/ciel/web_assets/index.html', 'utf8');
assert(html.includes('aria-label="Sources for Ciel\'s reply"'));
class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.textContent = '';
    this.classes = new Set();
    this.classList = {
      toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name)
    };
  }
  get firstElementChild() { return this.children[0]; }
  set innerHTML(_value) { throw Error('Source labels must use textContent'); }
  append(child) { child.parent = this; this.children.push(child); }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
}
const sources = new Element(), slot = new Element();
const context = {
  URL,
  element: id => id === 'speech-sources' ? sources : slot,
  document: {createElement: tag => { assert.equal(tag, 'a'); return new Element(); }}
};
vm.createContext(context);
const source = html.slice(
  html.indexOf('function safeSources('), html.indexOf('function showSpeech(')
);
vm.runInContext(source, context);
function show(value) { vm.runInContext('showSources(' + JSON.stringify(value) + ')', context); }
const valid = {title: '<img src=x onerror=alert(1)>', url: 'https://example.com/page'};
show([
  {title: 'Unsafe', url: 'javascript:alert(1)'},
  {title: 'Credentials', url: 'https://user:password@example.com/'},
  {title: 'Empty credentials', url: 'https://@example.com/'},
  {title: 'Controls', url: 'https://example.com/\npage'},
  {title: 'Backslash', url: 'https://example.com\\page'},
  {title: 'Bad port', url: 'https://example.com:99999/'},
  {title: 'Bad label\n', url: 'https://example.com/other'}, valid
]);
assert.equal(sources.children.length, 1);
const link = sources.children[0];
assert.equal(link.textContent, '[1] ' + valid.title);
assert.equal(link.href, valid.url);
assert.equal(link.target, '_blank');
assert.equal(link.rel, 'noopener noreferrer');
assert(!sources.hidden && slot.classes.has('has-sources'));
show([valid]);
assert.equal(sources.children[0], link, 'Polling must preserve the focused source link');
show(Array.from({length: 30}, (_, index) => ({
  title: String(index), url: 'https://example.com/' + index
})));
assert.equal(sources.children.length, 20);
show([]);
assert.equal(sources.children.length, 0);
assert(sources.hidden && !slot.classes.has('has-sources'));
"""
    result = subprocess.run(
        [node, "-"], cwd=root, input=script,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
