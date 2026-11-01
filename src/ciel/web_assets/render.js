'use strict';
// A deliberately small Markdown subset. Content never enters an HTML parser.
function safeSources(sources) {
  if (!Array.isArray(sources)) return [];
  const result = [], seen = new Set();
  for (const source of sources) {
    if (!source || typeof source.url !== 'string' || typeof source.title !== 'string') continue;
    if (!/^https?:\/\//i.test(source.url) || source.url.length > 2048
        || /^https?:\/\/[^/?#]*@/i.test(source.url)
        || /[\s\u0000-\u001f\u007f-\u009f\\]/.test(source.url)
        || /[\u0000-\u001f\u007f-\u009f]/.test(source.title)) continue;
    let url;
    try { url = new URL(source.url); } catch { continue; }
    if (!['http:', 'https:'].includes(url.protocol) || !url.hostname || url.username || url.password) continue;
    if (seen.has(source.url)) continue;
    seen.add(source.url);
    result.push({title: source.title.trim().slice(0, 300) || url.hostname, url: source.url});
    if (result.length === 20) break;
  }
  return result;
}
function clearNode(node) { while (node.firstElementChild) node.firstElementChild.remove(); node.textContent = ''; }
function renderMessageSources(node, sources) {
  const safe = safeSources(sources), key = JSON.stringify(safe);
  if (node.dataset.sources !== key) {
    clearNode(node);
    safe.forEach((source, index) => {
      const link = document.createElement('a');
      link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      link.textContent = '[' + (index + 1) + '] ' + source.title;
      node.append(link);
    });
    node.dataset.sources = key;
  }
  node.hidden = !safe.length;
}
function appendInline(node, value) {
  const tokens = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^\s)]+\))/g;
  let cursor = 0;
  function plain(s) { const span = document.createElement('span'); span.textContent = s; node.append(span); }
  for (const match of value.matchAll(tokens)) {
    if (match.index > cursor) plain(value.slice(cursor, match.index));
    const token = match[0];
    let child;
    if (token[0] === '`') { child = document.createElement('code'); child.textContent = token.slice(1, -1); }
    else if (token.startsWith('**')) { child = document.createElement('strong'); child.textContent = token.slice(2, -2); }
    else {
      const split = token.indexOf(']('), title = token.slice(1, split), url = token.slice(split + 2, -1);
      const safe = safeSources([{title, url}]);
      child = document.createElement(safe.length ? 'a' : 'span');
      child.textContent = safe.length ? title : token;
      if (safe.length) { child.href = url; child.target = '_blank'; child.rel = 'noopener noreferrer'; }
    }
    node.append(child); cursor = match.index + token.length;
  }
  if (cursor < value.length) plain(value.slice(cursor));
}
function renderMessageContent(node, value) {
  if (node.dataset.content === value) return;
  clearNode(node); node.dataset.content = value;
  const lines = String(value).split('\n');
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].trim()) { i++; continue; }
    if (/^\s*```/.test(lines[i])) {
      const language = lines[i++].replace(/^\s*```/, '').trim().slice(0, 30);
      const codeLines = [];
      while (i < lines.length && !/^\s*```/.test(lines[i])) codeLines.push(lines[i++]);
      if (i < lines.length) i++;
      const source = codeLines.join('\n'), wrap = document.createElement('div'); wrap.className = 'code-block';
      const button = document.createElement('button'); button.type = 'button'; button.textContent = 'Copy code';
      button.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(source); button.textContent = 'Copied'; }
        catch { button.textContent = 'Select code to copy'; }
      });
      const pre = document.createElement('pre'), code = document.createElement('code');
      pre.tabIndex = 0; code.textContent = source; code.setAttribute('aria-label', language || 'Code');
      pre.append(code); wrap.append(button, pre); node.append(wrap); continue;
    }
    const listMatch = lines[i].match(/^\s*(?:([-*])|\d+\.)\s+(.*)/);
    if (listMatch) {
      const ordered = !listMatch[1], list = document.createElement(ordered ? 'ol' : 'ul');
      while (i < lines.length) {
        const item = lines[i].match(ordered ? /^\s*\d+\.\s+(.*)/ : /^\s*[-*]\s+(.*)/);
        if (!item) break;
        const li = document.createElement('li'); appendInline(li, item[1]); list.append(li); i++;
      }
      node.append(list); continue;
    }
    const block = [];
    do { block.push(lines[i++]); }
    while (i < lines.length && lines[i].trim() && !/^\s*(```|[-*]\s|\d+\.\s)/.test(lines[i]));
    const paragraph = document.createElement('p'); appendInline(paragraph, block.join('\n')); node.append(paragraph);
  }
}
