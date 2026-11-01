'use strict';
// Slash-command picker for the composer. The command list comes from the server
// (ciel.terminal.command_catalog), and matching mirrors ciel.terminal.command_suggestions,
// so the web app completes exactly like the terminal: arrows move, Tab completes, Enter
// completes a partial command or sends a complete one, and Escape closes the list.

/** Same rules as the terminal: a leading command, or the supported options after it. */
function commandSuggestions(draft, catalog) {
  if (!catalog || !String(draft).startsWith('/')) return [];
  const rows = catalog.commands.map(row => [row.name, row.description]);
  if (!draft.includes(' ')) {
    const typed = draft.toLowerCase();
    return rows.filter(([name]) => name.startsWith(typed));
  }
  const lower = draft.toLowerCase(), space = lower.indexOf(' ');
  const command = lower.slice(0, space), argument = lower.slice(space + 1);
  const known = rows.find(([name]) => name === command);
  const description = known ? known[1] : '';
  return (catalog.arguments[command] || [])
    .filter(value => value.startsWith(argument))
    .map(value => [command + ' ' + value, description]);
}

class CommandPicker {
  constructor(input, menu) {
    this.input = input; this.menu = menu;
    this.catalog = null; this.loading = false;
    this.query = null; this.index = 0; this.dismissed = null; this.open = false;
  }
  async load() {
    if (this.catalog || this.loading) return;
    this.loading = true;
    try {
      const {response, data} = await requestJSON('/api/commands');
      if (response.ok && Array.isArray(data.commands)) this.catalog = data;
    } catch { /* The composer still sends commands without the picker. */ }
    finally { this.loading = false; }
  }
  choices() {
    const draft = this.input.value;
    if (draft !== this.query) { this.query = draft; this.index = 0; this.dismissed = null; }
    const end = draft.length;
    const atEnd = this.input.selectionStart === undefined
      || (this.input.selectionStart === end && this.input.selectionEnd === end);
    if (!this.open || !atEnd || this.dismissed === draft) return [];
    return commandSuggestions(draft, this.catalog);
  }
  complete(value, {spaced}) {
    const add = spaced && this.catalog && this.catalog.spaced.includes(value) ? ' ' : '';
    this.input.value = value + add;
    if (typeof Event === 'function' && this.input.dispatchEvent) this.input.dispatchEvent(new Event('input'));
    this.render();
  }
  /** Handle picker keys; returns true when the key was used by the picker. */
  key(event) {
    const choices = this.choices();
    if (!choices.length || event.isComposing) return false;
    const selected = choices[this.index % choices.length][0];
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      this.index = (this.index + (event.key === 'ArrowDown' ? 1 : -1) + choices.length) % choices.length;
      this.render();
    } else if (event.key === 'Tab' && !event.shiftKey) {
      this.complete(selected, {spaced: true});
    } else if (event.key === 'Enter' && !event.shiftKey) {
      if (selected.toLowerCase() === this.input.value.toLowerCase()) return false;
      this.complete(selected, {spaced: false});
    } else if (event.key === 'Escape') {
      this.dismissed = this.input.value; this.render();
    } else return false;
    event.preventDefault();
    return true;
  }
  setOpen(open) { this.open = open; this.render(); }
  render() {
    const choices = this.choices(), menu = this.menu;
    if (choices.length) this.index %= choices.length;
    clearNode(menu);
    menu.hidden = !choices.length;
    this.input.setAttribute('aria-expanded', String(Boolean(choices.length)));
    if (!choices.length) { this.input.removeAttribute?.('aria-activedescendant'); return; }
    choices.forEach(([name, description], index) => {
      const option = document.createElement('div');
      option.className = 'command-option'; option.id = 'command-option-' + index;
      option.setAttribute('role', 'option');
      option.setAttribute('aria-selected', String(index === this.index));
      const label = document.createElement('span'); label.className = 'command-name'; label.textContent = name;
      const detail = document.createElement('span'); detail.className = 'command-description'; detail.textContent = description;
      option.append(label, detail);
      // Keep focus in the composer; a tap or click completes like Tab.
      option.addEventListener('mousedown', event => event.preventDefault());
      option.addEventListener('click', () => { this.index = index; this.complete(name, {spaced: true}); this.input.focus(); });
      menu.append(option);
    });
    if (choices.length > 6) {
      const count = document.createElement('div'); count.className = 'command-count';
      count.textContent = (this.index + 1) + ' / ' + choices.length + ' commands';
      menu.append(count);
    }
    this.input.setAttribute('aria-activedescendant', 'command-option-' + this.index);
    const active = menu.children[this.index];
    if (active && active.scrollIntoView) active.scrollIntoView({block: 'nearest'});
  }
}

/** The /help panel: the terminal's command groups as a polished, tappable reference. */
class HelpPanel {
  constructor(panel, picker) { this.panel = panel; this.picker = picker; this.opened = false; this.returnFocus = null; }
  /** Open for `/help` or `/help COMMAND`; returns false when the command list is unavailable. */
  open(argument = '') {
    const catalog = this.picker && this.picker.catalog;
    if (!catalog || !Array.isArray(catalog.groups)) return false;
    const body = element('help-body'); clearNode(body);
    const wanted = '/' + String(argument).trim().replace(/^\//, '').split(/\s+/)[0].toLowerCase();
    const detail = wanted.length > 1 && catalog.details ? catalog.details[wanted] : '';
    if (detail) {
      const card = document.createElement('div'); card.className = 'help-detail';
      const [usage, ...rest] = detail.split(' — ');
      const name = document.createElement('code'); name.textContent = usage;
      const text = document.createElement('p'); text.textContent = rest.join(' — ') || detail;
      card.append(name, text); body.append(card);
    } else if (wanted.length > 1) {
      const missing = document.createElement('p'); missing.className = 'help-missing';
      missing.textContent = 'No command named ' + wanted + '. Here is everything Ciel understands.';
      body.append(missing);
    }
    for (const group of catalog.groups) {
      const section = document.createElement('section'); section.className = 'help-group';
      const heading = document.createElement('h3'); heading.textContent = group.title;
      section.append(heading);
      for (const command of group.commands) {
        const row = document.createElement('button'); row.type = 'button'; row.className = 'help-command';
        if (command.name === wanted) row.classList.add('help-current');
        const label = document.createElement('code'); label.textContent = command.label;
        const description = document.createElement('span'); description.textContent = command.description;
        row.append(label, description);
        row.addEventListener('click', () => this.use(command));
        section.append(row);
      }
      body.append(section);
    }
    this.returnFocus = document.activeElement || null;
    this.opened = true; this.panel.hidden = false;
    element('help-close').focus();
    return true;
  }
  /** Put the chosen command in the composer, ready to send or complete. */
  use(command) {
    const spaced = this.picker.catalog.spaced.includes(command.name);
    const input = element('input');
    input.value = command.name + (spaced ? ' ' : '');
    this.close(); input.focus();
    if (typeof Event === 'function' && input.dispatchEvent) input.dispatchEvent(new Event('input'));
  }
  close() {
    if (!this.opened) return;
    this.opened = false; this.panel.hidden = true;
    if (this.returnFocus && this.returnFocus.focus) this.returnFocus.focus();
  }
}
