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
