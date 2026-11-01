'use strict';
class CielSettings {
  constructor() { this.baseline=null; this.opened=false; this.busy=false; }
  async open(section='') {
    this.opened=true; element('settings-panel').hidden=false;
    document.querySelector('.app').inert=true;
    element('settings-close').focus();
    try {
      const {response,data}=await requestJSON('/api/settings');
      if(!response.ok || !data.values) throw Error('Settings could not be loaded. Try again.');
      this.render(data);
      await this.network();
      if(section==='connection') element('connection-settings').scrollIntoView({block:'start'});
    } catch(error) { text(element('settings-notice'),error.message); }
  }
  close() {
    this.opened=false; element('settings-panel').hidden=true;
    document.querySelector('.app').inert=false; element('settings-open').focus();
  }
  render(data) {
    this.baseline=data;
    for(const field of document.querySelectorAll('.setting-field')) {
      const key=field.dataset.setting;
      if(field.type==='password') field.value='';
      else if(field.type==='checkbox') field.checked=Boolean(data.values[key]);
      else field.value=String(data.values[key] ?? '');
    }
    for(const key of ['nim_api_key','groq_api_key','openrouter_api_key']) text(element('key-state-'+key),data.configured[key]?'Saved':'Not configured');
    for(const box of document.querySelectorAll('[data-remove-key]')) box.checked=false;
    element('setting-persona').value=data.persona || '';
    text(element('settings-notice'),data.needs_restart?'Saved. Restart the server to apply voice or image changes.':'');
  }
  async save() {
    if(this.busy || !this.baseline) return;
    const changes={};
    for(const field of document.querySelectorAll('.setting-field')) {
      const key=field.dataset.setting;
      const value=field.type==='checkbox'?field.checked:field.type==='number'?Number(field.value):field.value.trim();
      if(field.type==='password') { if(value) changes[key]=value; }
      else if(value!==this.baseline.values[key]) changes[key]=value;
    }
    for(const box of document.querySelectorAll('[data-remove-key]')) if(box.checked) changes[box.dataset.removeKey]=null;
    if(element('setting-persona').value!==(this.baseline.persona || '')) changes.persona=element('setting-persona').value;
    if(!Object.keys(changes).length) { text(element('settings-notice'),'No changes to save.'); return; }
    this.busy=true;element('settings-save').disabled=true;
    try {
      const {response,data}=await requestJSON('/api/settings',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:JSON.stringify(changes)});
      if(!response.ok) throw Error(data.error || 'Changes could not be saved.');
      this.render(data);
      if(!data.needs_restart) text(element('settings-notice'),'Saved. New replies use these settings.');
    } catch(error) { text(element('settings-notice'),error.message); }
    finally { this.busy=false;element('settings-save').disabled=false; }
  }
  async network() {
    const {response,data}=await requestJSON('/api/network');
    if(!response.ok) return;
    element('network-host').value=data.host || '';
    const local=browserAudio.device==='pc';
    element('network-enable').hidden=!local || Boolean(data.insecure_lan);
    element('network-code').hidden=!local || Boolean(data.insecure_lan);
    element('network-trust-instructions').hidden=Boolean(data.insecure_lan);
    element('network-setup').hidden=!data.setup_url;
    if(data.setup_url) {element('network-setup').href=data.setup_url;element('network-setup').textContent='On your phone, open '+data.setup_url;}
    element('network-certificate').hidden=!data.certificate_available;
    element('network-open').hidden=!data.enabled;
    if(data.enabled) { element('network-open').href=data.url; element('network-open').textContent=data.url; }
    text(element('network-result'),data.insecure_lan?'Open directly on your home network. No certificate or pairing code. Phone microphone recording needs HTTPS.':data.enabled?'Phone address: '+data.url:'Enable private Wi-Fi access from this PC.');
    text(element('network-fingerprint'),data.fingerprint?'Certificate SHA-256: '+data.fingerprint:'');
  }
  async enableNetwork() {
    element('network-enable').disabled=true;
    text(element('network-result'),'Preparing your local connection…');
    try {
      const {response,data}=await requestJSON('/api/network/setup',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:JSON.stringify({host:element('network-host').value.trim()})});
      if(!response.ok) throw Error(data.error || 'Phone access could not be enabled.');
      await this.network();
    } catch(error) { text(element('network-result'),error.message); }
    finally { element('network-enable').disabled=false; }
  }
  async pairCode() {
    const {response,data}=await requestJSON('/api/pair-code',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:'{}'});
    text(element('network-pair-code'),response.ok?'Pairing code: '+data.code+' · valid for 5 minutes':data.error || 'Open this section on your PC to make a pairing code.');
  }
  async memories() {
    const {response,data}=await requestJSON('/api/memory');
    const list=element('saved-memory');clearNode(list);
    if(!response.ok) { list.textContent='Saved memory could not be loaded.';return; }
    for(const item of data.memories || []) {
      const row=document.createElement('div');row.className='memory-row';
      const content=document.createElement('p');content.textContent=item.text;
      const forget=document.createElement('button');forget.type='button';forget.textContent='Forget';
      forget.addEventListener('click',async()=>{
        if(!window.confirm('Forget this saved fact?')) return;
        const {response}=await requestJSON('/api/memory/forget',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:JSON.stringify({id:item.id,confirmed:true})});
        if(response.ok) await this.memories();
      });
      row.append(content,forget);list.append(row);
    }
    if(!list.children.length) list.textContent='No saved facts yet.';
  }
  async restart() {
    if(!window.confirm('Restart Ciel? Phone devices will need a new pairing code.')) return;
    const {response}=await requestJSON('/api/server/restart',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:JSON.stringify({confirmed:true})});
    if(response.ok) text(element('settings-notice'),'Restarting Ciel. Keep this page open.');
  }
}
