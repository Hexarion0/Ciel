'use strict';
class AttachmentComposer {
  constructor() { this.items=[]; this.requestId=crypto.randomUUID().replaceAll('-',''); this.epoch=0; this.busy=false; }
  admittedIds() { this.checkExpiry(); return this.items.filter(item=>item.handle).map(item=>item.handle); }
  checkExpiry() {
    let changed=false;
    for(const item of this.items) if(item.handle && item.expiresAt <= Date.now()) { item.handle=null; item.expired=true; changed=true; }
    if(changed) this.render();
  }
  async add(files) {
    const images=Array.from(files);
    if (!images.length || !activeSelection) return;
    if (this.items.length+images.length>2) { feedback('Attach at most two images per message.',true); return; }
    if (images.some(file=>!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>10*1024*1024)) {
      feedback('Choose JPEG, PNG, or WebP images up to 10 MB each.',true); return;
    }
    const epoch=this.epoch, selection={...activeSelection};
    this.busy=true; element('attach').disabled=true;
    try {
      for(const file of images) {
        const item={file,url:URL.createObjectURL(file),handle:null,error:false}; this.items.push(item); this.render();
        try {
          const response=await fetch('/api/images',{method:'POST',headers:browserAudio.headers({
            'Content-Type':file.type,'X-Request-ID':this.requestId,'X-Revision':String(selection.revision),'X-Conversation':selection.handle
          }),body:file});
          const data=await response.json();
          if(epoch!==this.epoch || selection.revision!==activeSelection?.revision) { URL.revokeObjectURL(item.url); continue; }
          if(!response.ok) throw Error(data.error || 'Upload failed. Remove it and try again.');
          item.handle=data.handle; item.expiresAt=Date.now()+(data.expires_in || 900)*1000;
        } catch(error) { item.error=true; feedback(error.message,true); }
        this.render();
      }
    } finally { if(epoch===this.epoch){this.busy=false;element('attach').disabled=false;} }
  }
  remove(item) {
    this.items=this.items.filter(value=>value!==item); URL.revokeObjectURL(item.url);
    if(item.handle && activeSelection) requestJSON('/api/images/remove',{method:'POST',headers:browserAudio.headers({'Content-Type':'application/json'}),body:JSON.stringify({
      request_id:this.requestId,revision:activeSelection.revision,conversation_handle:activeSelection.handle,attachment_ids:[item.handle]
    })}).catch(()=>{});
    this.render();
  }
  clear() {
    this.epoch++; this.busy=false;
    for(const item of this.items) URL.revokeObjectURL(item.url);
    this.items=[]; this.requestId=crypto.randomUUID().replaceAll('-','');
    element('attach').disabled=false; this.render();
  }
  render() {
    const list=element('attachments'); clearNode(list);
    for(const item of this.items) {
      const row=document.createElement('div'); row.className='attachment';
      const image=document.createElement('img'); image.src=item.url; image.alt='Image attached to your message';
      const label=document.createElement('span'); label.textContent=item.expired?'Image expired. Attach it again.':item.error?'Upload failed':item.handle?'Image ready':'Uploading…';
      const remove=document.createElement('button'); remove.type='button'; remove.textContent='×'; remove.setAttribute('aria-label','Remove image');
      remove.addEventListener('click',()=>this.remove(item)); row.append(image,label,remove); list.append(row);
    }
    list.hidden=this.items.length===0;
  }
}
