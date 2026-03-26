async function fetchNames() {
  const res = await fetch('/api/names');
  const names = await res.json();
  return names;
}

function mk(text, tag='div', cls=''){
  const el = document.createElement(tag);
  if(cls) el.className = cls;
  el.textContent = text;
  return el;
}

function imageElem(path, alt=''){
  const img = document.createElement('img');
  // If path is a remote URL, use it directly so the browser can fetch it.
  if(typeof path === 'string' && (path.startsWith('http://') || path.startsWith('https://'))){
    img.src = path;
  } else {
    img.src = `/image?path=${encodeURIComponent(path)}`;
  }
  img.alt = alt;
  img.loading = 'lazy';
  return img;
}

async function searchAndRender(name){
  const res = await fetch(`/api/entity?name=${encodeURIComponent(name)}`);
  if(!res.ok){
    const err = await res.json().catch(()=>({error:'not found'}));
    document.getElementById('results').innerHTML = `<p class="muted">${err.error || 'Entity not found'}</p>`;
    return;
  }
  const payload = await res.json();
  renderEntity(payload);
}

function renderScore(label, val) {
  const row = mk('', 'div', 'score-row');
  row.appendChild(mk(label, 'span', 'score-lbl'));
  const disp = val !== undefined && val !== null ? (typeof val === 'number' ? val.toFixed(3) : val) : 'N/A';
  row.appendChild(mk(disp, 'span', 'score-val'));
  return row;
}

function renderEntity(payload){
  const main = document.getElementById('results');
  main.innerHTML = '';

  const h = document.createElement('h2');
  h.textContent = `${payload.name} — ${payload.category} / ${payload.subcategory}`;
  main.appendChild(h);

  const srcSection = document.createElement('div'); srcSection.className = 'source-section';
  
  // show source image
  const srcImgWrap = document.createElement('div'); srcImgWrap.className = 'source-image';
  const src = payload.data.source_entity;
  const image_url = src.image_url || (src.images && src.images[0] && src.images[0].local_path);
  if(image_url){
    srcImgWrap.appendChild(imageElem(image_url, payload.name));
  }
  srcSection.appendChild(srcImgWrap);

  const meta = document.createElement('div'); meta.className = 'source-meta';
  meta.appendChild(mk(`Source Region: ${payload.source_region} -> Target: ${payload.target_region}`, 'h4'));
  meta.appendChild(mk(src.description || '', 'p'));
  if(src.wikipedia_url){
    const a = document.createElement('a'); a.href = src.wikipedia_url; a.textContent = 'Wikipedia'; a.target='_blank';
    meta.appendChild(a);
  }
  srcSection.appendChild(meta);
  
  main.appendChild(srcSection);

  // alternatives list
  const list = document.createElement('div'); list.className = 'alt-list';
  const alts = payload.data.alternatives || [];
  
  const PROMPT_VARIANTS = ["baseline", "balanced_realism", "realism_focused", "structure_preserved"];
  
  for(let i=0;i<alts.length;i++){
    const a = alts[i];
    const card = document.createElement('div'); card.className='alt-card';

    // Header metadata
    const header = document.createElement('div'); header.className = 'alt-card-header';
    header.appendChild(mk(`${i+1}. ${a.target_item}`, 'h3'));
    header.appendChild(mk(`Axis: ${a.axis || ''}`, 'div', 'axis'));
    header.appendChild(mk(a.reason || '', 'p', 'reason'));
    if(a.scene_adjustments && a.scene_adjustments.length){
      const ul = document.createElement('ul');
      a.scene_adjustments.forEach(s=>ul.appendChild(mk(s,'li')));
      header.appendChild(ul);
    }
    card.appendChild(header);

    // Variants Grid
    const variantsGrid = document.createElement('div'); variantsGrid.className = 'variants-grid';
    
    // Ensure variants exist on this item
    const variants = a.variants || {};
    
    PROMPT_VARIANTS.forEach(vName => {
      const col = document.createElement('div'); col.className = 'variant-col';
      col.appendChild(mk(vName.replace('_', ' '), 'div', 'variant-header'));
      
      const vData = variants[vName];
      
      const imgWrap = document.createElement('div'); imgWrap.className = 'img-wrap';
      if(vData && vData.generated_image_path){
        imgWrap.appendChild(imageElem(vData.generated_image_path, a.target_item));
      } else {
        const ph = document.createElement('div'); ph.className = 'img-placeholder muted'; ph.textContent = 'Pending/Missing';
        imgWrap.appendChild(ph);
      }
      col.appendChild(imgWrap);
      
      if (vData && vData.eval_metrics) {
         const scoresDiv = document.createElement('div'); scoresDiv.className = 'scores';
         scoresDiv.appendChild(renderScore('ImageReward', vData.eval_metrics.image_reward));
         scoresDiv.appendChild(renderScore('MC-CLIP', vData.eval_metrics.mc_clip));
         scoresDiv.appendChild(renderScore('VLM Judge', vData.eval_metrics.vlm_judge));
         col.appendChild(scoresDiv);
      }
      
      variantsGrid.appendChild(col);
    });

    card.appendChild(variantsGrid);
    list.appendChild(card);
  }
  main.appendChild(list);
}

// wire search
window.addEventListener('DOMContentLoaded', async ()=>{
  const box = document.getElementById('search-box');
  const btn = document.getElementById('btn-search');
  const names = await fetchNames();

  const dl = document.createElement('datalist'); dl.id='names';
  names.forEach(n=>dl.appendChild((()=>{const o=document.createElement('option'); o.value=n; return o;})()));
  document.body.appendChild(dl);
  box.setAttribute('list','names');

  btn.onclick = ()=> searchAndRender(box.value.trim());
  box.onkeydown = (e)=>{ if(e.key==='Enter') searchAndRender(box.value.trim()); };
});
