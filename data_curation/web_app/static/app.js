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

function renderEntity(payload){
  const main = document.getElementById('results');
  main.innerHTML = '';

  const h = document.createElement('h2');
  h.textContent = `${payload.name} — ${payload.category} / ${payload.subcategory}`;
  main.appendChild(h);

  const src = payload.data.source_entity;
  const meta = document.createElement('div'); meta.className = 'source-meta';
  meta.appendChild(mk(src.description || ''));
  if(src.wikipedia_url){
    const a = document.createElement('a'); a.href = src.wikipedia_url; a.textContent = 'Wikipedia'; a.target='_blank';
    meta.appendChild(a);
  }
  main.appendChild(meta);

  // show source image
  const srcImgWrap = document.createElement('div'); srcImgWrap.className = 'source-image';
  const image_url = src.image_url || (src.images && src.images[0] && src.images[0].local_path);
  if(image_url){
    srcImgWrap.appendChild(imageElem(image_url, payload.name));
  }
  main.appendChild(srcImgWrap);

  // alternatives grid
  const grid = document.createElement('div'); grid.className = 'alt-grid';
  const alts = payload.data.alternatives || [];
  for(let i=0;i<alts.length;i++){
    const a = alts[i];
    const card = document.createElement('div'); card.className='alt-card';

    // Image first: create a fixed-size image wrapper so cards remain uniform
    const imgWrap = document.createElement('div'); imgWrap.className = 'img-wrap';
    if(a.generated_image_path){
      imgWrap.appendChild(imageElem(a.generated_image_path, a.target_item));
    } else {
      const ph = document.createElement('div'); ph.className = 'img-placeholder muted'; ph.textContent = 'No generated image';
      imgWrap.appendChild(ph);
    }
    card.appendChild(imgWrap);

    // Then metadata
    card.appendChild(mk(`${i+1}. ${a.target_item} (${a.target_item_local || ''})`, 'h3'));
    card.appendChild(mk(a.axis || '', 'h4', 'axis'));
    card.appendChild(mk(a.reason || '', 'p', 'reason'));
    if(a.scene_adjustments && a.scene_adjustments.length){
      const ul = document.createElement('ul');
      a.scene_adjustments.forEach(s=>ul.appendChild(mk(s,'li')));
      card.appendChild(ul);
    }
    grid.appendChild(card);
  }
  main.appendChild(grid);
}

// wire search
window.addEventListener('DOMContentLoaded', async ()=>{
  const box = document.getElementById('search-box');
  const btn = document.getElementById('btn-search');
  const names = await fetchNames();

  // simple autocomplete via datalist
  const dl = document.createElement('datalist'); dl.id='names';
  names.forEach(n=>dl.appendChild((()=>{const o=document.createElement('option'); o.value=n; return o;})()));
  document.body.appendChild(dl);
  box.setAttribute('list','names');

  btn.onclick = ()=> searchAndRender(box.value.trim());
  box.onkeydown = (e)=>{ if(e.key==='Enter') searchAndRender(box.value.trim()); };
});