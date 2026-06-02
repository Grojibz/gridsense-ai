"""Minimal, dependency-free web UI served at the app root.

A single self-contained HTML document (no external assets, no build step) with two tabs:
a DocRAG **chat** (calls ``POST /ask``) and a DegradeML **battery prediction** form (calls
``POST /predict``). It ships inside the package/image and works fully offline.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_CHAT_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>GridSense — DocRAG & Batterie</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --me:#2563eb; --bot:#334155;
          --accent:#38bdf8; --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:#e2e8f0; height:100vh; display:flex; flex-direction:column; }
  header { padding:12px 20px; background:linear-gradient(90deg,#0ea5e9,#6366f1);
           display:flex; align-items:center; gap:16px; }
  header .title { font-weight:600; font-size:18px; }
  .tabs { margin-left:auto; display:flex; gap:8px; }
  .tab { padding:8px 14px; border:none; border-radius:8px; cursor:pointer; font-weight:600;
         background:rgba(255,255,255,.15); color:#fff; }
  .tab.active { background:#fff; color:#1e293b; }
  #content { flex:1; min-height:0; display:flex; }
  .panel { flex:1; display:none; flex-direction:column; min-height:0; }
  .panel.active { display:flex; }
  /* --- chat --- */
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px;
         max-width:820px; width:100%; margin:0 auto; }
  .msg { padding:12px 14px; border-radius:14px; line-height:1.5; white-space:pre-wrap; max-width:85%; }
  .me { background:var(--me); align-self:flex-end; border-bottom-right-radius:4px; }
  .bot { background:var(--bot); align-self:flex-start; border-bottom-left-radius:4px; }
  .meta { font-size:12px; color:var(--muted); margin-top:8px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
  details { margin-top:8px; font-size:13px; }
  details summary { cursor:pointer; color:var(--accent); }
  .cite { margin:6px 0; padding:8px 10px; background:#0b1220; border-radius:8px;
          border-left:3px solid var(--accent); }
  .cite b { color:var(--accent); }
  .empty { color:var(--muted); align-self:center; margin-top:30vh; text-align:center; }
  form.chat { display:flex; gap:10px; padding:14px 20px; background:var(--panel);
              max-width:820px; width:100%; margin:0 auto; }
  form.chat input { flex:1; }
  input, button { padding:12px 14px; border-radius:10px; border:1px solid #334155;
                  background:#0b1220; color:#e2e8f0; font-size:15px; }
  button.send { border:none; background:var(--accent); color:#0b1220; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .dots span { animation: blink 1.2s infinite; } .dots span:nth-child(2){animation-delay:.2s;}
  .dots span:nth-child(3){animation-delay:.4s;}
  @keyframes blink { 0%,100%{opacity:.2;} 50%{opacity:1;} }
  /* --- predict --- */
  #panel-predict { overflow-y:auto; align-items:center; }
  .card { background:var(--panel); border-radius:14px; padding:22px; margin:24px;
          width:100%; max-width:520px; }
  .card h2 { margin:0 0 4px; font-size:18px; } .card p.hint { margin:0 0 16px; color:var(--muted); font-size:13px; }
  .field { margin-bottom:14px; } .field label { display:block; font-size:13px; margin-bottom:4px; color:var(--muted); }
  .field input { width:100%; }
  .card button.send { width:100%; }
  #result { margin-top:18px; display:none; }
  .gauge { height:26px; border-radius:6px; background:#0b1220; overflow:hidden; margin-top:8px; }
  .gauge > div { height:100%; transition:width .4s; }
  .soh { font-size:30px; font-weight:700; } .ver { color:var(--muted); font-size:12px; margin-top:6px; }
</style>
</head>
<body>
  <header>
    <span class="title">🔋 GridSense</span>
    <div class="tabs">
      <button class="tab active" data-tab="chat">💬 Chat (DocRAG)</button>
      <button class="tab" data-tab="predict">📈 Prédiction batterie</button>
    </div>
  </header>

  <div id="content">
    <!-- CHAT -->
    <div class="panel active" id="panel-chat">
      <div id="log"><div class="empty">Pose une question sur les documents techniques.<br>
        Ex. « Qu'est-ce que le thermal runaway ? »</div></div>
      <form class="chat" id="f">
        <input id="q" autocomplete="off" placeholder="Pose ta question…" autofocus />
        <button class="send" id="send" type="submit">Envoyer</button>
      </form>
    </div>

    <!-- PREDICT -->
    <div class="panel" id="panel-predict">
      <div class="card">
        <h2>Prédiction de l'état de santé (SOH)</h2>
        <p class="hint">Conditions d'usage de la cellule → SOH estimé (%).</p>
        <form id="pf">
          <div class="field"><label>Nombre de cycles</label>
            <input id="cycle_count" type="number" value="1500" min="0" step="50" /></div>
          <div class="field"><label>Température moyenne (°C)</label>
            <input id="avg_temperature_c" type="number" value="33" step="1" /></div>
          <div class="field"><label>Profondeur de décharge moyenne (0–1)</label>
            <input id="avg_dod" type="number" value="0.7" min="0" max="1" step="0.05" /></div>
          <div class="field"><label>C-rate moyen</label>
            <input id="avg_c_rate" type="number" value="1.0" min="0" step="0.1" /></div>
          <div class="field"><label>Âge calendaire (jours)</label>
            <input id="calendar_age_days" type="number" value="600" min="0" step="30" /></div>
          <button class="send" id="psend" type="submit">Prédire</button>
        </form>
        <div id="result">
          <div class="soh" id="sohval">—</div>
          <div class="gauge"><div id="sohbar"></div></div>
          <div class="ver" id="sohver"></div>
        </div>
      </div>
    </div>
  </div>

<script>
  const esc = s => { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; };

  // --- tabs ---
  document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
  }));

  // --- chat (DocRAG /ask) ---
  const log = document.getElementById('log');
  const form = document.getElementById('f');
  const input = document.getElementById('q');
  const send = document.getElementById('send');

  function clearEmpty(){ const e = log.querySelector('.empty'); if (e) e.remove(); }
  function addUser(t){ clearEmpty(); const d=document.createElement('div'); d.className='msg me'; d.textContent=t; log.appendChild(d); log.scrollTop=log.scrollHeight; }
  function addTyping(){ const d=document.createElement('div'); d.className='msg bot';
    d.innerHTML='<span class="dots"><span>●</span><span>●</span><span>●</span></span>';
    log.appendChild(d); log.scrollTop=log.scrollHeight; return d; }
  function renderAnswer(node, data){
    const pct = Math.round((data.confidence ?? 0) * 100);
    const color = pct >= 50 ? 'var(--ok)' : pct > 0 ? 'var(--warn)' : 'var(--muted)';
    let html = esc(data.answer);
    html += `<div class="meta"><span class="badge" style="background:${color};color:#0b1220">confiance ${pct}%</span></div>`;
    if (data.citations && data.citations.length){
      html += `<details open><summary>${data.citations.length} source(s)</summary>`;
      for (const c of data.citations) html += `<div class="cite"><b>${esc(c.source)}</b><br>${esc(c.snippet)}</div>`;
      html += `</details>`;
    }
    node.innerHTML = html; log.scrollTop = log.scrollHeight;
  }
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const question = input.value.trim(); if (!question) return;
    addUser(question); input.value=''; input.disabled=true; send.disabled=true;
    const typing = addTyping();
    try {
      const res = await fetch('/ask', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({ question }) });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      renderAnswer(typing, await res.json());
    } catch (err) { typing.textContent = '⚠️ Erreur : ' + err.message; }
    finally { input.disabled=false; send.disabled=false; input.focus(); }
  });

  // --- predict (DegradeML /predict) ---
  const pf = document.getElementById('pf');
  const psend = document.getElementById('psend');
  const result = document.getElementById('result');
  const fields = ['cycle_count','avg_temperature_c','avg_dod','avg_c_rate','calendar_age_days'];

  pf.addEventListener('submit', async e => {
    e.preventDefault();
    const body = {}; for (const f of fields) body[f] = parseFloat(document.getElementById(f).value);
    psend.disabled = true; psend.textContent = 'Calcul…';
    try {
      const res = await fetch('/predict', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const soh = data.predicted_soh;
      const color = soh >= 90 ? 'var(--ok)' : soh >= 80 ? 'var(--warn)' : 'var(--bad)';
      document.getElementById('sohval').textContent = 'SOH ≈ ' + soh.toFixed(1) + ' %';
      document.getElementById('sohval').style.color = color;
      const bar = document.getElementById('sohbar');
      bar.style.width = Math.max(0, Math.min(100, soh)) + '%'; bar.style.background = color;
      document.getElementById('sohver').textContent = 'modèle v' + data.model_version
        + (soh < 80 ? ' — proche de la fin de vie (≤80 %)' : '');
      result.style.display = 'block';
    } catch (err) {
      document.getElementById('sohval').textContent = '⚠️ ' + err.message;
      document.getElementById('sohval').style.color = 'var(--bad)';
      document.getElementById('sohbar').style.width = '0';
      document.getElementById('sohver').textContent = '';
      result.style.display = 'block';
    } finally { psend.disabled = false; psend.textContent = 'Prédire'; }
  });
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat() -> HTMLResponse:
    """Serve the GridSense web UI (DocRAG chat + battery prediction)."""
    return HTMLResponse(_CHAT_HTML)
