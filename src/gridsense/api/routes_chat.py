"""Minimal, dependency-free chat UI for DocRAG, served at the app root.

The page is a single self-contained HTML document (no external assets, no build step) so it
works fully offline and ships inside the package/image. It calls ``POST /ask`` and renders the
answer with its citations and confidence.
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
<title>GridSense — Assistant DocRAG</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --me:#2563eb; --bot:#334155;
          --accent:#38bdf8; --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:#e2e8f0; height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 20px; background:linear-gradient(90deg,#0ea5e9,#6366f1);
           font-weight:600; font-size:18px; display:flex; align-items:center; gap:10px; }
  header small { font-weight:400; opacity:.85; font-size:12px; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px;
         max-width:820px; width:100%; margin:0 auto; }
  .msg { padding:12px 14px; border-radius:14px; line-height:1.5; white-space:pre-wrap;
         max-width:85%; }
  .me { background:var(--me); align-self:flex-end; border-bottom-right-radius:4px; }
  .bot { background:var(--bot); align-self:flex-start; border-bottom-left-radius:4px; }
  .meta { font-size:12px; color:var(--muted); margin-top:8px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px;
           font-weight:600; }
  details { margin-top:8px; font-size:13px; }
  details summary { cursor:pointer; color:var(--accent); }
  .cite { margin:6px 0; padding:8px 10px; background:#0b1220; border-radius:8px;
          border-left:3px solid var(--accent); }
  .cite b { color:var(--accent); }
  .empty { color:var(--muted); align-self:center; margin-top:30vh; text-align:center; }
  form { display:flex; gap:10px; padding:14px 20px; background:var(--panel);
         max-width:820px; width:100%; margin:0 auto; }
  input { flex:1; padding:12px 14px; border-radius:10px; border:1px solid #334155;
          background:#0b1220; color:#e2e8f0; font-size:15px; }
  button { padding:12px 18px; border:none; border-radius:10px; background:var(--accent);
           color:#0b1220; font-weight:600; cursor:pointer; font-size:15px; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .dots span { animation: blink 1.2s infinite; } .dots span:nth-child(2){animation-delay:.2s;}
  .dots span:nth-child(3){animation-delay:.4s;}
  @keyframes blink { 0%,100%{opacity:.2;} 50%{opacity:1;} }
</style>
</head>
<body>
  <header>🔋 GridSense <small>— Assistant DocRAG (répond à partir des documents indexés)</small></header>
  <div id="log"><div class="empty">Pose une question sur les documents techniques.<br>
    Ex. « Qu'est-ce que le thermal runaway ? »</div></div>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="Pose ta question…" autofocus />
    <button id="send" type="submit">Envoyer</button>
  </form>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('f');
  const input = document.getElementById('q');
  const send = document.getElementById('send');

  function clearEmpty() { const e = log.querySelector('.empty'); if (e) e.remove(); }
  function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

  function addUser(text){
    clearEmpty();
    const d = document.createElement('div'); d.className='msg me'; d.textContent=text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
  }
  function addTyping(){
    const d = document.createElement('div'); d.className='msg bot';
    d.innerHTML = '<span class="dots"><span>●</span><span>●</span><span>●</span></span>';
    log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
  }
  function render(node, data){
    const pct = Math.round((data.confidence ?? 0) * 100);
    const color = pct >= 50 ? 'var(--ok)' : pct > 0 ? 'var(--warn)' : 'var(--muted)';
    let html = esc(data.answer);
    html += `<div class="meta"><span class="badge" style="background:${color};color:#0b1220">`
          + `confiance ${pct}%</span></div>`;
    if (data.citations && data.citations.length){
      html += `<details open><summary>${data.citations.length} source(s)</summary>`;
      for (const c of data.citations){
        html += `<div class="cite"><b>${esc(c.source)}</b><br>${esc(c.snippet)}</div>`;
      }
      html += `</details>`;
    }
    node.innerHTML = html; log.scrollTop = log.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    addUser(question);
    input.value = ''; input.disabled = true; send.disabled = true;
    const typing = addTyping();
    try {
      const res = await fetch('/ask', {
        method:'POST', headers:{'content-type':'application/json'},
        body: JSON.stringify({ question })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      render(typing, await res.json());
    } catch (err) {
      typing.textContent = '⚠️ Erreur : ' + err.message;
    } finally {
      input.disabled = false; send.disabled = false; input.focus();
    }
  });
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat() -> HTMLResponse:
    """Serve the DocRAG chat UI."""
    return HTMLResponse(_CHAT_HTML)
