// Shared layout for the problem pages: parameter controls built from a spec list, a run/pause/step/reset loop driven
// by requestAnimationFrame, the fitness chart and a read-out table.
import { app, esc, list, animate, onLeave } from './core.js';
import { lineChart } from './chart.js';

// spec: { id, label, type: 'range'|'select'|'check', min, max, step, value, options: [[value, text]], unit }
function controlHtml(c) {
  if (c.type === 'select') {
    return `<label for="p-${c.id}">${esc(c.label)}</label><select id="p-${c.id}">${list(c.options, ([v, t]) =>
      `<option value="${esc(v)}"${String(v) === String(c.value) ? ' selected' : ''}>${esc(t)}</option>`)}</select>`;
  }
  if (c.type === 'check') return `<label><input type="checkbox" id="p-${c.id}"${c.value ? ' checked' : ''}> ${esc(c.label)}</label>`;
  return `<label for="p-${c.id}">${esc(c.label)} <b id="v-${c.id}">${c.value}${c.unit || ''}</b></label>
    <input type="range" id="p-${c.id}" min="${c.min}" max="${c.max}" step="${c.step || 1}" value="${c.value}">`;
}

// Renders the page and returns { params(), stage, chartCanvas, setReadout(rows), setStatus(text) }.
// opts: { kicker, title, lead, controls, stages: [{id, w, h, label}], chartLabel, onReset(params), step(),
//         render(), done(), notes (HTML), source: 'file.py', interval: ms between ticks, maxSpeed, seed }
export function lab(opts) {
  const controls = [...opts.controls,
    { id: 'seed', label: 'Random seed', min: 1, max: 999, value: opts.seed ?? 1 },
    { id: 'speed', label: 'Steps per frame', min: 1, max: opts.maxSpeed || 20, value: opts.speed ?? 1 }];
  app().innerHTML = `
  <section class="hero"><p class="kicker">${esc(opts.kicker)}</p><h1 class="glow">${esc(opts.title)}</h1>
    <p class="lead">${opts.lead}</p></section>
  <div class="lab">
    <aside class="controls panel" aria-label="Parameters">
      ${list(controls, c => `<div class="ctl">${controlHtml(c)}</div>`)}
      <div class="btns"><button id="run">Run</button><button class="alt" id="stepb">Step</button><button class="alt" id="reset">Reset</button></div>
      <p class="small mute" id="status" aria-live="polite"></p>
    </aside>
    <div class="view">
      ${list(opts.stages, s => `<figure class="fig"><canvas id="${s.id}" width="${s.w}" height="${s.h}" role="img" aria-label="${esc(s.label)}"></canvas>
        <figcaption class="small mute">${esc(s.label)}</figcaption></figure>`)}
      <figure class="fig"><canvas id="chart" width="600" height="240" role="img" aria-label="${esc(opts.chartLabel)} over generations"></canvas>
        <figcaption class="small mute">${esc(opts.chartLabel)}</figcaption></figure>
      <div class="readout" id="readout"></div>
    </div>
  </div>
  <div class="notes">${opts.notes || ''}
  <p class="small mute">Ported from <code>${esc(opts.source)}</code> in the repository; every random number comes from a seeded generator, so a seed always replays the same run.</p></div>`;

  const val = c => { const el = document.getElementById('p-' + c.id);
    return c.type === 'select' ? el.value : c.type === 'check' ? el.checked : +el.value; };
  const params = () => Object.fromEntries(controls.map(c => [c.id, val(c)]));
  let running = false, stop = null, ready = false;
  const runBtn = document.getElementById('run'), status = document.getElementById('status');
  const draw = () => { opts.render(); };
  const reset = () => { pause(); opts.onReset(params()); ready = true; runBtn.textContent = 'Run'; draw(); };
  let last = 0;
  const tick = t => {
    if (opts.interval && t - last < opts.interval) return;      // slow pages (e.g. one augmenting path per tick)
    last = t;
    const n = val(controls[controls.length - 1]);
    for (let i = 0; i < n && !opts.done(); i++) opts.step();
    draw();
    if (opts.done()) pause();
  };
  function pause() { running = false; if (stop) stop(); stop = null; runBtn.textContent = ready && opts.done() ? 'Run again' : 'Run'; }
  runBtn.onclick = () => {
    if (running) return pause();
    if (opts.done()) reset();
    running = true; runBtn.textContent = 'Pause'; stop = animate(tick);
  };
  document.getElementById('stepb').onclick = () => { pause(); if (!opts.done()) { opts.step(); draw(); } };
  document.getElementById('reset').onclick = reset;
  for (const c of controls) {
    const el = document.getElementById('p-' + c.id);
    el.addEventListener('input', () => {
      const b = document.getElementById('v-' + c.id); if (b) b.textContent = el.value + (c.unit || '');
      if (c.id !== 'speed' && !c.live) reset(); else if (c.live && opts.onLive) { opts.onLive(params()); draw(); }
    });
  }
  onLeave(() => { running = false; if (stop) stop(); });
  const api = {
    params,
    setStatus: t => { status.textContent = t; },
    setReadout: rows => { document.getElementById('readout').innerHTML = list(rows, ([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`); },
    chart: (series, o) => lineChart(document.getElementById('chart'), series, o),
    canvas: id => document.getElementById(id),
  };
  queueMicrotask(reset);
  return api;
}
