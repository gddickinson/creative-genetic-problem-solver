// Reaction-diffusion page: ReactionDiffusionSimulator from reaction-diffusion.py (rd_engine.js).
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { GrayScott, PATTERNS, INITS } from './rd_engine.js';
import { lab } from './lab.js';
import { fitCanvas, lineChart, fmt } from './chart.js';
import { ramp } from './heat.js';

const CLASSES = [[0.1, 'Uniform/Stable'], [0.2, 'Regular Spots/Stripes'], [0.3, 'Complex Patterns'], [Infinity, 'Chaotic/Turbulent']];

route('rd', () => {
  let g, frames, off;
  const L = lab({
    kicker: 'Gray-Scott model · Turing patterns · not a genetic algorithm', title: 'Reaction-Diffusion Simulator',
    lead: `Two chemicals U and V diffuse (V at half U's rate) and react: U + 2V → 3V, while U is fed in at rate F and V
      is removed at rate F + k. From a few seeded patches of V, spots, stripes or waves emerge. Pattern presets only
      change F and k. This demo is in the collection but uses no genetic algorithm.`,
    controls: [
      { id: 'pattern', label: 'Pattern type (sets F and k)', type: 'select', value: 'spots', options: Object.entries(PATTERNS).map(([k, [F, kk]]) => [k, `${k} (F ${F}, k ${kk})`]) },
      { id: 'init', label: 'Initial condition', type: 'select', value: 'random', options: INITS },
      { id: 'size', label: 'Grid size', min: 30, max: 160, step: 10, value: 100 },
      { id: 'frames', label: 'Frames (10 time steps each)', min: 20, max: 1000, step: 20, value: 200 },
    ],
    stages: [{ id: 'v', w: 600, h: 600, label: 'Concentration of V (bright = high), periodic boundaries' },
      { id: 'feat', w: 600, h: 180, label: 'Feature count: cells with V above mean + 1 standard deviation' }],
    chartLabel: 'Pattern complexity (mean gradient magnitude of V)', source: 'reaction-diffusion.py', maxSpeed: 10,
    onReset(p) {
      g = new GrayScott(p.size, p.size, new Rng(p.seed)); g.setParameters(p.pattern); g.initializePattern(p.init);
      frames = p.frames; off = document.createElement('canvas'); off.width = off.height = p.size;
    },
    step: () => g.frame(), done: () => g.metrics.length >= frames, render,
    notes: `<h2>The model as coded</h2>
      <p class="eq">dU = 0.16·∇²U − U·V² + F·(1 − U) &nbsp;&nbsp; dV = 0.08·∇²V + U·V² − (F + k)·V</p>
      <p>∇² is the five-point Laplacian with wrap-around edges; each time step adds dU and dV (dt = 1) and clips both to
      [0, 1]. The desktop app classifies the final pattern by its complexity: below 0.1 uniform/stable, below 0.2 regular
      spots/stripes, below 0.3 complex, otherwise chaotic.</p>`,
  });

  function render() {
    const cv = L.canvas('v'), ctx = fitCanvas(cv), n = g.W, octx = off.getContext('2d'), img = octx.createImageData(n, n);
    g.V.forEach((v, i) => img.data.set([...ramp(v), 255], i * 4));
    octx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, cv.width, cv.height);
    const m = g.metrics;
    L.chart([{ data: m.map(x => x.complexity), label: 'complexity' }], { yLabel: 'complexity', xLabel: 'Frame', x0: 1 });
    lineChart(L.canvas('feat'), [{ data: m.map(x => x.feature_count), label: 'features', color: '#ff4fd8' }], { yLabel: 'cells', xLabel: 'Frame', x0: 1 });
    const last = m.at(-1);
    L.setReadout([['Time step', g.steps], ['F / k', `${g.F} / ${g.k}`], ['Mean U / V', last ? `${fmt(last.avg_u)} / ${fmt(last.avg_v)}` : '-'],
      ['Complexity', last ? fmt(last.complexity) : '-'], ['Feature count', last ? last.feature_count : '-'],
      ['Classification', last ? CLASSES.find(([t]) => last.complexity < t)[1] : '-']]);
    L.setStatus(m.length >= frames ? 'Finished.' : m.length ? 'Simulating...' : 'Press Run or Step.');
  }
});
