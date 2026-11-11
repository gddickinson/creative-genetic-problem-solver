// Neural network page: weights evolved by NeuralNetVisualizer.genetic_train from neural-vis.py (neural_engine.js).
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { PATTERNS, randomNetwork, generateData, forward, NeuroGA } from './neural_engine.js';
import { lab } from './lab.js';
import { fitCanvas } from './chart.js';
import { grid, paint } from './heat.js';

const ARCHS = ['2 5 5 1', '2 3 1', '2 8 1', '2 8 8 1', '2 10 10 10 1'];

route('neural', () => {
  let ga, bg = null, bgFor = null;
  const L = lab({
    kicker: 'Neuroevolution · no gradients', title: 'Neural Network Visualizer',
    lead: `A small feed-forward network (ReLU hidden layers, sigmoid output) learns to separate two classes of 2-D points,
      but its weights are found by a genetic algorithm instead of backpropagation. Fitness is the share of points
      classified correctly (output above 0.5 means class 1).`,
    controls: [
      { id: 'pattern', label: 'Pattern', type: 'select', value: 'spiral', options: PATTERNS },
      { id: 'arch', label: 'Layer sizes (input … output)', type: 'select', value: ARCHS[0], options: ARCHS.map(a => [a, `[${a.split(' ').join(', ')}]`]) },
      { id: 'pop', label: 'Population size', min: 10, max: 150, step: 10, value: 50 },
      { id: 'n', label: 'Training points', min: 40, max: 400, step: 20, value: 200 },
      { id: 'gens', label: 'Generations', min: 10, max: 400, step: 10, value: 100 },
    ],
    stages: [{ id: 'dec', w: 600, h: 440, label: 'Decision surface of the best network (lime = class 1, indigo = class 0) over the training points' },
      { id: 'net', w: 600, h: 230, label: 'Best network: green weights positive, red negative, brightness = magnitude' }],
    chartLabel: 'Best fitness (training accuracy)', source: 'neural-vis.py', maxSpeed: 10,
    onReset(p) {
      const R = new Rng(p.seed), sizes = p.arch.split(' ').map(Number);
      randomNetwork(sizes, R);                       // __init__ builds an initial network before the data
      const { X, y } = generateData(p.n, p.pattern, R);
      ga = new NeuroGA(sizes, X, y, { populationSize: p.pop }, R); ga.maxGen = p.gens; bg = null;
    },
    step: () => ga.step(), done: () => ga.generation >= ga.maxGen, render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Genome</b><span>Every weight and bias of the network (initialised as N(0, 1) × 0.1).</span></div>
      <div class="card"><b>Selection</b><span>Two parents drawn at random from the 10 most accurate networks.</span></div>
      <div class="card"><b>Crossover</b><span>Uniform: each weight and bias comes from either parent with probability ½.</span></div>
      <div class="card"><b>Mutation</b><span>Each weight and bias, with probability 0.1, gets N(0, 1) × 0.1 added.</span></div>
      <div class="card"><b>Elitism</b><span>The two most accurate networks are kept.</span></div></div>
      <p class="small mute">The points are normalised to zero mean and unit variance per axis; the surface is drawn over
      −3…3 like the desktop app. Accuracy changes in steps of one point, so the curve is a staircase.</p>`,
  });

  function render() {
    const cv = L.canvas('dec'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
    const side = Math.min(W, H) - 16 * s, ox = (W - side) / 2, oy = 8 * s;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    if (ga.best && bgFor !== ga.best) { bg = paint(grid(x => forward(x, ga.best), [-3, 3], [-3, 3], 70), v => v); bgFor = ga.best; }
    if (bg) { ctx.imageSmoothingEnabled = true; ctx.drawImage(bg, ox, oy, side, side); }
    else { ctx.strokeStyle = '#212c4c'; ctx.strokeRect(ox, oy, side, side); }
    const X = v => ox + side * (v + 3) / 6, Y = v => oy + side * (3 - v) / 6;
    ga.X.forEach((p, i) => { ctx.fillStyle = ga.y[i] ? '#3dd6ff' : '#ff4fd8'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.2 * s, 0, 7); ctx.fill(); ctx.stroke(); });
    drawNet(L.canvas('net'), ga.sizes, ga.best);
    L.chart([{ data: ga.history, label: 'accuracy' }], { yLabel: 'accuracy', x0: 1 });
    L.setReadout([['Generation', `${ga.generation} / ${ga.maxGen}`], ['Best accuracy', ga.best ? (ga.bestFitness * 100).toFixed(1) + '%' : '-'],
      ['Weights + biases', ga.sizes.slice(1).reduce((a, n, i) => a + n * (ga.sizes[i] + 1), 0)], ['Points', `${ga.X.length} (cyan class 1, pink class 0)`]]);
    L.setStatus(ga.generation >= ga.maxGen ? 'Finished.' : ga.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});

function drawNet(cv, sizes, net) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
  ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
  const xs = sizes.map((_, i) => 40 * s + (W - 80 * s) * i / (sizes.length - 1));
  const ys = sizes.map(n => Array.from({ length: n }, (_, k) => H / 2 + (k - (n - 1) / 2) * Math.min(22 * s, (H - 30 * s) / n)));
  for (let i = 0; i < sizes.length - 1; i++) for (let a = 0; a < sizes[i]; a++) for (let b = 0; b < sizes[i + 1]; b++) {
    const w = net ? net[i].W[a][b] : 0;
    ctx.strokeStyle = w < 0 ? `rgba(255,92,122,${Math.min(1, Math.abs(w))})` : `rgba(61,255,143,${Math.min(1, Math.abs(w))})`;
    ctx.lineWidth = 1.4 * s; ctx.beginPath(); ctx.moveTo(xs[i], ys[i][a]); ctx.lineTo(xs[i + 1], ys[i + 1][b]); ctx.stroke();
  }
  sizes.forEach((n, i) => ys[i].forEach(y => { ctx.fillStyle = '#3dd6ff'; ctx.beginPath(); ctx.arc(xs[i], y, 6 * s, 0, 7); ctx.fill(); }));
  ctx.fillStyle = '#94a3c6'; ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`; ctx.textAlign = 'center';
  ctx.fillText('inputs (x, y)', xs[0], H - 6 * s); ctx.fillText(`output ${net ? '' : '(press Run)'}`, xs.at(-1), H - 6 * s);
}
