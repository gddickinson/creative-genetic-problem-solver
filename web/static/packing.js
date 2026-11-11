// Packing page: PackingOptimizer from packing-optimizer.py (packing_engine.js), circles or rectangles in a box.
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { Packing, overlaps, footprint } from './packing_engine.js';
import { lab } from './lab.js';
import { fitCanvas, frame, fmt, COLORS } from './chart.js';

route('packing', () => {
  let p;
  const L = lab({
    kicker: 'Layout GA · uniform crossover · local-search mutation', title: 'Packing Optimizer',
    lead: `Place circles or rectangles inside a container without overlaps (a 0.2 safety margin is enforced), pulling
      them as close to the centre as possible. A layout's fitness is minus the summed distance of the items from the
      centre, or −∞ if anything overlaps or sticks out.`,
    controls: [
      { id: 'kind', label: 'Problem type', type: 'select', value: 'circles', options: [['circles', 'Circle packing'], ['rectangles', 'Rectangle packing']] },
      { id: 'n', label: 'Number of items', min: 2, max: 25, value: 10 },
      { id: 'w', label: 'Container width', min: 6, max: 20, value: 10 },
      { id: 'h', label: 'Container height', min: 6, max: 20, value: 10 },
      { id: 'pop', label: 'Population size', min: 10, max: 100, step: 5, value: 50 },
      { id: 'mut', label: 'Mutation rate (per item)', min: 0, max: 1, step: 0.05, value: 0.1 },
      { id: 'gens', label: 'Generations', min: 10, max: 300, step: 10, value: 100 },
    ],
    stages: [{ id: 'box', w: 600, h: 460, label: 'Best valid layout so far (or the top-ranked layout while none is valid); dashed rings show the 0.2 margin, red crosses mark overlaps' }],
    chartLabel: 'Best fitness (minus total distance to centre; higher is better)', source: 'packing-optimizer.py',
    onReset(q) {
      p = new Packing({ width: q.w, height: q.h, populationSize: q.pop, mutationRate: q.mut }, new Rng(q.seed));
      p.generateItems(q.kind, q.n); p.maxGen = q.gens;
      p.population = Array.from({ length: p.populationSize }, () => p.randomSolution());
      p.current = p.population[0];
    },
    step: () => p.step(), done: () => p.generation >= p.maxGen, render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Population</b><span>Largest items first, each tried at up to 200 random positions; the first position with no overlap is kept, otherwise the one with fewest overlaps.</span></div>
      <div class="card"><b>Selection</b><span>Two parents drawn at random from the 10 best layouts.</span></div>
      <div class="card"><b>Crossover</b><span>Uniform: each item's position comes from either parent with probability ½.</span></div>
      <div class="card"><b>Mutation</b><span>With the mutation rate, an item gets 5 gaussian moves (σ = 20% of the mean spacing); the move giving the best whole-layout fitness is kept.</span></div>
      <div class="card"><b>Elitism</b><span>The two best layouts are copied unchanged.</span></div></div>
      <p class="small mute">Rectangles can be turned by 90° (mutation flips the turn with probability 0.1), which swaps
      their width and height. Every invalid layout has fitness −∞, so the GA ranks invalid layouts by how much they
      overlap or stick out, which lets evolution push items apart until a valid layout appears; the chart starts then.</p>`,
  });

  function render() {
    const cv = L.canvas('box'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    const f = frame(cv, -0.5, p.W + 0.5, -0.5, p.H + 0.5, 0.04), k = f.scale;
    ctx.strokeStyle = '#e8f1ff'; ctx.lineWidth = 2 * s; ctx.strokeRect(f.X(0), f.Y(p.H), p.W * k, p.H * k);
    const sol = p.best || p.current || [];
    sol.forEach((it, i) => {
      const c = COLORS[i % COLORS.length];
      ctx.fillStyle = c + '88'; ctx.strokeStyle = c; ctx.lineWidth = 1 * s;
      if (it.h === null) {
        ctx.beginPath(); ctx.arc(f.X(it.x), f.Y(it.y), it.w * k, 0, 7); ctx.fill();
        ctx.setLineDash([4 * s, 3 * s]); ctx.beginPath(); ctx.arc(f.X(it.x), f.Y(it.y), (it.w + 0.1) * k, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      } else {
        const [w, h] = footprint(it);
        ctx.fillRect(f.X(it.x), f.Y(it.y + h), w * k, h * k);
        ctx.setLineDash([4 * s, 3 * s]); ctx.strokeRect(f.X(it.x - 0.1), f.Y(it.y + h + 0.1), (w + 0.2) * k, (h + 0.2) * k); ctx.setLineDash([]);
      }
      if (sol.some((o, j) => j !== i && overlaps(it, o))) {
        const [w, h] = footprint(it), x = f.X(it.h === null ? it.x : it.x + w / 2), y = f.Y(it.h === null ? it.y : it.y + h / 2);
        ctx.strokeStyle = '#ff5c7a'; ctx.lineWidth = 2.5 * s; ctx.beginPath();
        ctx.moveTo(x - 6 * s, y - 6 * s); ctx.lineTo(x + 6 * s, y + 6 * s); ctx.moveTo(x + 6 * s, y - 6 * s); ctx.lineTo(x - 6 * s, y + 6 * s); ctx.stroke();
      }
    });
    const area = sol.reduce((a, it) => a + (it.h === null ? Math.PI * it.w * it.w : it.w * it.h), 0);
    L.chart([{ data: p.history, label: 'best fitness' }], { yLabel: 'fitness', x0: 1 });
    L.setReadout([['Generation', `${p.generation} / ${p.maxGen}`], ['Best fitness', p.best ? fmt(p.bestFitness) : 'no valid layout yet'],
      ['Valid layouts in population', p.generation ? `${p.valid} / ${p.populationSize}` : '-'],
      ['Rotated rectangles', sol.filter(it => it.h !== null && it.rot % 180 === 90).length],
      ['Packing density', (area / (p.W * p.H) * 100).toFixed(1) + '%']]);
    L.setStatus(p.generation >= p.maxGen ? (p.best ? 'Finished.' : 'Finished without a valid layout: try fewer items or a bigger container.') : p.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});
