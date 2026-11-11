// Generic problem solver page: CreativeGeneticSolver from genetic-solver.py on any of its twelve 2-D problems,
// shown over the problem's fitness landscape (creative_engine.js).
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { PROBLEMS, CreativeSolver } from './creative_engine.js';
import { lab } from './lab.js';
import { fitCanvas, lineChart, fmt } from './chart.js';
import { grid, paint } from './heat.js';

route('creative', (args, query) => {
  let s, problem, bg;
  const start = Math.max(0, PROBLEMS.findIndex(p => p.name === query.p));
  const L = lab({
    kicker: 'Real-valued GA · blend crossover · gaussian mutation', title: 'Generic Problem Solver',
    lead: `<code>CreativeGeneticSolver</code> searches a two-dimensional box for the point of highest fitness. The repo
      ships twelve problems: four "practical" toy models (solar panel, delivery, garden, wind turbine) and eight
      "creative" landscapes. The background shows the landscape (bright = high fitness); dots are the population.`,
    controls: [
      { id: 'problem', label: 'Problem', type: 'select', value: start, options: PROBLEMS.map((p, i) => [i, p.name]) },
      { id: 'pop', label: 'Population size', min: 10, max: 300, step: 10, value: 100 },
      { id: 'mut', label: 'Mutation rate (per gene)', min: 0, max: 1, step: 0.05, value: 0.1 },
      { id: 'cx', label: 'Crossover rate', min: 0, max: 1, step: 0.05, value: 0.7 },
      { id: 'elite', label: 'Elitism (individuals kept)', min: 0, max: 10, value: 2 },
      { id: 'gens', label: 'Generations', min: 10, max: 300, step: 10, value: 50 },
    ],
    stages: [{ id: 'land', w: 600, h: 440, label: 'Fitness landscape with the population (pink) and best solution (cyan star)' },
      { id: 'div', w: 600, h: 180, label: 'Population diversity (mean distance between the first 10 individuals)' }],
    chartLabel: 'Best fitness (higher is better)', source: 'genetic-solver.py',
    onReset(p) {
      const prev = problem; problem = PROBLEMS[+p.problem];
      if (prev !== problem || !bg) bg = paint(grid(problem.f, problem.bounds[0], problem.bounds[1], 140));
      s = new CreativeSolver({ populationSize: p.pop, mutationRate: p.mut, crossoverRate: p.cx, elitism: p.elite }, new Rng(p.seed));
      s.population = s.initializePopulation(problem); s.maxGen = p.gens;
    },
    step: () => s.step(problem), done: () => s.generation >= s.maxGen, render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Population</b><span>Points drawn uniformly inside the problem's bounds.</span></div>
      <div class="card"><b>Selection</b><span>Two tournaments, each of max(2, 10% of the population) distinct individuals; the fittest of each becomes a parent.</span></div>
      <div class="card"><b>Crossover</b><span>With the crossover rate, blend: child = α·p1 + (1-α)·p2 and its mirror, α uniform in [0, 1]; otherwise the parents pass through.</span></div>
      <div class="card"><b>Mutation</b><span>Each coordinate, with the mutation rate, gets gaussian noise (σ = 0.1, the same in every problem's units) and is clipped to the bounds.</span></div>
      <div class="card"><b>Elitism</b><span>The top individuals (default 2) are copied into the next generation.</span></div></div>
      <p class="small mute">The mutation step is 0.1 whatever the problem's scale, so on wide boxes such as the wind turbine
      (heights 20-100 m) new points come almost only from crossover. The desktop script solves each problem for 50 generations.</p>`,
  });

  function render() {
    const cv = L.canvas('land'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, sc = W / 600;
    const pad = 44 * sc, w = W - pad - 8 * sc, h = H - pad - 8 * sc;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(bg, pad, 8 * sc, w, h);
    const [[x0, x1], [y0, y1]] = problem.bounds, X = v => pad + w * (v - x0) / (x1 - x0), Y = v => 8 * sc + h * (y1 - v) / (y1 - y0);
    for (const ind of s.population) { ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.beginPath(); ctx.arc(X(ind[0]), Y(ind[1]), 2.6 * sc, 0, 7); ctx.fill(); }
    if (s.bestSolution) star(ctx, X(s.bestSolution[0]), Y(s.bestSolution[1]), 9 * sc);
    ctx.fillStyle = '#94a3c6'; ctx.font = `${11 * sc}px ui-monospace,Menlo,monospace`; ctx.textAlign = 'center';
    ctx.fillText(`${problem.axes[0]}  [${x0}, ${x1}]`, pad + w / 2, H - 10 * sc);
    ctx.save(); ctx.translate(14 * sc, 8 * sc + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`${problem.axes[1]}  [${y0}, ${y1}]`, 0, 0); ctx.restore();
    L.chart([{ data: s.history, label: 'best fitness' }], { yLabel: 'fitness', x0: 1 });
    lineChart(L.canvas('div'), [{ data: s.diversityHistory, label: 'diversity', color: '#3dd6ff' }], { yLabel: 'diversity', x0: 1 });
    L.setReadout([['Problem', problem.name], ['Generation', `${s.generation} / ${s.maxGen}`], ['Best fitness', fmt(s.bestFitness)],
      ['Best solution', s.bestSolution ? s.bestSolution.map(fmt).join(', ') : '-'],
      ['Diversity', s.diversityHistory.length ? fmt(s.diversityHistory.at(-1)) : '-']]);
    L.setStatus(s.generation >= s.maxGen ? 'Finished.' : s.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});

function star(ctx, x, y, r) {
  ctx.fillStyle = '#3dd6ff'; ctx.strokeStyle = '#060814'; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
  ctx.closePath(); ctx.fill(); ctx.stroke();
}
