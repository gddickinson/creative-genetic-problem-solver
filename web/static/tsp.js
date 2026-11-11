// Travelling salesperson page: live GA from travelling-sales-person-solver.py (tsp_engine.js).
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { TSP, exampleCities } from './tsp_engine.js';
import { lab } from './lab.js';
import { fitCanvas, frame, fmt } from './chart.js';

route('tsp', () => {
  let t;
  const L = lab({
    kicker: 'Permutation GA · order crossover · swap mutation', title: 'Travelling Salesperson',
    lead: `Find the shortest closed tour through 15 cities (ten on a rough circle, five in the middle, as in
      <code>create_example_problem()</code>). Each individual is an ordering of the cities; its fitness is the total tour length.`,
    controls: [
      { id: 'pop', label: 'Population size', min: 10, max: 300, step: 10, value: 100 },
      { id: 'gens', label: 'Generations', min: 10, max: 500, step: 10, value: 100 },
      { id: 'mut', label: 'Mutation rate (swap probability)', min: 0, max: 0.5, step: 0.01, value: 0.01 },
    ],
    stages: [{ id: 'map', w: 600, h: 420, label: 'Best route so far (cyan) and the current population (faint)' }],
    chartLabel: 'Best distance (lower is better)',
    source: 'travelling-sales-person-solver.py',
    onReset(p) {
      const R = new Rng(p.seed);
      t = new TSP(exampleCities(R), { populationSize: p.pop, mutationRate: p.mut }, R);
      t.initializePopulation(); t.maxGen = p.gens;
    },
    step: () => t.step(),
    done: () => t.generation >= t.maxGen,
    render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Population</b><span>Random shuffles of the city list.</span></div>
      <div class="card"><b>Selection</b><span>Tournament: sample 5 routes, keep the shortest.</span></div>
      <div class="card"><b>Crossover</b><span>Order crossover (OX): copy a random slice of parent 1, fill the other positions with the remaining cities in parent 2's order.</span></div>
      <div class="card"><b>Mutation</b><span>With the mutation probability (desktop default 0.01), swap two cities.</span></div>
      <div class="card"><b>Elitism</b><span>The shortest route of each generation is copied unchanged into the next.</span></div></div>`,
  });

  function render() {
    const cv = L.canvas('map'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    const xs = t.cities.map(c => c.x), ys = t.cities.map(c => c.y);
    const f = frame(cv, Math.min(...xs) - 1, Math.max(...xs) + 1, Math.min(...ys) - 1, Math.max(...ys) + 1, 0.08);
    const tour = (r, color, w) => { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath();
      r.forEach((i, k) => { const c = t.cities[i]; k ? ctx.lineTo(f.X(c.x), f.Y(c.y)) : ctx.moveTo(f.X(c.x), f.Y(c.y)); }); ctx.closePath(); ctx.stroke(); };
    t.population.slice(0, 30).forEach(r => tour(r, 'rgba(255,79,216,.07)', 1 * s));
    if (t.bestRoute) tour(t.bestRoute, '#3dd6ff', 2.5 * s);
    ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`;
    t.cities.forEach(c => { ctx.fillStyle = '#b6ff3d'; ctx.beginPath(); ctx.arc(f.X(c.x), f.Y(c.y), 5 * s, 0, 7); ctx.fill();
      ctx.fillStyle = '#94a3c6'; ctx.fillText(c.name.replace('City_', ''), f.X(c.x) + 7 * s, f.Y(c.y) - 6 * s); });
    L.chart([{ data: t.history, label: 'best distance' }], { yLabel: 'distance', x0: 1 });
    L.setReadout([['Generation', `${t.generation} / ${t.maxGen}`], ['Best distance', fmt(t.bestDistance)],
      ['Population', t.populationSize], ['Best route', t.bestRoute ? t.bestRoute.map(i => i + 1).join('-') : '-']]);
    L.setStatus(t.generation >= t.maxGen ? 'Finished.' : t.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});
