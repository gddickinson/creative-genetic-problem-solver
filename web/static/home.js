// Home page: hero with a live TSP run, counts from the API, cards for every solver.
import { route, api, app, esc, list, animate } from './core.js';
import { Rng } from './ga_rng.js';
import { TSP, exampleCities } from './tsp_engine.js';
import { fitCanvas, frame } from './chart.js';

route('home', async () => {
  const [info, cat] = await Promise.all([api('/info'), api('/problems')]);
  app().innerHTML = `
  <section class="hero two" style="align-items:center">
    <div><p class="kicker">Genetic algorithms · live in your browser</p>
      <h1 class="glow">Creative Genetic Algorithm Visualizers</h1>
      <p class="lead">A collection of interactive visualizations demonstrating genetic algorithms solving various complex
      problems. Every solver from the Python repository runs here in your browser, ported line by line and checked
      against the Python: change the parameters and watch the population evolve.</p>
      <a class="btn" href="#/tsp">Watch a GA solve the TSP</a><a class="btn alt" href="#/ga">How GAs work</a></div>
    <figure class="fig"><canvas id="hero" width="560" height="400" role="img" aria-label="Live travelling-salesperson GA: the best tour shortens as generations pass"></canvas>
      <figcaption class="small mute" id="herocap"></figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.solvers}</b><span>solvers ported</span></div>
    <div class="tile"><b>${info.ga_solvers}</b><span>genetic algorithms</span></div>
    <div class="tile"><b>${info.creative_problems}</b><span>landscapes in the generic solver</span></div>
    <div class="tile"><b>0</b><span>installs needed</span></div>
  </div>
  <h2>Genetic algorithms</h2>
  <div class="grid wide">${list(cat.filter(c => c.is_ga), card)}</div>
  <h2>Also in the collection</h2>
  <div class="grid wide">${list(cat.filter(c => !c.is_ga), card)}</div>`;
  hero();
});

const card = c => `<a class="card door" href="#/${c.route}"><b>${esc(c.title)}</b><span>${esc(c.summary)}</span>
  <span class="tag c">${esc(c.script)}</span></a>`;

function hero() {
  const cv = document.getElementById('hero'), cap = document.getElementById('herocap');
  let R, t, hold = 0;
  const reset = () => { R = new Rng(1 + Math.floor(Math.random() * 999)); t = new TSP(exampleCities(R), { populationSize: 60 }, R); t.initializePopulation(); };
  const draw = () => {
    const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 560;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    const f = frame(cv, -3, 23, -3, 23, 0.05);
    if (t.bestRoute) { ctx.strokeStyle = '#3dd6ff'; ctx.lineWidth = 2.5 * s; ctx.beginPath();
      t.bestRoute.forEach((i, k) => { const c = t.cities[i]; k ? ctx.lineTo(f.X(c.x), f.Y(c.y)) : ctx.moveTo(f.X(c.x), f.Y(c.y)); }); ctx.closePath(); ctx.stroke(); }
    t.cities.forEach(c => { ctx.fillStyle = '#b6ff3d'; ctx.beginPath(); ctx.arc(f.X(c.x), f.Y(c.y), 5 * s, 0, 7); ctx.fill(); });
    cap.textContent = `Generation ${t.generation}: best tour ${t.bestDistance.toFixed(1)} (population 60, order crossover, tournament of 5)`;
  };
  reset(); t.step();
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { for (let g = 0; g < 80; g++) t.step(); draw(); return; }
  let tick = 0;
  animate(() => {
    if (++tick % 4) return;
    if (t.generation >= 120) { if (++hold > 40) { reset(); hold = 0; } } else t.step();
    draw();
  });
}
