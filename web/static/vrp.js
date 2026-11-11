// Vehicle routing page: VehicleRoutingOptimizer from vehicle-routing-optimizer.py (vrp_engine.js).
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { VRP } from './vrp_engine.js';
import { lab } from './lab.js';
import { fitCanvas, frame, fmt, COLORS } from './chart.js';

route('vrp', () => {
  let v;
  const L = lab({
    kicker: 'Routing GA · capacities · time windows', title: 'Vehicle Router',
    lead: `Deliver from a central depot to random locations on a 100 × 100 map with a small fleet. Each location has a
      demand, a time window (opening between hour 0 and 12, lasting 4-8 h), a service time and a priority; each vehicle has a
      capacity of 30-50. A solution is one route per vehicle; its cost is the total distance plus 1,000 for every
      location left undelivered, or ∞ if any route breaks a capacity or arrives after a window closes (lower is better).`,
    controls: [
      { id: 'n', label: 'Delivery locations (incl. depot)', min: 5, max: 40, value: 20 },
      { id: 'veh', label: 'Vehicles', min: 1, max: 6, value: 3 },
      { id: 'speed', label: 'Vehicle speed, map units per hour (desktop default 1)', min: 1, max: 40, value: 15 },
      { id: 'pop', label: 'Population size', min: 10, max: 150, step: 10, value: 50 },
      { id: 'gens', label: 'Generations', min: 10, max: 300, step: 10, value: 100 },
    ].map(c => (c.id === 'speed' ? { ...c, id: 'vspeed' } : c)),
    stages: [{ id: 'map', w: 600, h: 460, label: 'Best routes so far: depot (white square), locations by priority (high: large), unassigned locations grey' },
      { id: 'tw', w: 600, h: 260, label: 'Time windows (bars) and arrival times (dots, before any wait for a window to open, as in the desktop plot) along the best routes' }],
    chartLabel: 'Best cost in each generation: distance + 1000 per undelivered location (∞ = no valid solution)', source: 'vehicle-routing-optimizer.py',
    onReset(p) {
      v = new VRP({ nLocations: p.n, nVehicles: p.veh, speed: p.vspeed }, new Rng(p.seed));
      v.populationSize = p.pop; v.maxGen = p.gens;
      v.population = Array.from({ length: v.populationSize }, () => v.initialSolution());
      v.current = v.population[0];
    },
    step: () => v.step(), done: () => v.generation >= v.maxGen, render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Population</b><span>Cheapest-insertion heuristic, highest priority first: each location goes where it adds least distance while keeping its route valid. It has no randomness, so every individual starts identical.</span></div>
      <div class="card"><b>Selection</b><span>Two-way tournaments among the 20 best solutions.</span></div>
      <div class="card"><b>Crossover</b><span>Keep each of parent 1's routes up to a random cut, then deal parent 2's remaining locations round-robin over the vehicles.</span></div>
      <div class="card"><b>Mutation</b><span>With probability 0.1: swap two locations, reverse part of a route, or move a location to another position or vehicle.</span></div>
      <div class="card"><b>Elitism</b><span>The best 10% survive unchanged.</span></div></div>
      <div class="callout"><b>Vehicle speed.</b> The script originally took one map unit to take one hour (speed 1, still its
      default). Legs are then tens of hours long while windows close by hour 20, so almost no location can be served and
      the routes stay empty; this page therefore starts at speed 15. Set it to 1 to see the desktop behaviour. Every
      undelivered location adds 1,000 to the cost, so a solution that serves more locations always ranks higher; note
      that the operators only rearrange locations the first heuristic placed, so they never add a missing one back.</div>`,
  });

  function render() {
    const sol = v.best || v.current || v.vehicles.map(() => []);
    drawMap(L.canvas('map'), v, sol); drawWindows(L.canvas('tw'), v, sol);
    const served = new Set(sol.flat());
    L.chart([{ data: v.history.map(f => (Number.isFinite(f) ? f : NaN)), label: 'best cost' }], { yLabel: 'cost', x0: 1 });
    L.setReadout([['Generation', `${v.generation} / ${v.maxGen}`], ['Best cost', v.best ? fmt(v.bestFitness) : '-'], ['Distance', fmt(v.fitness(sol)[0])],
      ['Locations served', `${served.size} / ${v.nLocations - 1}`],
      ...sol.map((r, i) => [`Vehicle ${i}`, `${r.length} stops, load ${(r.reduce((a, id) => a + v.locations[id].demand, 0) / v.vehicles[i].capacity * 100).toFixed(0)}%`])]);
    L.setStatus(v.generation >= v.maxGen ? 'Finished.' : v.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});

function drawMap(cv, v, sol) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600, f = frame(cv, 0, 100, 0, 100, 0.05);
  ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#212c4c'; ctx.strokeRect(f.X(0), f.Y(100), 100 * f.scale, 100 * f.scale);
  sol.forEach((r, i) => {
    if (!r.length) return;
    ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 2 * s; ctx.beginPath(); ctx.moveTo(f.X(50), f.Y(50));
    r.forEach(id => ctx.lineTo(f.X(v.locations[id].x), f.Y(v.locations[id].y))); ctx.lineTo(f.X(50), f.Y(50)); ctx.stroke();
  });
  const owner = new Map(); sol.forEach((r, i) => r.forEach(id => owner.set(id, i)));
  ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`;
  v.locations.slice(1).forEach(l => {
    ctx.fillStyle = owner.has(l.id) ? COLORS[owner.get(l.id) % COLORS.length] : '#55607e';
    ctx.beginPath(); ctx.arc(f.X(l.x), f.Y(l.y), (2.5 + 1.6 * l.priority) * s, 0, 7); ctx.fill();
    ctx.fillStyle = '#94a3c6'; ctx.fillText(String(l.id), f.X(l.x) + 7 * s, f.Y(l.y) - 5 * s);
  });
  ctx.fillStyle = '#fff'; ctx.fillRect(f.X(50) - 6 * s, f.Y(50) - 6 * s, 12 * s, 12 * s);
}

function drawWindows(cv, v, sol) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600, n = v.nLocations - 1;
  ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
  const x0 = 40 * s, x1 = W - 10 * s, tMax = 24, X = t => x0 + (x1 - x0) * Math.min(t, tMax) / tMax, rowH = (H - 26 * s) / n;
  ctx.font = `${9 * s}px ui-monospace,Menlo,monospace`; ctx.fillStyle = '#94a3c6'; ctx.textAlign = 'center';
  for (let t = 0; t <= tMax; t += 4) { ctx.fillText(`${t}h`, X(t), H - 6 * s); ctx.fillStyle = '#212c4c'; ctx.fillRect(X(t), 0, 1, H - 20 * s); ctx.fillStyle = '#94a3c6'; }
  const color = new Map(), arrive = new Map();
  sol.forEach((r, i) => { let t = 0, prev = v.depot;
    r.forEach(id => { const l = v.locations[id]; t += v.dist(prev, l) / v.speed; arrive.set(id, t); color.set(id, COLORS[i % COLORS.length]); t += l.service; prev = l; }); });
  v.locations.slice(1).forEach((l, k) => {
    const y = 4 * s + k * rowH;
    ctx.fillStyle = (color.get(l.id) || '#55607e') + '55'; ctx.fillRect(X(l.tw[0]), y, X(l.tw[1]) - X(l.tw[0]), Math.max(2, rowH - 2 * s));
    ctx.fillStyle = '#94a3c6'; ctx.textAlign = 'right'; ctx.fillText(String(l.id), x0 - 4 * s, y + rowH * 0.75);
    if (arrive.has(l.id)) { const a = arrive.get(l.id); ctx.fillStyle = color.get(l.id); ctx.beginPath(); ctx.arc(X(a), y + rowH / 2, 3 * s, 0, 7); ctx.fill();
      if (a > tMax) { ctx.textAlign = 'left'; ctx.fillText(`${a.toFixed(0)}h →`, X(tMax) - 34 * s, y + rowH * 0.75); } }
  });
}
