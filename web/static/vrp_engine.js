// JS port of vehicle-routing-optimizer.py (VehicleRoutingOptimizer): capacitated vehicle routing with time windows.
// A solution is an array of routes (location ids, depot = 0 implicit at both ends), one per vehicle.
const asc = (a, b) => (a.f > b.f) - (a.f < b.f);  // stable ascending sort that copes with Infinity

export class VRP {
  constructor({ nLocations = 20, nVehicles = 3, speed = 1.0 } = {}, R) {
    Object.assign(this, { nLocations, nVehicles, speed, R });
    this.depot = { id: 0, x: 50, y: 50, demand: 0, tw: [0, 24], service: 0, priority: 1 };
    this.locations = [this.depot]; this.vehicles = [];
    this.best = null; this.bestFitness = Infinity; this.history = []; this.generation = 0;
    this.generateProblem();
  }
  generateProblem() {
    const R = this.R;
    for (let i = 1; i < this.nLocations; i++) {
      const x = R.uniform(0, 100), y = R.uniform(0, 100), demand = R.uniform(1, 10);
      const start = R.uniform(0, 12), len = R.uniform(4, 8), service = R.uniform(0.2, 1.0), priority = R.randint(1, 3);
      this.locations.push({ id: i, x, y, demand, tw: [start, start + len], service, priority });
    }
    for (let i = 0; i < this.nVehicles; i++) this.vehicles.push({ id: i, capacity: R.uniform(30, 50) });
  }
  dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }
  // calculate_route_metrics: [length, valid]; arrival times use distance / speed, waiting until a window opens.
  routeMetrics(route, capacity) {
    if (!route.length) return [0, true];
    let total = 0, time = 0, load = 0, valid = true, prev = this.depot;
    for (const id of route) {
      const loc = this.locations[id], leg = this.dist(prev, loc);
      total += leg; time += leg / this.speed;
      if (time < loc.tw[0]) time = loc.tw[0]; else if (time > loc.tw[1]) valid = false;
      time += loc.service; load += loc.demand;
      if (load > capacity) valid = false;
      prev = loc;
    }
    return [total + this.dist(prev, this.depot), valid];
  }
  fitness(sol) {
    let total = 0, valid = true;
    sol.forEach((r, v) => { const [d, ok] = this.routeMetrics(r, this.vehicles[v].capacity); total += d; valid = valid && ok; });
    return [total, valid];
  }
  // Cheapest feasible insertion, highest priority first; locations with no feasible slot stay unassigned.
  initialSolution() {
    const routes = this.vehicles.map(() => []);
    const order = Array.from({ length: this.nLocations - 1 }, (_, i) => i + 1)
      .sort((a, b) => this.locations[b].priority - this.locations[a].priority);
    for (const id of order) {
      let bv = -1, bp = -1, bc = Infinity;
      routes.forEach((r, v) => {
        for (let i = 0; i <= r.length; i++) {
          const trial = [...r.slice(0, i), id, ...r.slice(i)], [d, ok] = this.routeMetrics(trial, this.vehicles[v].capacity);
          if (ok && d < bc) { bv = v; bp = i; bc = d; }
        }
      });
      if (bv >= 0) routes[bv].splice(bp, 0, id);
    }
    return routes;
  }
  crossover(p1, p2) {
    const child = this.vehicles.map(() => []), all = p1.flat();
    const cut = this.R.randint(0, all.length), taken = new Set();
    p1.forEach((r, v) => r.slice(0, cut).forEach(id => { child[v].push(id); taken.add(id); }));
    let cur = 0;
    for (const r of p2) for (const id of r) if (!taken.has(id)) { child[cur].push(id); cur = (cur + 1) % child.length; }
    return child;
  }
  mutate(sol, rate = 0.1) {
    const R = this.R;
    if (R.random() > rate) return sol;
    const m = sol.map(r => r.slice()), n = m.length, type = R.choice(['swap', 'reverse', 'relocate']);
    if (type === 'swap') {
      const v1 = R.randint(0, n - 1), v2 = R.randint(0, n - 1);
      if (m[v1].length && m[v2].length) {
        const i1 = R.randint(0, m[v1].length - 1), i2 = R.randint(0, m[v2].length - 1);
        [m[v1][i1], m[v2][i2]] = [m[v2][i2], m[v1][i1]];
      }
    } else if (type === 'reverse') {
      const v = R.randint(0, n - 1);
      if (m[v].length > 2) {
        const i = R.randint(0, m[v].length - 2), j = R.randint(i + 1, m[v].length);
        m[v].splice(i, j - i, ...m[v].slice(i, j).reverse());
      }
    } else {
      const v1 = R.randint(0, n - 1);
      if (m[v1].length) {
        const i = R.randint(0, m[v1].length - 1), v2 = R.randint(0, n - 1);
        if (v1 !== v2 || m[v1].length > 1) { const [loc] = m[v1].splice(i, 1); m[v2].splice(R.randint(0, m[v2].length), 0, loc); }
      }
    }
    return m;
  }
  // One generation of optimize(); invalid solutions score Infinity.
  step() {
    if (!this.population) this.population = Array.from({ length: this.populationSize }, () => this.initialSolution());
    const scored = this.population.map(s => { const [d, ok] = this.fitness(s); return { s, f: ok ? d : Infinity }; }).sort(asc);
    if (scored[0].f < this.bestFitness) { this.bestFitness = scored[0].f; this.best = scored[0].s.map(r => r.slice()); }
    this.current = scored[0].s; this.history.push(scored[0].f);
    const next = scored.slice(0, Math.floor(this.populationSize / 10)).map(x => x.s), top = scored.slice(0, 20);
    const pick = () => { const [a, b] = this.R.sample(top, 2); return (b.f < a.f ? b : a).s; };
    while (next.length < this.populationSize) { const a = pick(), b = pick(); next.push(this.mutate(this.crossover(a, b))); }
    this.population = next; this.generation++;
  }
}
