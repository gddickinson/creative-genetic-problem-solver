// JS port of genetic-solver.py: the twelve 2-D problems of create_creative_problems() and CreativeGeneticSolver
// (uniform initialisation, blend crossover, gaussian mutation, tournament selection, elitism, diversity).
const { sin, cos, exp, sqrt, log, atan2, abs, PI } = Math;
const mod1 = v => ((v % 1) + 1) % 1;            // Python's x % 1 (always non-negative)

export const PROBLEMS = [
  ['Solar Panel Optimization', [[0, 90], [0, 10]], ['tilt angle (deg)', 'height (m)'], x => {
    let total = 0;
    for (let hour = 0; hour < 24; hour++) {
      const sunAngle = sin(hour / 24 * PI);
      const direct = cos(x[0] * PI / 180 - sunAngle), shadow = 1 - exp(-x[1] / 2);
      total += Math.max(0, direct * shadow);
    }
    return total;
  }],
  ['Delivery Route Planning', [[0, 10], [0, 5]], ['warehouse distance (km)', 'coverage radius (km)'], x => {
    const transport = x[0] * 10, area = PI * x[1] ** 2, served = area * exp(-x[0] / 5), maintenance = x[1] ** 2 * 5;
    return served - transport - maintenance;
  }],
  ['Garden Design', [[0, 10], [0, 5]], ['plant density', 'water (L/day/m²)'], x => {
    const overcrowding = exp(-x[0] / 3), water = exp(-((x[1] - 2.5) ** 2) / 2), usage = x[0] * x[1];
    return x[0] * overcrowding * water - usage * 0.1;
  }],
  ['Wind Turbine Layout', [[20, 100], [10, 50]], ['height (m)', 'spacing (m)'], x => {
    const wind = log(x[0] / 2) * 2, wake = exp(-x[1] / 30), cost = x[0] * 0.1 + x[1] * 0.05;
    return wind ** 3 * (1 - wake) - cost;
  }],
  ['Artistic Pattern', [[-4, 4], [-4, 4]], ['x', 'y'], x => sin(5 * x[0]) * cos(5 * x[1]) * exp(-(x[0] ** 2 + x[1] ** 2) / 8)],
  ['Creative Path', [[-5, 5], [-5, 5]], ['x', 'y'],
    x => sin(3 * x[0]) + sin(3 * x[1]) + sin(5 * x[0] + 2 * x[1]) + exp(-(x[0] ** 2 + x[1] ** 2) / 10)],
  ['Mandelbrot Pattern', [[-2, 0.5], [-1.25, 1.25]], ['re(c)', 'im(c)'], x => {
    let zr = 0, zi = 0;
    for (let i = 0; i < 20; i++) {
      [zr, zi] = [zr * zr - zi * zi + x[0], 2 * zr * zi + x[1]];
      if (Math.hypot(zr, zi) > 2) return i / 20;
    }
    return 1.0;
  }],
  ['Spiral Galaxy', [[-4, 4], [-4, 4]], ['x', 'y'], x => {
    const r = sqrt(x[0] ** 2 + x[1] ** 2), theta = atan2(x[1], x[0]);
    return (sin(3 * theta + r) + 1) * exp(-r / 2);
  }],
  ['Sound Wave Synthesis', [[-2, 2], [-2, 2]], ['x', 'y'],
    x => (sin(2 * PI * x[0]) + sin(4 * PI * x[1]) * 0.5) * exp(-abs(x[0]))],
  ['Neural Architecture', [[0, 2], [0, 2]], ['x', 'y'], x => {
    const l1 = abs(sin(x[0] * PI)) * 100, l2 = abs(sin(x[1] * PI)) * 50;
    return sin(l1 / 10) * sin(l2 / 5) * exp(-(l1 + l2) / 100);
  }],
  ['Color Harmony', [[0, 1], [0, 1]], ['hue 1', 'hue 2'], x => {
    const h1 = mod1(x[0]), h2 = mod1(x[1]);
    return cos(2 * PI * (h1 - h2)) * 0.6 + sin(PI * abs(h1 - h2)) * 0.4;
  }],
  ['Particle System', [[-2, 2], [-2, 2]], ['x', 'y'], x => {
    const emission = sin(x[0] * PI), lifetime = cos(x[1] * PI), turbulence = sin(x[0] * x[1] * PI);
    return emission * lifetime * (turbulence + exp(-(x[0] ** 2 + x[1] ** 2) / 4));
  }],
].map(([name, bounds, axes, f]) => ({ name, bounds, axes, f }));

const argmax = a => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);

export class CreativeSolver {
  constructor({ populationSize = 100, mutationRate = 0.1, crossoverRate = 0.7, elitism = 2 } = {}, R) {
    Object.assign(this, { populationSize, mutationRate, crossoverRate, elitism, R });
    this.generation = 0; this.history = []; this.diversityHistory = []; this.bestSolution = null; this.bestFitness = -Infinity;
  }
  // np.random.uniform(lo, hi, N) for each dimension in turn (column-wise).
  initializePopulation(problem) {
    const N = this.populationSize, pop = Array.from({ length: N }, () => [0, 0]);
    problem.bounds.forEach(([lo, hi], d) => this.R.uniformN(lo, hi, N).forEach((v, i) => { pop[i][d] = v; }));
    return pop;
  }
  mutate(ind, problem) {
    const strength = this.R.normalN(0, 0.1, ind.length), mask = this.R.randomN(ind.length).map(u => u < this.mutationRate);
    return ind.map((v, i) => { const [lo, hi] = problem.bounds[i]; return Math.min(hi, Math.max(lo, v + strength[i] * (mask[i] ? 1 : 0))); });
  }
  crossover(p1, p2) {
    if (this.R.random() > this.crossoverRate) return [p1, p2];
    const a = this.R.random();
    return [p1.map((v, i) => a * v + (1 - a) * p2[i]), p1.map((v, i) => (1 - a) * v + a * p2[i])];
  }
  selectParents(pop, fit) {
    const ts = Math.max(2, Math.trunc(0.1 * this.populationSize));
    const i1 = this.R.choiceNoReplace(pop.length, ts), i2 = this.R.choiceNoReplace(pop.length, ts);
    const win = idx => pop[idx[argmax(idx.map(i => fit[i]))]];
    return [win(i1), win(i2)];
  }
  diversity(pop) {
    const n = Math.min(10, pop.length), d = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) d.push(Math.hypot(...pop[i].map((v, k) => v - pop[j][k])));
    return d.length ? d.reduce((a, b) => a + b, 0) / d.length : 0;
  }
  // One frame of evolve()'s update(): evaluate, keep the best, elitism, offspring, record history.
  step(problem) {
    if (!this.population) this.population = this.initializePopulation(problem);
    const pop = this.population, fit = pop.map(problem.f);
    this.fitness = fit;
    const m = argmax(fit);
    if (fit[m] > this.bestFitness) { this.bestFitness = fit[m]; this.bestSolution = pop[m].slice(); }
    const order = fit.map((v, i) => i).sort((a, b) => fit[a] - fit[b]);         // np.argsort (ties: identical copies)
    const next = order.slice(order.length - this.elitism).map(i => pop[i]);
    while (next.length < this.populationSize) {
      const [a, b] = this.selectParents(pop, fit);
      const [c1, c2] = this.crossover(a, b);
      next.push(this.mutate(c1, problem), this.mutate(c2, problem));
    }
    this.population = next.slice(0, this.populationSize);
    this.history.push(this.bestFitness); this.diversityHistory.push(this.diversity(this.population));
    this.generation++;
  }
}
