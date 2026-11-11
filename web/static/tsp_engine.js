// JS port of travelling-sales-person-solver.py (TSPSolver + create_example_problem). Routes are arrays of city
// indices; every operator follows the Python line by line, drawing random numbers from an Rng (ga_rng.js).

// create_example_problem(): 10 cities on a noisy circle of radius 10 around (10, 10) plus 5 near the centre.
export function exampleCities(R, nCities = 15) {
  const cities = [], ring = nCities - 5, cx = 10, cy = 10, radius = 10;
  for (let i = 0; i < ring; i++) {
    const angle = (i / ring) * 2 * Math.PI;
    const x = cx + radius * Math.cos(angle) + R.uniform(-2, 2);
    const y = cy + radius * Math.sin(angle) + R.uniform(-2, 2);
    cities.push({ x, y, name: `City_${i + 1}` });
  }
  for (let i = 0; i < 5; i++) {
    const x = cx + R.uniform(-5, 5), y = cy + R.uniform(-5, 5);
    cities.push({ x, y, name: `City_${nCities - 4 + i}` });
  }
  return cities;
}

export class TSP {
  constructor(cities, { populationSize = 100, mutationRate = 0.01 } = {}, R) {
    Object.assign(this, { cities, populationSize, mutationRate, R });
    this.population = []; this.bestRoute = null; this.bestDistance = Infinity; this.generation = 0; this.history = [];
  }
  initializePopulation() {
    this.population = [];
    for (let k = 0; k < this.populationSize; k++) {
      const route = this.cities.map((_, i) => i); this.R.shuffle(route); this.population.push(route);
    }
  }
  distance(route) {
    let total = 0;
    for (let i = 0; i < route.length; i++) {
      const a = this.cities[route[i]], b = this.cities[route[(i + 1) % route.length]];
      total += Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }
    return total;
  }
  // Order crossover (OX): keep parent1[start..end], fill the rest in parent2's order.
  crossover(p1, p2) {
    const size = p1.length, [s, e] = this.R.sample([...Array(size).keys()], 2).sort((a, b) => a - b);
    const child = new Array(size).fill(null), kept = new Set();
    for (let i = s; i <= e; i++) { child[i] = p1[i]; kept.add(p1[i]); }
    const rest = p2.filter(c => !kept.has(c));
    let j = 0;
    for (let i = 0; i < size; i++) if (child[i] === null) child[i] = rest[j++];
    return child;
  }
  // Swap mutation: with probability mutationRate swap two random positions.
  mutate(route) {
    if (this.R.random() < this.mutationRate) {
      const [i, j] = this.R.sample([...Array(route.length).keys()], 2);
      [route[i], route[j]] = [route[j], route[i]];
    }
    return route;
  }
  minBy(routes) { let best = routes[0], bd = this.distance(best);
    for (const r of routes.slice(1)) { const d = this.distance(r); if (d < bd) { best = r; bd = d; } } return best; }
  selectParent() { return this.minBy(this.R.sample(this.population, 5)); }   // tournament of 5
  // One animation frame of TSPSolver.evolve: elitism (best route copied), then children until full.
  step() {
    if (!this.population.length) this.initializePopulation();
    const next = [this.minBy(this.population).slice()];
    while (next.length < this.populationSize) {
      const a = this.selectParent(), b = this.selectParent();
      next.push(this.mutate(this.crossover(a, b)));
    }
    this.population = next;
    const cur = this.minBy(this.population), d = this.distance(cur);
    if (d < this.bestDistance) { this.bestDistance = d; this.bestRoute = cur.slice(); }
    this.history.push(this.bestDistance);
    this.generation++;
  }
}
