// Node test: tsp_engine.js reproduces travelling-sales-person-solver.py (tests/fixtures/tsp.json).
// Run: node web/static/tsp_engine.test.mjs
import { Rng } from './ga_rng.js';
import { TSP, exampleCities } from './tsp_engine.js';
import { fixture, near } from './util.test.mjs';

const F = fixture('tsp');
for (const run of F.runs) {
  const R = new Rng(run.seed), cities = exampleCities(R);
  near(cities.map(c => [c.x, c.y, c.name]), run.cities, 'cities');
  const t = new TSP(cities, { populationSize: run.pop, mutationRate: run.mutation_rate ?? 0.01 }, R);
  t.initializePopulation();
  for (let g = 0; g < run.gens; g++) t.step();
  near(t.history, run.history, `seed ${run.seed} history`);
  near(t.bestRoute, run.best_route, 'best route');
  near(t.bestDistance, run.best_distance, 'best distance');
}
{
  const O = F.operators, R = new Rng(5), cities = exampleCities(R);
  const t = new TSP(cities, { populationSize: 12 }, R);
  t.initializePopulation();
  near(t.population, O.population, 'population');
  for (const s of O.steps) {
    const a = t.selectParent(), b = t.selectParent(), child = t.crossover(a, b);
    near([a, b, child], [s.a, s.b, s.child], 'select/crossover');
    t.mutationRate = 1.0; near(t.mutate(child.slice()), s.mutated, 'mutate'); t.mutationRate = 0.01;
    near(t.distance(child), s.distance, 'distance');
  }
}
console.log('tsp_engine: ok');
