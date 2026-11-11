// Node test: creative_engine.js reproduces genetic-solver.py (tests/fixtures/creative.json).
// Run: node web/static/creative_engine.test.mjs
import { Rng } from './ga_rng.js';
import { PROBLEMS, CreativeSolver } from './creative_engine.js';
import { fixture, near } from './util.test.mjs';

const F = fixture('creative');
F.values.forEach((v, i) => {
  near(PROBLEMS[i].name, v.name);
  near(v.points.map(PROBLEMS[i].f), v.values, v.name);
});
for (const r of F.runs) {
  const p = r.params, s = new CreativeSolver({ populationSize: p.population_size, mutationRate: p.mutation_rate,
    crossoverRate: p.crossover_rate, elitism: p.elitism }, new Rng(r.seed));
  for (let g = 0; g < r.gens; g++) s.step(PROBLEMS[r.problem]);
  near(s.history, r.history, `${PROBLEMS[r.problem].name} history`);
  near(s.diversityHistory, r.diversity, 'diversity');
  near(s.bestSolution, r.best, 'best'); near(s.bestFitness, r.best_fitness, 'best fitness');
}
console.log('creative_engine: ok');
