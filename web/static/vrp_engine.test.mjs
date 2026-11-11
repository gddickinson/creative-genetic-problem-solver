// Node test: vrp_engine.js reproduces vehicle-routing-optimizer.py (tests/fixtures/vrp.json).
// Run: node web/static/vrp_engine.test.mjs
import { Rng } from './ga_rng.js';
import { VRP } from './vrp_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('vrp').runs) {
  const v = new VRP({ nLocations: r.n_loc, nVehicles: r.n_veh, speed: r.speed }, new Rng(r.seed));
  near(v.locations.map(l => [l.x, l.y, l.demand, l.tw[0], l.tw[1], l.service, l.priority]), r.locations, 'locations');
  near(v.vehicles.map(x => x.capacity), r.capacities, 'capacities');
  const init = v.initialSolution();
  near(init, r.initial, 'initial solution');
  near([v.cost(init), v.undelivered(init)], [r.initial_cost, r.initial_undelivered], 'cost of initial solution');
  v.populationSize = r.pop;
  for (let g = 0; g < r.gens; g++) v.step();
  near(v.history, r.history, `seed ${r.seed} history`);
  near(v.best, r.best, 'best routes'); near(v.bestFitness, r.best_fitness, 'best fitness');
}
console.log('vrp_engine: ok');
