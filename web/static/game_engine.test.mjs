// Node test: game_engine.js reproduces game-evolution.py (tests/fixtures/game.json).
// Run: node web/static/game_engine.test.mjs
import { Rng } from './ga_rng.js';
import { GameEvolution } from './game_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('game').runs) {
  const g = new GameEvolution(r.game, { populationSize: r.pop }, new Rng(r.seed));
  for (let k = 0; k < r.gens; k++) g.step();
  near(g.history.map(h => h.dist), r.dist, `${r.game} shares`);
  near(g.history.map(h => h.perf), r.perf, 'performance');
  near(g.population.map(s => s.p), r.final_p, 'final cooperation probabilities');
  near(g.population.map(s => s.name), r.final_names, 'final strategies');
}
console.log('game_engine: ok');
