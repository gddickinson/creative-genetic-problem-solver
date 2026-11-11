// Node test: portfolio_engine.js reproduces portfolio-optimizer.py (tests/fixtures/portfolio.json).
// Run: node web/static/portfolio_engine.test.mjs
import { Rng } from './ga_rng.js';
import { generateSampleData, PortfolioGA } from './portfolio_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('portfolio').runs) {
  const R = new Rng(1), market = generateSampleData(r.n_assets, R);
  near(market.expected_returns, r.expected_returns, 'returns'); near(market.cov_matrix, r.cov_matrix, 'covariance');
  const ga = new PortfolioGA(market, { populationSize: r.pop }, R);
  near(ga.efficientFrontier(r.frontier_points), r.frontier, 'frontier');
  for (let g = 0; g < r.gens; g++) ga.step();
  near(ga.history.map(p => p.sharpe), r.sharpe, 'sharpe history');
  near(ga.best.weights, r.best_weights, 'best weights');
  near([ga.best.ret, ga.best.vol], [r.best_return, r.best_volatility], 'best metrics');
}
console.log('portfolio_engine: ok');
