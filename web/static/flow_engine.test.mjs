// Node test: flow_engine.js reproduces flow-network.py (tests/fixtures/flow.json).
// Run: node web/static/flow_engine.test.mjs
import { Rng } from './ga_rng.js';
import { FlowNetwork } from './flow_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('flow').runs) {
  const f = new FlowNetwork(new Rng(r.seed)).generate(r.n, r.density);
  const order = r.edges.map(([u, v]) => f.index.get(`${u},${v}`));       // networkx lists edges grouped by tail node
  near(order.map(e => [f.edges[e].u, f.edges[e].v, f.edges[e].cap]), r.edges, `seed ${r.seed} edges`);
  const steps = f.maxFlow();
  near(steps.map(s => s.path.map(e => [f.edges[e].u, f.edges[e].v])), r.paths, 'augmenting paths');
  near(steps.map(s => order.map(e => s.flows[e])), r.flows, 'flows');
  near(f.value, r.value, 'max flow');
}
console.log('flow_engine: ok');
