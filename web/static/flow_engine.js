// JS port of flow-network.py (FlowNetworkVisualizer): random layered network and Ford-Fulkerson max flow with BFS
// augmenting paths in the residual network. Edge and neighbour order follow networkx DiGraph (insertion order).

export class FlowNetwork {
  constructor(R) { this.R = R; }
  generate(nNodes = 8, density = 0.3) {
    const R = this.R, nodes = Array.from({ length: nNodes }, (_, i) => i), source = 0, sink = nNodes - 1;
    this.n = nNodes; this.edges = []; this.index = new Map();
    this.succ = nodes.map(() => []); this.pred = nodes.map(() => []);
    const add = (u, v, cap) => {
      const k = `${u},${v}`;
      if (this.index.has(k)) { this.edges[this.index.get(k)].cap = cap; return; }
      this.index.set(k, this.edges.length); this.succ[u].push(this.edges.length); this.pred[v].push(this.edges.length);
      this.edges.push({ u, v, cap, flow: 0 });
    };
    const layers = this.splitIntoLayers(nodes.slice(1, -1), 3);
    const link = (l1, l2) => { for (const u of l1) { const ts = R.sample(l2, R.randint(1, Math.max(1, l2.length))); for (const v of ts) add(u, v, R.randint(5, 20)); } };
    link([source], layers[0]);
    for (let i = 0; i < layers.length - 1; i++) link(layers[i], layers[i + 1]);
    link(layers[layers.length - 1], [sink]);
    for (let k = 0; k < Math.trunc(nNodes * density); k++) {
      const u = R.choice(nodes.slice(0, -1)), targets = nodes.filter(v => v > u);
      if (targets.length) { const v = R.choice(targets); if (!this.index.has(`${u},${v}`)) add(u, v, R.randint(5, 20)); }
    }
    this.column = nodes.map(() => 0);                    // display only: source, the three layers, sink
    layers.forEach((l, i) => l.forEach(v => { this.column[v] = i + 1; })); this.column[sink] = layers.length + 1;
    return this;
  }
  splitIntoLayers(nodes, nLayers) {
    nodes = nodes.slice();
    const layers = [], per = Math.max(1, Math.floor(nodes.length / nLayers));
    for (let k = 0; k < nLayers - 1; k++) {
      if (nodes.length) { const layer = this.R.sample(nodes, Math.min(per, nodes.length)); layer.forEach(v => nodes.splice(nodes.indexOf(v), 1)); layers.push(layer); }
    }
    if (nodes.length) layers.push(nodes);
    return layers;
  }
  // BFS over forward edges with spare capacity, then backward along edges carrying flow. Returns [[edgeIndex, ±1]].
  augmentingPath(source, sink) {
    const prev = new Map([[source, null]]), queue = [source];
    for (let qi = 0; qi < queue.length; qi++) {
      const u = queue[qi];
      if (u === sink) break;
      for (const e of this.succ[u]) { const E = this.edges[e]; if (!prev.has(E.v) && E.cap - E.flow > 0) { prev.set(E.v, [e, 1]); queue.push(E.v); } }
      for (const e of this.pred[u]) { const E = this.edges[e]; if (!prev.has(E.u) && E.flow > 0) { prev.set(E.u, [e, -1]); queue.push(E.u); } }
    }
    if (!prev.has(sink)) return [];
    const path = [];
    for (let cur = sink; cur !== source;) { const [e, d] = prev.get(cur); path.push([e, d]); cur = d === 1 ? this.edges[e].u : this.edges[e].v; }
    return path.reverse();
  }
  // find_max_flow: returns per-step snapshots { path: [edge indices], flows: [...], value }.
  maxFlow() {
    this.edges.forEach(E => { E.flow = 0; });
    const steps = [];
    let value = 0;
    for (;;) {
      const path = this.augmentingPath(0, this.n - 1);
      if (!path.length) break;
      const m = Math.min(...path.map(([e, d]) => (d === 1 ? this.edges[e].cap - this.edges[e].flow : this.edges[e].flow)));
      path.forEach(([e, d]) => { this.edges[e].flow += d * m; });
      value += m;
      steps.push({ path: path.map(([e]) => e), back: path.filter(([, d]) => d < 0).map(([e]) => e), flows: this.edges.map(E => E.flow), value, push: m });
    }
    this.value = value;
    return steps;
  }
}
