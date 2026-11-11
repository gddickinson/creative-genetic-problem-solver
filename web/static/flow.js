// Flow network page: FlowNetworkVisualizer from flow-network.py (flow_engine.js), replayed one augmenting path at a time.
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { FlowNetwork } from './flow_engine.js';
import { lab } from './lab.js';
import { fitCanvas } from './chart.js';

route('flow', () => {
  let net, steps, i;
  const L = lab({
    kicker: 'Ford-Fulkerson · not a genetic algorithm', title: 'Flow Network Optimizer',
    lead: `How much can flow from the source (node 0) to the sink (the last node) when every pipe has a capacity? The
      network is random but layered (source → three layers → sink, plus a few forward shortcuts, capacities 5-20). The
      maximum flow is found by Ford-Fulkerson: repeatedly find a path with spare capacity by breadth-first search and push
      as much as it allows. This demo is in the collection but uses no genetic algorithm.`,
    controls: [
      { id: 'n', label: 'Number of nodes', min: 4, max: 20, value: 8 },
      { id: 'd', label: 'Edge density (extra shortcut edges = nodes × density)', min: 0, max: 1, step: 0.05, value: 0.3 },
    ],
    speed: 1, maxSpeed: 3, interval: 700,
    stages: [{ id: 'net', w: 600, h: 420, label: 'Edges labelled flow/capacity, thicker and warmer when fuller; the latest augmenting path is red (dashed where it cancels flow)' }],
    chartLabel: 'Total flow after each augmenting path', source: 'flow-network.py',
    onReset(p) { net = new FlowNetwork(new Rng(p.seed)).generate(p.n, p.d); steps = net.maxFlow(); i = 0; },
    step: () => { i++; }, done: () => i >= steps.length, render,
    notes: `<h2>How it works</h2><ol class="flowsteps">
      <li>Breadth-first search from the source over the residual network: forward along edges with spare capacity, and
      backwards along edges that already carry flow (pushing "backwards" cancels earlier flow).</li>
      <li>The smallest residual capacity on the path is pushed along it.</li>
      <li>Repeat until no path remains; the total pushed is the maximum flow, and saturated edges (flow = capacity) are the bottlenecks.</li></ol>
      <p class="small mute">Node positions here are a simple layered layout; the desktop app uses a spring layout. The
      backward step was missing from the original script, which could report less than the true maximum; it was fixed
      while building this page.</p>`,
  });

  function render() {
    const s = i ? steps[i - 1] : null, flows = s ? s.flows : net.edges.map(() => 0);
    const cv = L.canvas('net'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, k = W / 600;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    const cols = Math.max(...net.column) + 1, byCol = Array.from({ length: cols }, () => []);
    net.column.forEach((c, v) => byCol[c].push(v));
    const pos = [];
    byCol.forEach((vs, c) => vs.forEach((v, j) => { pos[v] = [40 * k + (W - 80 * k) * c / (cols - 1), H * (j + 1) / (vs.length + 1)]; }));
    const onPath = new Set(s ? s.path : []), back = new Set(s ? s.back : []);
    ctx.font = `${10 * k}px ui-monospace,Menlo,monospace`; ctx.textAlign = 'center';
    net.edges.forEach((E, e) => {
      const [x1, y1] = pos[E.u], [x2, y2] = pos[E.v], r = flows[e] / E.cap, bend = E.v - E.u > 1 && net.column[E.v] - net.column[E.u] > 1 ? 30 * k : 0;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - bend;
      ctx.strokeStyle = onPath.has(e) ? '#ff3355' : `hsl(${200 - 160 * r},90%,${45 + 15 * r}%)`;
      ctx.lineWidth = (1 + 3 * r) * k; ctx.setLineDash(back.has(e) ? [6 * k, 4 * k] : []);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke(); ctx.setLineDash([]);
      const ax = 0.25 * x1 + 0.5 * mx + 0.25 * x2, ay = 0.25 * y1 + 0.5 * my + 0.25 * y2, ang = Math.atan2(y2 - my, x2 - mx);
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.moveTo(x2 - 13 * k * Math.cos(ang), y2 - 13 * k * Math.sin(ang));
      ctx.lineTo(x2 - 21 * k * Math.cos(ang - 0.35), y2 - 21 * k * Math.sin(ang - 0.35)); ctx.lineTo(x2 - 21 * k * Math.cos(ang + 0.35), y2 - 21 * k * Math.sin(ang + 0.35)); ctx.fill();
      ctx.fillStyle = '#e8f1ff'; ctx.fillText(`${flows[e]}/${E.cap}`, ax, ay - 3 * k);
    });
    pos.forEach(([x, y], v) => { ctx.fillStyle = v === 0 ? '#b6ff3d' : v === net.n - 1 ? '#ff4fd8' : '#3dd6ff';
      ctx.beginPath(); ctx.arc(x, y, 11 * k, 0, 7); ctx.fill(); ctx.fillStyle = '#060814'; ctx.fillText(String(v), x, y + 4 * k); });
    L.chart([{ data: [0, ...steps.slice(0, i).map(x => x.value)], label: 'flow' }], { yLabel: 'flow value', xLabel: 'Augmenting paths', x0: 0 });
    const done = i >= steps.length, sat = net.edges.filter((E, e) => flows[e] === E.cap);
    const util = net.edges.reduce((a, E, e) => a + flows[e] / E.cap, 0) / net.edges.length * 100;
    L.setReadout([['Nodes / edges', `${net.n} / ${net.edges.length}`], ['Augmenting paths', `${i} / ${steps.length}`],
      ['Flow so far', s ? s.value : 0], ['Maximum flow', done ? net.value : '…'],
      ['Bottleneck edges (flow = capacity)', sat.map(E => `${E.u}→${E.v}`).join(' ') || '-'], ['Average edge utilisation', util.toFixed(1) + '%']]);
    L.setStatus(done ? 'Maximum flow reached: no augmenting path left.' : i ? 'Pushing flow...' : 'Press Step to push flow along the first path.');
  }
});
