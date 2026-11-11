// JS port of game-evolution.py (Strategy + GameEvolutionVisualizer): strategies play every other member of the
// population for 10 rounds, then reproduce in proportion to their (shifted) total payoff, with occasional mutation
// of their cooperation probability. Moves: true = cooperate (dove / hunt stag), false = defect.

export const GAMES = {
  prisoners_dilemma: { name: "Prisoner's Dilemma", desc: 'Classic cooperation vs. defection', pay: { CC: [3, 3], CD: [0, 5], DC: [5, 0], DD: [1, 1] } },
  hawk_dove: { name: 'Hawk-Dove Game', desc: 'Competition vs. peaceful coexistence', pay: { CC: [0, 0], CD: [1, 3], DC: [3, 1], DD: [-2, -2] } },
  stag_hunt: { name: 'Stag Hunt', desc: 'Coordination and trust', pay: { CC: [4, 4], CD: [0, 3], DC: [3, 0], DD: [2, 2] } },
};

// name, cooperate_prob, memory_length, mutation_rate (initialize_strategies)
export const STRATEGIES = [['Cooperative', 0.9, 1, 0.1], ['Defective', 0.1, 1, 0.1], ['Tit-for-Tat', 0.5, 1, 0.1],
  ['Grudger', 0.7, 3, 0.1], ['Random', 0.5, 0, 0.1]].map(([name, p, mem, mut]) => ({ name, p, mem, mut }));

export function decide(s, history, R) {
  if (!history.length || history.length < s.mem) return R.random() < s.p;
  const recent = s.mem === 0 ? history : history.slice(-s.mem);          // Python history[-0:] is the whole list
  const rate = recent.reduce((a, b) => a + b, 0) / recent.length;
  return R.random() < s.p * (1 + rate) / 2;
}

export function playGame(s1, s2, pay, R, rounds = 10) {
  const h1 = [], h2 = [];
  let t1 = 0, t2 = 0;
  for (let k = 0; k < rounds; k++) {
    const m1 = decide(s1, h2, R), m2 = decide(s2, h1, R), [p1, p2] = pay[(m1 ? 'C' : 'D') + (m2 ? 'C' : 'D')];
    t1 += p1; t2 += p2; h1.push(m1 ? 1 : 0); h2.push(m2 ? 1 : 0);
  }
  return [t1 / rounds, t2 / rounds];
}

export class GameEvolution {
  constructor(game, { populationSize = 100 } = {}, R) {
    Object.assign(this, { pay: GAMES[game].pay, populationSize, R });
    this.population = R.choices(STRATEGIES, populationSize);
    this.history = []; this.generation = 0;
  }
  step() {
    const N = this.populationSize, pop = this.population, R = this.R;
    let payoffs = new Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) payoffs[i] += playGame(pop[i], pop[j], this.pay, R)[0];
    const mn = Math.min(...payoffs);
    payoffs = payoffs.map(v => v - mn + 1e-10);
    const dist = {}, perf = {}, coop = {};
    for (const s of STRATEGIES) {
      const idx = pop.map((p, i) => (p.name === s.name ? i : -1)).filter(i => i >= 0);
      dist[s.name] = idx.length / N;
      perf[s.name] = idx.length ? idx.reduce((a, i) => a + payoffs[i], 0) / idx.length : 0;
      coop[s.name] = idx.length ? idx.reduce((a, i) => a + pop[i].p, 0) / idx.length : null;
    }
    const total = payoffs.reduce((a, b) => a + b, 0), probs = payoffs.map(v => v / total), next = [];
    for (let k = 0; k < N; k++) {
      const parent = pop[R.choiceP(probs)];
      let p = parent.p;
      if (R.random() < parent.mut) p = Math.min(1, Math.max(0, p + R.gauss(0, 0.1)));
      next.push({ ...parent, p });
    }
    this.population = next;
    this.history.push({ dist, perf, coop });
    this.generation++;
  }
}
