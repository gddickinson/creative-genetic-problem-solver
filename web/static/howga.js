// "How GAs work": the generic loop and operators exactly as this repository implements them (ga_core.run_ga and the
// solver scripts' own operators), with a table comparing the eight GA solvers.
import { route, app, esc, list } from './core.js';

const LOOP = [
  ['Initialise', 'Create a population of random candidate solutions (random routes, points, weights, layouts...).'],
  ['Evaluate', 'Score every individual with the problem\'s fitness function (tour length, Sharpe ratio, accuracy...).'],
  ['Keep the elite', 'Copy the best few individuals unchanged into the next generation (elitism), so the best solution is never lost.'],
  ['Select parents', 'Pick parents with a bias towards fitness: tournaments (the best of a small random group) in most solvers, fitness-proportional (roulette) sampling in the game-theory demo.'],
  ['Crossover', 'Combine two parents into children: a blend of numbers, an order-preserving splice of a permutation, a mix of genes taken from either parent.'],
  ['Mutate', 'Make small random changes (gaussian noise, swapping two cities, moving one item) so the search keeps exploring.'],
  ['Repeat', 'The children form the next generation; loop for a fixed number of generations and report the best individual found.'],
];

// [page, solver, genome, fitness, selection, crossover, mutation, elitism]
const TABLE = [
  ['tsp', 'Travelling salesperson', 'order of the 15 cities', 'tour length (minimise)', 'tournament of 5', 'order crossover (OX)', 'swap two cities, p 0.01', 'best 1'],
  ['creative', 'Generic problem solver', 'point (x, y) in a box', 'one of 12 landscape functions (maximise)', 'tournament of 10% of the population', 'blend α·p1 + (1−α)·p2, p 0.7', 'gaussian σ 0.1 per gene, p 0.1, clipped', 'best 2'],
  ['vrp', 'Vehicle routing', 'one route per vehicle', 'total distance, ∞ if a capacity or time window is broken', '2-way tournament in top 20', 'prefix of parent 1 + rest of parent 2', 'swap / reverse / relocate, p 0.1', 'best 10%'],
  ['packing', 'Packing', 'position of every item', '−(distance to centre), −∞ if overlapping', 'random pair from top 10', 'each item from either parent', '5 gaussian moves, keep the best, p 0.1 per item', 'best 2'],
  ['portfolio', 'Portfolio', 'asset weights summing to 1', 'Sharpe ratio (maximise)', '2-way tournament in top 20', 'average of the parents', 'gaussian σ 0.1, p 0.1, renormalise', 'best 10%'],
  ['neural', 'Neural network', 'all weights and biases', 'training accuracy', 'random pair from top 10', 'each weight from either parent', 'gaussian σ 0.1, p 0.1 per weight', 'best 2'],
  ['game', 'Game theory', 'a strategy and its cooperation probability', 'total payoff against everyone', 'fitness-proportional (roulette)', 'none (asexual copies)', 'cooperation probability ± N(0, 0.1), p 0.1', 'none'],
  ['maze', 'Maze pathfinder', 'a walk from the start', '+1000 at goal − distance left − length/2 − 2·revisits', '2-way tournament in top 20', 'splice at a shared cell', 'regrow the tail from a random point, p 0.1', 'best 2'],
];

const CORE = [
  ['tournament_selection', 'best of a random group (size 5)'], ['roulette_selection', 'probability proportional to shifted fitness'],
  ['blend_crossover', 'α-blend of two real vectors'], ['order_crossover', 'OX for permutations'],
  ['gaussian_mutation', 'per-gene gaussian noise with bounds'], ['swap_mutation', 'swap two positions'],
  ['elitist_replacement', 'keep the best k'], ['calculate_diversity', 'mean pairwise distance of a sample'],
  ['run_ga', 'the generic loop above, configured by GAConfig'],
];

route('ga', () => {
  app().innerHTML = `
  <section class="hero"><p class="kicker">Evolution as a search method</p><h1 class="glow">How genetic algorithms work</h1>
    <p class="lead">A genetic algorithm searches for good solutions the way breeding improves a population: keep many
    candidate solutions, let the better ones have more offspring, mix their features and add small random changes.
    Every solver in this repository runs a version of the same loop.</p></section>
  <h2>The loop</h2><ol class="flowsteps">${list(LOOP, ([t, d]) => `<li><b>${esc(t)}.</b> ${esc(d)}</li>`)}</ol>
  <div class="callout a">The ingredients that change from problem to problem are the <b>genome</b> (how a solution is
    written down), the <b>fitness</b> function, and crossover and mutation operators that keep children valid, e.g. a
    route must still visit every city exactly once, so TSP uses order crossover rather than cutting two lists in half.</div>
  <h2>The eight GA solvers side by side</h2>
  <div class="scroll"><table><tr><th>Solver</th><th>Genome</th><th>Fitness</th><th>Selection</th><th>Crossover</th><th>Mutation</th><th>Elitism</th></tr>
  ${list(TABLE, ([r, ...c]) => `<tr><td><a href="#/${r}">${esc(c[0])}</a></td>${list(c.slice(1), x => `<td>${esc(x)}</td>`)}</tr>`)}</table></div>
  <p class="small mute">Read from the scripts' code; "p" is a probability. Two more demos in the collection,
  <a href="#/flow">max-flow</a> and <a href="#/rd">reaction-diffusion</a>, are classic algorithms with no genetic search.</p>
  <h2>The shared operators in <code>ga_core.py</code></h2>
  <p>The repository also contains <code>ga_core.py</code>, the common operators extracted into one module (with smoke
  tests in <code>tests/test_ga_core.py</code>). The ten scripts still carry their own copies of these operators, which
  are what the pages on this site port.</p>
  <div class="ops">${list(CORE, ([f, d]) => `<div class="card"><b><code>${f}</code></b><span>${esc(d)}</span></div>`)}</div>`;
});
