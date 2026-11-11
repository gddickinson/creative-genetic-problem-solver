// Entry point: registers the lazily loaded pages, sets the footer mode text and starts the router.
import { start, route, lazy, app, isStatic, $ } from './core.js';

const PAGES = {
  'home.js': ['home'], 'howga.js': ['ga'], 'about.js': ['about'],
  'tsp.js': ['tsp'], 'creative.js': ['creative'], 'vrp.js': ['vrp'], 'packing.js': ['packing'], 'portfolio.js': ['portfolio'],
  'neural.js': ['neural'], 'game.js': ['game'], 'maze.js': ['maze'], 'flow.js': ['flow'], 'rd.js': ['rd'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: every solver runs in your browser (JS ports of the Python code)'
  : 'local app: Python server on this machine';
start();
