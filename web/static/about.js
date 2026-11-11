// About page: what the project is, how to run the desktop scripts and the web app, what was ported, credits.
import { route, app, api, esc, list, isStatic } from './core.js';

route('about', async () => {
  const cat = await api('/problems');
  app().innerHTML = `
  <section class="hero"><p class="kicker">About</p><h1 class="glow">Creative Genetic Algorithm Visualizers</h1>
    <p class="lead">Ten standalone Python scripts, each visualising an algorithm at work on a different optimisation
    problem with matplotlib animations: eight genetic algorithms plus a max-flow solver and a reaction-diffusion simulator.
    This site is their web edition.</p></section>
  <h2>How this site works</h2>
  <p>Each solver's code (fitness function, GA operators, problem generator) was ported to JavaScript, keeping the
  Python's order of operations. Random numbers come from a small seeded generator shared by both languages: the
  repository's tests swap it into the real Python scripts, run them, and check that the JavaScript produces the same
  routes, layouts, fitness histories and networks (to 1 part in 10<sup>9</sup>). The portfolio page uses the market data
  exactly as the Python generates it (NumPy seed 42). ${isStatic() ? 'This copy is the static GitHub Pages site: everything runs in your browser.' : 'You are running the local app (FastAPI server).'}</p>
  <h2>Run the desktop scripts</h2>
  <pre>pip install -r requirements.txt
python travelling-sales-person-solver.py      # or any script below; most ask for parameters first</pre>
  <div class="scroll"><table><tr><th>Script</th><th>Web page</th><th>Desktop defaults</th></tr>
  ${list(cat, c => `<tr><td><code>${esc(c.script)}</code></td><td><a href="#/${c.route}">${esc(c.title)}</a></td>
    <td class="small mono">${esc(Object.entries(c.defaults).map(([k, v]) => `${k}=${v}`).join(', '))}</td></tr>`)}</table></div>
  <p class="small mute">The underscore modules (<code>flow_network.py</code> …) import the same code, for use from Python.</p>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py                 # http://127.0.0.1:8154
python scripts/build_static.py    # the static site in site/</pre>
  <h2>Credits</h2>
  <p>By George Dickinson (<a href="https://github.com/gddickinson/creative-genetic-problem-solver">gddickinson/creative-genetic-problem-solver</a>),
  MIT licence. Built with AI assistance from Claude (Anthropic). Uses numpy, matplotlib, networkx, pandas and seaborn
  on the desktop; the web edition has no external dependencies in the browser.</p>`;
});
