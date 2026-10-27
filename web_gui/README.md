# Web GUI — Castle Research

Open `index.html` in a modern browser. Keep `app.js`, `engine.js`, and `style.css` in this same folder. No installation, build, Python backend, or internet connection is required for the GUI.

If your browser restricts local-file behavior, run this from the extracted `castle_simulator` directory:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory web_gui
```

Then open http://127.0.0.1:8000 . Stop the server with Ctrl+C.

## Controls

- Choose head-to-head or four-way scoring.
- Change castle allocations; they must be whole numbers totaling 100. Rebalance normalizes the current weights.
- Select a synthetic opponent model or import a CSV with five bids per applicant (optional name column/header).
- Run Experiment compares your allocation with five baseline strategies.
- Search Better Allocation selects on training data, then evaluates on separate test data. It is not guaranteed to improve held-out results.
- Pause/Play controls battle replay and the ASCII-inspired visualization.
- Export Results downloads the current experiment as JSON.

All computation and imported-data processing occur locally in your browser. The browser uses a JavaScript port of the Python scoring logic, with a different seeded random generator. Results for an identical seed therefore need not match Python. Game 1 uses exact best-response dynamic programming. Game 2 searches roughly 500 candidates over 300 shared training groups, then evaluates with the selected test-group count. The Python CLI uses its own configurable search sizes.

The included defaults and synthetic populations are examples, not actual applicant data. Four-way confidence intervals describe Monte Carlo variation, not uncertainty in the opponent model. Reloading resets the session; export results before closing if needed.

## Optional engine tests

With Node.js installed, run from this folder:

```sh
node test-engine.cjs
```
