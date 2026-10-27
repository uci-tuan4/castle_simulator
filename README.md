# Five Castle Allocation Simulator

Python 3 and NumPy simulator for integer allocations of 100 soldiers across five castles worth 1, 2, 3, 4, and 5 points.

## Run

```sh
python -m pip install -r requirements.txt
python castle_simulator.py --seed 20261027 --groups 10000 --pool-size 1000 --candidates 1500 --out results
```

The default run evaluates balanced, diverse, and concentrated synthetic pools, then writes `summary.csv` and `summary.json` and prints a console summary. The defaults are designed for a quick run under 45 seconds on a typical laptop; lower `--groups` or `--candidates` for a quicker exploratory run.

Options:

- `--seed N` sets the reproducible random seed.
- `--groups N` sets held-out Monte Carlo groups per scenario.
- `--pool-size N` sets synthetic applicants per pool; at least 3 are required.
- `--candidates N` sets the random candidates searched for Game 2.
- `--out PATH` sets the output directory.
- `--scenario all|balanced|diverse|concentrated` runs one or all pool scenarios.
- `--allocation 1,0,27,32,40` adds a custom allocation to the report.
- `--opponents PATH.csv` loads a supplied population instead of synthetic pools. Each row has five nonnegative integer allocations summing to 100; an optional first applicant label is accepted. Repeated rows are allowed. Applicants are sampled without replacement by row index within each group.

The named baselines are `equal20`, `proportional`, `prior_game1`, `prior_game2`, and `high_pair`. The balanced pool mixes equal and proportional allocations, with some small jitter. The diverse pool combines symmetric and castle-value-weighted Dirichlet draws, 2-castle concentration, and allocations rounded to 5% increments. The concentrated pool favors a random pair or trio of castles and weights higher-value castles more heavily. Game 1 uses a dynamic program to find the exact best response against the fixed training pool distribution. Its held-out score is the exact average across every row in the test pool, so its conditional SE and interval width are zero. Game 2 searches random allocations using 1,000 common training groups, then tests the selected candidates on independent held-out groups. With a supplied CSV, the fixed empirical population is reused for training and test, with separate sampled groups, and the report uses the single `empirical` scenario. Test outcomes do not select candidates.

Castle score arithmetic uses exact twelfths. Tied castles split their points. Game 2 ranks by total castle points, then splits occupied payout slots; results are reported in PTS for payouts `[9, 4, 2, 0]`. Means, standard errors, and 95% Monte Carlo intervals reflect group sampling variation. For a fixed empirical CSV, the intervals do not represent uncertainty about how well the supplied population represents future applicants. JSON and CSV preserve candidate labels and their allocations.

Synthetic scenarios are examples, not observed field data. Random candidate search is finite and can overfit its training groups; reports include the top five training candidates to make that selection visible. No result claims a globally optimal strategy.

## Tests

```sh
python -m unittest -v
```

## Web GUI

The `web_gui/` folder contains the complete terminal-style web interface. Open `web_gui/index.html` in a browser to run it locally without installing dependencies. See `web_gui/README.md` for local-server instructions, controls, and differences between browser and Python simulation settings.
