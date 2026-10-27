#!/usr/bin/env python3
"""Monte Carlo simulator for five-castle allocation games."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

VALUES = np.arange(1, 6, dtype=np.int16)
BASELINES = {
    "equal20": (20, 20, 20, 20, 20),
    "proportional": (7, 13, 20, 27, 33),
    "prior_game1": (1, 0, 27, 32, 40),
    "prior_game2": (0, 0, 46, 0, 54),
    "high_pair": (0, 0, 0, 46, 54),
}
PAYOUTS = (9, 4, 2, 0)
TOTAL = 100


def validate_allocation(allocation: Iterable[int]) -> tuple[int, ...]:
    raw = tuple(allocation)
    try:
        numeric = tuple(float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("allocation entries must be integers") from exc
    if any(not math.isfinite(x) or not x.is_integer() for x in numeric):
        raise ValueError("allocation entries must be integers")
    a = tuple(int(x) for x in numeric)
    if len(a) != 5 or min(a) < 0 or sum(a) != TOTAL:
        raise ValueError("an allocation must contain five nonnegative integers summing to 100")
    return a


def game1_score_units(candidate: Iterable[int], opponent: Iterable[int]) -> int:
    """Game 1 score in twelfths; castle ties award half their point value."""
    a, b = validate_allocation(candidate), validate_allocation(opponent)
    return sum(int(v) * (12 if x > y else 6 if x == y else 0) for v, x, y in zip(VALUES, a, b))


def game2_castle_points_units(allocations: np.ndarray) -> np.ndarray:
    """Castle-point totals for four players, in twelfths."""
    raw = np.asarray(allocations)
    if raw.shape != (4, 5):
        raise ValueError("game 2 needs a 4x5 matrix of valid allocations")
    a = np.asarray([validate_allocation(row) for row in raw], dtype=np.int16)
    points = np.zeros(4, dtype=np.int16)
    for j, value in enumerate(VALUES):
        winners = a[:, j] == a[:, j].max()
        points[winners] += int(value) * (12 // int(winners.sum()))
    return points


def game2_payout_units(points: Iterable[int]) -> np.ndarray:
    """Return payouts in quarter-point units, splitting occupied rank slots."""
    pts = np.asarray(tuple(points), dtype=np.int32)
    if pts.shape != (4,):
        raise ValueError("exactly four applicant scores are required")
    order = np.argsort(-pts, kind="stable")
    payouts = np.zeros(4, dtype=np.int16)
    pos = 0
    while pos < 4:
        end = pos + 1
        while end < 4 and pts[order[end]] == pts[order[pos]]:
            end += 1
        # Quarter-point units make every tie split exact (including 15/4).
        units = sum(PAYOUTS[pos:end]) * 4 // (end - pos)
        payouts[order[pos:end]] = units
        pos = end
    return payouts


def game2_score_units(allocations: np.ndarray) -> np.ndarray:
    return game2_payout_units(game2_castle_points_units(allocations))


def allocation_from_weights(weights: np.ndarray) -> tuple[int, ...]:
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(5)
    raw = w / w.sum() * TOTAL
    result = np.floor(raw).astype(int)
    remainder = TOTAL - int(result.sum())
    if remainder:
        result[np.argsort(-(raw - result), kind="stable")[:remainder]] += 1
    return tuple(int(x) for x in result)


def make_pool(rng: np.random.Generator, size: int, scenario: str) -> np.ndarray:
    rows = []
    for i in range(size):
        if scenario == "balanced":
            target = np.asarray(BASELINES["equal20"] if i % 2 == 0 else BASELINES["proportional"])
            jitter = rng.integers(-3, 4, 5) if i % 4 >= 2 else np.zeros(5, dtype=int)
            rows.append(allocation_from_weights(np.maximum(1, target + jitter)))
        elif scenario == "diverse":
            mode = i % 5
            if mode == 0:
                alpha = np.full(5, 0.8)
            elif mode == 1:
                alpha = rng.uniform(0.15, 1.2, 5) * np.arange(1, 6)
            elif mode == 2:
                alpha = np.full(5, rng.choice([0.35, 0.65, 1.0]))
            elif mode == 3:
                alpha = np.full(5, 0.35)
                alpha[rng.choice(5, 2, replace=False)] = 3.0
            else:
                weights = rng.dirichlet(np.full(5, 0.55))
                rows.append(allocation_from_weights(np.round(weights * 20) / 20))
                continue
            rows.append(allocation_from_weights(rng.dirichlet(alpha)))
        elif scenario == "concentrated":
            alpha = np.full(5, 0.035)
            chosen = rng.choice(5, int(rng.choice([2, 3])), replace=False)
            alpha[chosen] = rng.uniform(2.5, 5.5, len(chosen)) * np.linspace(0.9, 1.8, 5)[chosen]
            rows.append(allocation_from_weights(rng.dirichlet(alpha)))
        else:
            raise ValueError(f"unknown scenario {scenario}")
    return np.asarray(rows, dtype=np.int16)


def load_opponents(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.reader(f), 1):
            if not row or all(not x.strip() for x in row):
                continue
            if len(row) == 6:
                if line_no == 1 and any("applicant" in x.lower() for x in row):
                    continue
                row = row[1:]  # Optional applicant label followed by five castle counts.
            if len(row) != 5:
                if line_no == 1 and any("castle" in x.lower() for x in row):
                    continue
                raise ValueError(f"{path}:{line_no}: expected five allocation columns")
            try:
                rows.append(validate_allocation(int(x.strip()) for x in row))
            except ValueError as exc:
                if line_no == 1 and any(not x.strip().lstrip("+-").isdigit() for x in row):
                    continue
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if len(rows) < 3:
        raise ValueError("opponents CSV must contain at least three applicant rows")
    return np.asarray(rows, dtype=np.int16)


def best_response_game1(pool: np.ndarray) -> tuple[int, ...]:
    """Exact DP best response against the fixed pool's per-opponent distribution."""
    pool = np.asarray(pool, dtype=np.int16)
    n = len(pool)
    neg = -10**12
    dp = np.full(TOTAL + 1, neg, dtype=np.int64)
    dp[0] = 0
    choices = []
    for j, value in enumerate(VALUES):
        bids = np.arange(TOTAL + 1)[:, None]
        opp = pool[None, :, j]
        sums = ((bids > opp).sum(axis=1) * 12 + (bids == opp).sum(axis=1) * 6) * int(value)
        # Keep the numerator over the pool size throughout the DP. This avoids
        # rounding each castle's marginal payoff before adding them.
        payoff = sums
        new = np.full_like(dp, neg)
        chosen = np.full(TOTAL + 1, -1, dtype=np.int16)
        for used in range(TOTAL + 1):
            if dp[used] == neg:
                continue
            for bid in range(TOTAL - used + 1):
                total = used + bid
                score = dp[used] + int(payoff[bid])
                if score > new[total]:
                    new[total], chosen[total] = score, bid
        choices.append(chosen)
        dp = new
    allocation = [0] * 5
    total = TOTAL
    for j in range(4, -1, -1):
        allocation[j] = int(choices[j][total])
        total -= allocation[j]
    return validate_allocation(allocation)


def sample_groups(rng: np.random.Generator, pool_size: int, groups: int) -> np.ndarray:
    if pool_size < 3:
        raise ValueError("pool size must be at least 3")
    return np.asarray([rng.choice(pool_size, 3, replace=False) for _ in range(groups)], dtype=np.int32)


def evaluate_game1(candidate: tuple[int, ...], pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
    a = np.asarray(candidate, dtype=np.int16)
    opponents = pool[indices]
    wins, ties = a[None, None, :] > opponents, a[None, None, :] == opponents
    scores = ((wins * 12 + ties * 6) * VALUES[None, None, :]).sum(axis=2)
    return scores.mean(axis=1) / 12.0


def evaluate_game2_batch(candidates: list[tuple[int, ...]], pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vectorized held-out Game 2 scores, shape candidates x groups, in PTS."""
    cand = np.asarray(candidates, dtype=np.int16)
    opp = pool[indices]
    alloc = np.concatenate((np.broadcast_to(cand[:, None, None, :], (len(cand), len(indices), 1, 5)),
                            np.broadcast_to(opp[None, :, :, :], (len(cand), len(indices), 3, 5))), axis=2)
    winners = alloc == alloc.max(axis=2, keepdims=True)
    counts = winners.sum(axis=2)
    shares = VALUES[None, None, None, :] * (12 // counts)[:, :, None, :]
    castle_points = (winners * shares).sum(axis=3)
    higher = (castle_points[:, :, None, :] > castle_points[:, :, :, None]).sum(axis=3)
    tied = (castle_points[:, :, None, :] == castle_points[:, :, :, None]).sum(axis=3)
    # Prefix sums for [9, 4, 2, 0] payouts, expressed in quarter-point units.
    prefix = np.array([0, 36, 52, 60, 60], dtype=np.int16)
    payout_units = (prefix[higher + tied] - prefix[higher]) // tied
    return payout_units[:, :, 0] / 4.0


def evaluate_game2(candidate: tuple[int, ...], pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return evaluate_game2_batch([candidate], pool, indices)[0]


def evaluate_game1_exact(candidate: tuple[int, ...], pool: np.ndarray) -> dict:
    a = np.asarray(candidate, dtype=np.int16)
    wins, ties = a[None, :] > pool, a[None, :] == pool
    scores = ((wins * 12 + ties * 6) * VALUES[None, :]).sum(axis=1) / 12.0
    mean = float(scores.mean())
    return {"mean": mean, "se": 0.0, "ci95_low": mean, "ci95_high": mean,
            "groups": len(pool), "method": "exact_average_over_test_pool"}


def random_candidates(rng: np.random.Generator, count: int, scenario: str) -> list[tuple[int, ...]]:
    rows = list(BASELINES.values())
    for i in range(count):
        if scenario == "balanced" and i % 3 == 0:
            alpha = np.full(5, rng.uniform(1.5, 6.0))
        elif scenario == "concentrated" or i % 4 == 0:
            alpha = np.full(5, 0.05)
            chosen = rng.choice(5, int(rng.choice([2, 3])), replace=False)
            alpha[chosen] = rng.uniform(1.5, 6.0, len(chosen)) * np.linspace(0.8, 1.9, 5)[chosen]
        else:
            alpha = rng.uniform(0.2, 1.6, 5)
        rows.append(allocation_from_weights(rng.dirichlet(alpha)))
    return rows


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "se": se, "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se, "groups": len(values)}


def run(args: argparse.Namespace) -> dict:
    scenarios = (["empirical"] if args.opponents else
                 ["balanced", "diverse", "concentrated"] if args.scenario == "all" else [args.scenario])
    seeds = np.random.SeedSequence(args.seed).spawn(len(scenarios))
    report = {"seed": args.seed, "groups": args.groups, "pool_size": args.pool_size,
              "candidates": args.candidates,
              "synthetic_note": "Synthetic applicant pools do not establish field behavior or global optimality.",
              "scenarios": {}}
    rows = []
    for scenario, seed in zip(scenarios, seeds):
        rng = np.random.default_rng(seed)
        train_pool = load_opponents(args.opponents) if args.opponents else make_pool(rng, args.pool_size, scenario)
        test_pool = load_opponents(args.opponents) if args.opponents else make_pool(rng, args.pool_size, scenario)
        train_indices = sample_groups(rng, len(train_pool), min(1000, args.groups))
        test_indices = sample_groups(rng, len(test_pool), args.groups)
        candidates = random_candidates(rng, args.candidates, scenario)
        named: dict[tuple[int, ...], list[str]] = {}
        def add(label, alloc): named.setdefault(alloc, []).append(label)
        for label, alloc in BASELINES.items(): add(label, alloc)
        if args.allocation:
            custom = validate_allocation(int(x.strip()) for x in args.allocation.split(","))
            add("custom", custom)
        br = best_response_game1(train_pool)
        add("game1_best_response", br)
        train_scores = np.empty(len(candidates), dtype=float)
        for start in range(0, len(candidates), 48):
            end = min(start + 48, len(candidates))
            train_scores[start:end] = evaluate_game2_batch(candidates[start:end], train_pool, train_indices).mean(axis=1)
        best = int(train_scores.argmax())
        add("game2_search_best", candidates[best])
        for rank, idx in enumerate(np.argsort(train_scores)[-min(5, len(candidates)):][::-1], 1):
            add(f"game2_training_top{rank}", candidates[int(idx)])
        result = {"game1": {}, "game2": {}, "training_selected_game2": list(candidates[best])}
        for alloc, labels in named.items():
            name = "+".join(labels)
            g1 = evaluate_game1_exact(alloc, test_pool)
            g2 = {**summarize(evaluate_game2(alloc, test_pool, test_indices)),
                  "method": "held_out_monte_carlo"}
            result["game1"][name] = {"allocation": list(alloc), **g1}
            result["game2"][name] = {"allocation": list(alloc), **g2, "method": "held_out_monte_carlo"}
            for game, stat in (("game1", g1), ("game2", g2)):
                rows.append({"scenario": scenario, "candidate": name, "allocation": ",".join(map(str, alloc)),
                             "game": game, **stat,
                             "training_game2_mean": float(train_scores[best]) if "game2_search_best" in labels else ""})
        report["scenarios"][scenario] = result
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scenario", "candidate", "allocation", "game", "mean", "se", "ci95_low", "ci95_high", "groups", "method", "training_game2_mean"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out / 'summary.csv'} and {args.out / 'summary.json'}")
    for scenario, result in report["scenarios"].items():
        print(f"\n{scenario}")
        for game in ("game1", "game2"):
            print(f"  {game}")
            for name, stat in result[game].items():
                print(f"    {name}: {stat['mean']:.3f} PTS (SE {stat['se']:.3f}, 95% CI {stat['ci95_low']:.3f} to {stat['ci95_high']:.3f})")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--groups", type=int, default=10000)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--opponents", type=Path)
    parser.add_argument("--allocation", help="custom allocation such as 1,0,27,32,40")
    parser.add_argument("--scenario", choices=["all", "balanced", "diverse", "concentrated"], default="all")
    args = parser.parse_args(argv)
    if args.groups <= 0 or args.pool_size < 3 or args.candidates <= 0:
        parser.error("groups and candidates must be positive; pool-size must be at least 3")
    return args


if __name__ == "__main__":
    run(parse_args())
