/* Small deterministic simulation engine for the five-castle allocation game. */
(function (root) {
  'use strict';

  const VALUES = [1, 2, 3, 4, 5];
  const BASELINES = [
    [20, 20, 20, 20, 20], [7, 13, 20, 27, 33], [1, 0, 27, 32, 40],
    [0, 0, 46, 0, 54], [0, 0, 0, 46, 54]
  ];
  const PAYOUTS = [9, 4, 2, 0];

  function rng(seed) {
    // Mulberry32: compact, repeatable, and suitable for simulation UX.
    let state = Number(seed) >>> 0;
    const random = function () {
      state = (state + 0x6D2B79F5) | 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    random.int = (lo, hi) => lo + Math.floor(random() * (hi - lo));
    random.choice = (items) => items[Math.floor(random() * items.length)];
    random.shuffle = (items) => {
      const a = items.slice();
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };
    return random;
  }

  function validateAllocation(a) {
    if (!Array.isArray(a) || a.length !== 5 || a.some(x => !Number.isFinite(x) || !Number.isInteger(x) || x < 0) || a.reduce((x, y) => x + y, 0) !== 100) {
      throw new Error('allocation must contain five nonnegative integers summing to 100');
    }
    return a;
  }

  function gamma(random, shape) {
    // Marsaglia-Tsang gamma sampler (with the usual shape<1 transform).
    if (shape < 1) return gamma(random, shape + 1) * Math.pow(Math.max(random(), 1e-14), 1 / shape);
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { const u1 = Math.max(random(), 1e-14), u2 = random(); x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = random();
      if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function dirichlet(random, alpha) {
    const v = alpha.map(a => gamma(random, a));
    const sum = v.reduce((a, b) => a + b, 0);
    return v.map(x => x / sum);
  }
  function fromWeights(weights) {
    let sum = weights.reduce((a, b) => a + b, 0);
    if (!(sum > 0)) { weights = [1, 1, 1, 1, 1]; sum = 5; }
    const raw = weights.map(x => x / sum * 100);
    const out = raw.map(Math.floor);
    let remainder = 100 - out.reduce((a, b) => a + b, 0);
    const order = raw.map((v, i) => [v - out[i], i]).sort((a, b) => b[0] - a[0]);
    for (let i = 0; i < remainder; i++) out[order[i][1]]++;
    return out;
  }
  function pool(random, n, scenario) {
    if (!Number.isInteger(n) || n < 1) throw new Error('pool size must be a positive integer');
    if (!['balanced', 'diverse', 'concentrated'].includes(scenario)) throw new Error('unknown scenario: ' + scenario);
    const rows = [];
    for (let i = 0; i < n; i++) {
      if (scenario === 'balanced') {
        const base = i % 2 ? [7, 13, 20, 27, 33] : [20, 20, 20, 20, 20];
        const jitter = i % 4 >= 2 ? base.map(() => random.int(-3, 4)) : [0, 0, 0, 0, 0];
        rows.push(fromWeights(base.map((x, j) => Math.max(1, x + jitter[j]))));
      } else if (scenario === 'diverse') {
        const mode = i % 5;
        let alpha;
        if (mode === 0) alpha = [0.8, 0.8, 0.8, 0.8, 0.8];
        else if (mode === 1) alpha = Array.from({ length: 5 }, (_, j) => random() * 1.05 + 0.15).map((x, j) => x * (j + 1));
        else if (mode === 2) { const x = random.choice([0.35, 0.65, 1]); alpha = [x, x, x, x, x]; }
        else if (mode === 3) { alpha = [0.35, 0.35, 0.35, 0.35, 0.35]; const ix = random.shuffle([0, 1, 2, 3, 4]).slice(0, 2); ix.forEach(j => alpha[j] = 3); }
        else { rows.push(fromWeights(dirichlet(random, [0.55, 0.55, 0.55, 0.55, 0.55]).map(x => Math.round(x * 20) / 20))); continue; }
        rows.push(fromWeights(dirichlet(random, alpha)));
      } else {
        const alpha = [0.035, 0.035, 0.035, 0.035, 0.035];
        const chosen = random.shuffle([0, 1, 2, 3, 4]).slice(0, random.choice([2, 3]));
        chosen.forEach((j, k) => alpha[j] = (random() * 3 + 2.5) * [0.9, 1.125, 1.35, 1.575, 1.8][j]);
        rows.push(fromWeights(dirichlet(random, alpha)));
      }
    }
    return rows;
  }

  function rankPayout(points) {
    const order = points.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const payouts = Array(points.length).fill(0);
    for (let pos = 0; pos < order.length;) {
      let end = pos + 1;
      while (end < order.length && order[end][0] === order[pos][0]) end++;
      const slots = PAYOUTS.slice(pos, end).reduce((a, b) => a + b, 0) / (end - pos);
      for (let k = pos; k < end; k++) payouts[order[k][1]] = slots;
      pos = end;
    }
    return payouts;
  }
  function battle(players) {
    if (!Array.isArray(players) || (players.length !== 2 && players.length !== 4)) throw new Error('battle needs two or four players');
    players.forEach(validateAllocation);
    const pointUnits = Array(players.length).fill(0);
    const winners = [];
    for (let j = 0; j < 5; j++) {
      let high = -Infinity;
      for (const p of players) high = Math.max(high, p[j]);
      const ix = [];
      for (let i = 0; i < players.length; i++) if (players[i][j] === high) ix.push(i);
      winners.push(ix);
      // Game payouts are represented in twelfths, so ties (including thirds)
      // remain exact when total scores are compared for rank.
      const shareUnits = 12 / ix.length;
      for (const i of ix) pointUnits[i] += VALUES[j] * shareUnits;
    }
    const points = pointUnits.map(x => x / 12);
    return { points, payouts: players.length === 2 ? points.slice() : rankPayout(pointUnits), winners };
  }

  function bestResponseGame1(oppPool) {
    if (!Array.isArray(oppPool) || !oppPool.length) throw new Error('pool must not be empty');
    oppPool.forEach(validateAllocation);
    let dp = Array(101).fill(-Infinity); dp[0] = 0;
    const choices = [];
    for (let j = 0; j < 5; j++) {
      const pay = Array.from({ length: 101 }, (_, bid) => oppPool.reduce((s, opp) => s + VALUES[j] * (bid > opp[j] ? 1 : bid === opp[j] ? 0.5 : 0), 0) / oppPool.length);
      const next = Array(101).fill(-Infinity), choice = Array(101).fill(-1);
      for (let used = 0; used <= 100; used++) if (Number.isFinite(dp[used])) {
        for (let bid = 0; bid <= 100 - used; bid++) {
          const total = used + bid, score = dp[used] + pay[bid];
          if (score > next[total]) { next[total] = score; choice[total] = bid; }
        }
      }
      choices.push(choice); dp = next;
    }
    const a = Array(5); let total = 100;
    for (let j = 4; j >= 0; j--) { a[j] = choices[j][total]; total -= a[j]; }
    return a;
  }

  function sampleIndices(random, poolSize, count) {
    if (poolSize < count || count < 0) throw new Error('pool size must be at least the sample size');
    const out = [];
    while (out.length < count) {
      const ix = Math.floor(random() * poolSize);
      if (!out.includes(ix)) out.push(ix);
    }
    return out;
  }
  function groupsList(groups, random, poolSize) {
    if (Array.isArray(groups)) return groups;
    const count = Number.isInteger(groups) && groups > 0 ? groups : 1000;
    return Array.from({ length: count }, () => sampleIndices(random, poolSize, 3));
  }
  function summarize(values) {
    const n = values.length, mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? values.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1) : 0;
    const se = Math.sqrt(variance / n);
    return { mean, se, ci: [mean - 1.96 * se, mean + 1.96 * se] };
  }
  function evaluate(allocation, oppPool, game, groups, random) {
    validateAllocation(allocation);
    if (!Array.isArray(oppPool) || oppPool.length < (game === 1 ? 1 : 3)) throw new Error('pool is too small for this game');
    const rnd = typeof random === 'function' ? random : rng(1);
    const selected = game === 1 ? oppPool.map((_, i) => [i]) : groupsList(groups, rnd, oppPool.length);
    if (game !== 1 && game !== 2) throw new Error('game must be 1 or 2');
    const vals = [], first = [], castle = [0, 0, 0, 0, 0], samples = [], trace = [];
    for (let k = 0; k < selected.length; k++) {
      const ids = selected[k];
      if (game === 1) {
        const result = battle([allocation, oppPool[ids[0]]]);
        vals.push(result.points[0]);
        first.push(result.points[0] > result.points[1] ? 1 : result.points[0] === result.points[1] ? 0.5 : 0);
        for (let j = 0; j < 5; j++) if (result.winners[j].includes(0)) castle[j] += 1 / result.winners[j].length;
        if (samples.length < 30) samples.push({ players: [allocation.slice(), oppPool[ids[0]].slice()], points: result.points, payouts: result.payouts, winners: result.winners });
      } else {
        if (!Array.isArray(ids) || ids.length !== 3) throw new Error('each game 2 group must contain three pool indices');
        const players = [allocation, ...ids.map(i => oppPool[i])], result = battle(players);
        vals.push(result.payouts[0]);
        const units = result.points.map(x => Math.round(x * 12));
        const high = Math.max(...units), nTop = units.filter(x => x === high).length;
        first.push(units[0] === high ? 1 / nTop : 0);
        result.winners.forEach((w, j) => { if (w.includes(0)) castle[j] += 1 / w.length; });
        if (samples.length < 30) samples.push({ players: players.map(p => p.slice()), points: result.points, payouts: result.payouts, winners: result.winners });
      }
      if (trace.length < 80 && (selected.length <= 80 || Math.floor((k + 1) * 80 / selected.length) > Math.floor(k * 80 / selected.length))) {
        const partial = vals.slice(0, k + 1);
        trace.push({ n: k + 1, mean: partial.reduce((a, b) => a + b, 0) / partial.length });
      }
    }
    const stat = summarize(vals), count = vals.length;
    if (game === 1) { stat.se = 0; stat.ci = [stat.mean, stat.mean]; }
    return { ...stat, firstRate: first.reduce((a, b) => a + b, 0) / count,
      castleRates: castle.map(x => x / count), trace, samples };
  }

  function candidateRows(random, count, scenario) {
    const rows = BASELINES.map(x => x.slice());
    for (let i = 0; i < count; i++) {
      let alpha;
      if (scenario === 'balanced' && i % 3 === 0) { const x = random() * 4.5 + 1.5; alpha = [x, x, x, x, x]; }
      else if (scenario === 'concentrated' || i % 4 === 0) {
        alpha = [0.05, 0.05, 0.05, 0.05, 0.05];
        const ix = random.shuffle([0, 1, 2, 3, 4]).slice(0, random.choice([2, 3]));
        ix.forEach(j => alpha[j] = (random() * 4.5 + 1.5) * [0.8, 1.075, 1.35, 1.625, 1.9][j]);
      } else alpha = Array.from({ length: 5 }, () => random() * 1.4 + 0.2);
      rows.push(fromWeights(dirichlet(random, alpha)));
    }
    return rows;
  }
  function trainingPayout(candidate, poolRows, ids) {
    const pointUnits = [0, 0, 0, 0];
    for (let j = 0; j < 5; j++) {
      const v = VALUES[j];
      let high = candidate[j];
      for (let k = 0; k < 3; k++) if (poolRows[ids[k]][j] > high) high = poolRows[ids[k]][j];
      let count = candidate[j] === high ? 1 : 0;
      for (let k = 0; k < 3; k++) if (poolRows[ids[k]][j] === high) count++;
      const share = 12 / count;
      if (candidate[j] === high) pointUnits[0] += v * share;
      for (let k = 0; k < 3; k++) if (poolRows[ids[k]][j] === high) pointUnits[k + 1] += v * share;
    }
    let higher = 0, tied = 0;
    for (let i = 1; i < 4; i++) {
      if (pointUnits[i] > pointUnits[0]) higher++;
      if (pointUnits[i] === pointUnits[0]) tied++;
    }
    tied++; // include the candidate
    let payout = 0;
    for (let slot = higher; slot < higher + tied; slot++) payout += PAYOUTS[slot];
    return payout / tied;
  }
  async function optimize(allocation, oppPool, game, groups, random, onProgress) {
    validateAllocation(allocation);
    const rnd = typeof random === 'function' ? random : rng(1);
    const scenario = typeof groups === 'string' ? groups : 'diverse';
    if (game === 1) {
      const best = bestResponseGame1(oppPool);
      return { allocation: best, trainingMean: evaluate(best, oppPool, 1, groups, rnd).mean };
    }
    if (game !== 2) throw new Error('game must be 1 or 2');
    // Train on independent random groups from the pool, shared by every candidate.
    const trainGroups = Array.from({ length: 300 }, () => sampleIndices(rnd, oppPool.length, 3));
    const candidates = candidateRows(rnd, 495, scenario);
    let best = 0, bestAlloc = allocation.slice();
    for (let g = 0; g < trainGroups.length; g++) best += trainingPayout(bestAlloc, oppPool, trainGroups[g]);
    best /= trainGroups.length;
    for (let start = 0; start < candidates.length; start += 10) {
      const end = Math.min(start + 10, candidates.length);
      for (let i = start; i < end; i++) {
        let score = 0;
        for (let g = 0; g < trainGroups.length; g++) score += trainingPayout(candidates[i], oppPool, trainGroups[g]);
        score /= trainGroups.length;
        if (score > best) { best = score; bestAlloc = candidates[i]; }
      }
      if (onProgress) onProgress({ done: end, total: candidates.length, best, allocation: bestAlloc.slice() });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return { allocation: bestAlloc.slice(), trainingMean: best };
  }

  const api = { rng, pool, battle, evaluate, optimize, validateAllocation, bestResponseGame1 };
  root.CastleEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
