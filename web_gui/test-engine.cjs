const assert = require('node:assert/strict');
const Engine = require('./engine.js');

assert.equal(globalThis.CastleEngine, Engine);

// A tied high castle splits its value, and rank ties split occupied payout slots.
const fixture = Engine.battle([
  [40, 30, 20, 10, 0], [40, 30, 20, 10, 0],
  [0, 20, 25, 25, 30], [0, 20, 25, 25, 30]
]);
assert.deepEqual(fixture.winners, [[0, 1], [0, 1], [2, 3], [2, 3], [2, 3]]);
assert.deepEqual(fixture.points, [1.5, 1.5, 6, 6]);
assert.deepEqual(fixture.payouts, [1, 1, 6.5, 6.5]);

// Deterministic pools and valid allocation conservation across all scenarios.
for (const scenario of ['balanced', 'diverse', 'concentrated']) {
  const a = Engine.pool(Engine.rng(99), 150, scenario);
  const b = Engine.pool(Engine.rng(99), 150, scenario);
  assert.deepEqual(a, b);
  assert.equal(a.length, 150);
  for (const row of a) {
    assert.equal(row.length, 5);
    assert.ok(row.every(x => Number.isInteger(x) && x >= 0));
    assert.equal(row.reduce((x, y) => x + y, 0), 100);
  }
}

// Randomized battles conserve each castle's value across its winners.
const random = Engine.rng(1234);
const generated = Engine.pool(random, 800, 'diverse');
for (let k = 0; k < 200; k++) {
  const players = [generated[k], generated[k + 200], generated[k + 400], generated[k + 600]];
  const result = Engine.battle(players);
  assert.ok(Math.abs(result.points.reduce((x, y) => x + y, 0) - 15) < 1e-10);
  for (let j = 0; j < 5; j++) {
    const share = result.winners[j].length * (j + 1) / result.winners[j].length;
    assert.equal(share, j + 1);
  }
  assert.ok(Math.abs(result.payouts.reduce((x, y) => x + y, 0) - 15) < 1e-10);
}

const equalPool = Array.from({ length: 20 }, () => [20, 20, 20, 20, 20]);
const optimal = Engine.bestResponseGame1(equalPool);
assert.equal(optimal.reduce((x, y) => x + y, 0), 100);
const score = Engine.evaluate(optimal, equalPool, 1, 1, Engine.rng(1)).mean;
assert.equal(score, 14);
const exact = Engine.evaluate(optimal, equalPool, 1, 999, Engine.rng(2));
assert.equal(exact.se, 0);
assert.deepEqual(exact.ci, [14, 14]);
assert.equal(Engine.bestResponseGame1(equalPool).reduce((x, y) => x + y, 0), 100);

const stat = Engine.evaluate([20, 20, 20, 20, 20], generated, 2, 120, Engine.rng(5));
assert.equal(stat.trace.length, 80);
assert.equal(stat.samples.length, 30);
assert.equal(stat.castleRates.length, 5);
assert.ok(stat.mean >= 0 && stat.mean <= 9);

// Optimization uses the same 300 distinct-opponent groups for the incumbent
// and candidates, so its result cannot be worse on its training sample.
const seed = 778;
const groupRng = Engine.rng(seed);
const trainingGroups = Array.from({ length: 300 }, () => {
  const ids = [];
  while (ids.length < 3) {
    const id = Math.floor(groupRng() * generated.length);
    if (!ids.includes(id)) ids.push(id);
  }
  return ids;
});
(async () => {
  const incumbent = [20, 20, 20, 20, 20];
  const baseline = Engine.evaluate(incumbent, generated, 2, trainingGroups).mean;
  const result = await Engine.optimize(incumbent, generated, 2, 'diverse', Engine.rng(seed));
  assert.ok(result.trainingMean + 1e-12 >= baseline);
  console.log('engine tests passed');
})().catch(err => { console.error(err); process.exitCode = 1; });
