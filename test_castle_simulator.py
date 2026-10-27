import unittest
import numpy as np

from castle_simulator import (
    best_response_game1,
    game1_score_units,
    game2_castle_points_units,
    game2_payout_units,
    game2_score_units,
    sample_groups,
    evaluate_game2,
    validate_allocation,
)


class ScoringTests(unittest.TestCase):
    def test_game1_win_tie_loss_exact_twelfths(self):
        self.assertEqual(
            game1_score_units((20, 10, 30, 20, 20), (19, 10, 40, 20, 11)),
            12 + 12 + 0 + 24 + 60,
        )

    def test_castle_points_total_fifteen(self):
        allocations = np.array([
            [40, 0, 0, 0, 60], [40, 0, 0, 0, 60],
            [10, 50, 20, 10, 10], [10, 50, 20, 10, 10],
        ])
        self.assertEqual(int(game2_castle_points_units(allocations).sum()), 15 * 12)

    def test_three_way_castle_tie_shares_exactly(self):
        allocations = np.array([
            [30, 70, 0, 0, 0], [30, 0, 70, 0, 0],
            [30, 0, 0, 70, 0], [20, 20, 20, 20, 20],
        ])
        points = game2_castle_points_units(allocations)
        self.assertEqual(int(points.sum()), 15 * 12)
        self.assertEqual(points[0] % 12, 4)  # One point divided among three winners.

    def test_all_identical_allocations_split_all_castles(self):
        allocations = np.tile([20, 20, 20, 20, 20], (4, 1))
        self.assertEqual(game2_castle_points_units(allocations).tolist(), [45] * 4)
        self.assertEqual(game2_score_units(allocations).tolist(), [15] * 4)
        self.assertAlmostEqual(game2_score_units(allocations)[0] / 4, 3.75)

    def test_rank_ties_split_occupied_payouts(self):
        self.assertEqual(game2_payout_units([5, 4, 3, 3]).tolist(), [36, 16, 4, 4])
        self.assertEqual(game2_payout_units([100, 100, 50, 50]).tolist(), [26, 26, 4, 4])
        self.assertEqual(game2_payout_units([5, 5, 5, 0]).tolist(), [20, 20, 20, 0])

    def test_game2_payout_total_is_fifteen_pts(self):
        rng = np.random.default_rng(52)
        for _ in range(100):
            points = rng.integers(0, 181, 4)
            self.assertEqual(int(game2_payout_units(points).sum()), 60)

    def test_groups_have_distinct_applicant_indices(self):
        groups = sample_groups(np.random.default_rng(7), 10, 100)
        self.assertEqual(groups.shape, (100, 3))
        self.assertTrue(all(len(set(row)) == 3 for row in groups))

    def test_best_response_is_valid(self):
        pool = np.array([[20, 20, 20, 20, 20], [0, 0, 0, 0, 100], [10, 20, 30, 20, 20]])
        response = best_response_game1(pool)
        self.assertEqual(sum(response), 100)
        self.assertTrue(all(x >= 0 for x in response))

    def test_equal_twenty_pool_best_response_gets_fourteen(self):
        pool = np.tile([20, 20, 20, 20, 20], (4, 1))
        response = best_response_game1(pool)
        self.assertEqual(game1_score_units(response, pool[0]) / 12, 14)

    def test_vectorized_game2_matches_exact_scalar_scoring(self):
        rng = np.random.default_rng(91)
        candidates = [(20, 20, 20, 20, 20), (0, 0, 46, 0, 54), (1, 0, 27, 32, 40)]
        pool = np.array([
            [20, 20, 20, 20, 20], [20, 20, 20, 20, 20],
            [40, 0, 0, 0, 60], [10, 50, 20, 10, 10],
            *[tuple(int(x) for x in rng.multinomial(100, rng.dirichlet(np.ones(5)))) for _ in range(20)],
        ])
        indices = np.vstack((np.array([[0, 1, 2], [0, 3, 2]]), sample_groups(rng, len(pool), 25)))
        for candidate in candidates:
            actual = evaluate_game2(candidate, pool, indices)
            expected = np.array([
                game2_score_units(np.vstack((candidate, pool[group])))[0] / 4.0
                for group in indices
            ])
            np.testing.assert_array_equal(actual, expected)

    def test_non_integer_allocation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_allocation([20, 20, 20.5, 20, 19.5])
        for invalid in (
            np.array([[20.5, 20, 20, 20, 19.5]] * 4),
            np.array([[-1, 21, 20, 20, 40]] * 4),
            np.array([[20, 20, 20, 20, 19]] * 4),
        ):
            with self.assertRaises(ValueError):
                game2_castle_points_units(invalid)


if __name__ == "__main__":
    unittest.main()
