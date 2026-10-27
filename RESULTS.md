# Example run results

Seed 20261027; 1,000 applicants in each synthetic training/test pool; 1,500 random Game 2 candidates plus baselines; 1,000 training groups and 10,000 independent test groups per scenario. Game 1 is evaluated against every test applicant. These are invented opponent models, not actual submissions.

| Pool | Game | Original allocation | Original mean | Training-selected allocation | Test mean |
|---|---|---|---:|---|---:|
| balanced | game1 | [1, 0, 27, 32, 40] | 11.994 | [8, 0, 24, 30, 38] | 12.305 |
| balanced | game2 | [0, 0, 46, 0, 54] | 9.000 | [1, 0, 27, 32, 40] | 9.000 |
| diverse | game1 | [1, 0, 27, 32, 40] | 9.306 | [5, 11, 20, 26, 38] | 9.676 |
| diverse | game2 | [0, 0, 46, 0, 54] | 6.427 | [0, 0, 0, 36, 64] | 6.627 |
| concentrated | game1 | [1, 0, 27, 32, 40] | 8.612 | [1, 2, 25, 2, 70] | 9.839 |
| concentrated | game2 | [0, 0, 46, 0, 54] | 5.103 | [0, 2, 6, 1, 91] | 6.855 |

Game 1 scores are castle points out of 15. Game 2 scores are ranking payouts out of 9. Selected strategies differ by opponent model. They are chosen on training data, not this test data. Game 1 optimization is exact for its training pool; Game 2 is a finite candidate search, not a proof of optimality. Monte Carlo intervals in summary.csv cover group sampling only, not model uncertainty.

The concentrated-pool result challenges the earlier two-castle advice: dominating castle 5 plus cheaply contesting other castles can outperform the original pair allocation in this synthetic population. It is not a universal recommendation.
