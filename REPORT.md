# Bikeshare demand: report

## Results

_To be written._

## Headline finding

_To be written._

## Exercise 4: target transformation

**The problem.** Gradient boosting (hgb) scores validation R^2 0.2415 and test
-0.4600, against ridge's 0.6839 and 0.6239. A tree predicts by averaging
training targets in a leaf, so it cannot predict above the largest count it saw
in 2011 (6,043). Ridership grew, and 57% of test days are above that ceiling.
Untransformed hgb's highest test prediction is 5,671: it predicts above the
ceiling on 0% of days.

**The transformation.** Predict `cnt / cnt_roll_7`, today's rides as a multiple
of the mean of the seven days before, then multiply the prediction back by
`cnt_roll_7`. The tree now learns a bounded multiplier ("a warm Saturday is
about 1.2x the past week"), and the growth comes from the lagged level, which
keeps rising. It runs as `python -m src.train --target ratio7`. `make train` and
`reports/metrics.json` remain the untransformed pipeline.

**How it was chosen.** Three candidates were compared on validation only
(`--valid-only` never scores the test split; `tests/test_target.py` checks this):

| target | ridge valid R^2 | ridge valid MAE | hgb valid R^2 | hgb valid MAE |
|---|---|---|---|---|
| none | 0.6839 | 777.5 | 0.2415 | 1296.6 |
| log (control) | 0.3745 | 983.0 | 0.1805 | 1328.1 |
| ratio7 | 0.4380 | 1042.4 | 0.7329 | 711.1 |
| diff7 (`cnt - cnt_lag_7`) | 0.6839 | 776.6 | 0.4811 | 981.3 |

`log` was a control. It compresses the scale but keeps the ceiling, and it lost,
as expected. `ratio7` was recorded as the choice in commit `f71a1c2` before any
transformed test score existed. Then the test split was scored once.

**Test, scored once:**

| model | valid R^2 | valid MAE | test R^2 | test MAE |
|---|---|---|---|---|
| baseline_mean | -0.6894 | 1877.3 | -1.7717 | 2659.6 |
| baseline_persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | 0.6839 | 777.5 | 0.6239 | 879.3 |
| hgb | 0.2415 | 1296.6 | -0.4600 | 1952.4 |
| ridge + ratio7 | 0.4380 | 1042.4 | 0.0890 | 1437.9 |
| hgb + ratio7 | 0.7329 | 711.1 | 0.6165 | 839.7 |

With `ratio7`, 42% of hgb's test predictions are above the 2011 ceiling (max
8,930), so the mechanism did what it was meant to.

**Is it defensible?** The mechanism is. The level it multiplies back by is a
lagged feature known before the day starts. The round trip is exact, and
`tests/test_target.py` checks that the inverse reads only features, never the
true count. The choice was fixed on validation before the test split was scored.
It fixes the specific failure the headline is about.

**Is it a win?** No, and the report should not present it as one:

- **It ties ridge rather than beating it.** Test R^2 is 0.6165 vs 0.6239, and
  that 0.007 gap is far inside the noise of 184 test days. Its validation lead
  (0.7329 vs 0.6839) did not survive to the test set.
- **On MAE it only matches persistence.** 839.7 vs 840.1 is a tie with
  "yesterday again".
- **It suits trees, not the comparison.** It was picked from three candidates
  to help the tree, and the same transform sinks ridge (test R^2 0.0890). A
  transform that helps one model and wrecks the other is a modelling choice
  for that model, not a better pipeline.

**Conclusion.** Under a growth trend, hgb's failure was about scale, not model
class: once the target is relative to recent demand, the tree is as good as
ridge. Neither beats persistence on MAE by a meaningful margin, so the honest
claim about this data stays modest. A lag-driven model is about 0.1 R^2 better
than "yesterday again" and no better in typical absolute error.

The test split had been seen once before this run: `agent/audit.py` quoted
untransformed test scores in Exercise 3, after all feature work was committed.
Nothing in this exercise was tuned on it.

## Exercise 5: ridge vs persistence

_To be written._

## Attribution

_To be written. Must open with the model used (Opus, switched from the local
qwen build) and credit the ratio/difference idea's exposure via the Exercise 3
audit of jecDSC/bikeshare-open, which used ratio28 and diff1._
