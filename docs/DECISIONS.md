# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-25: Hook `check_leakage.sh` now runs `.venv/bin/python` when it exists. Hooks do not inherit the activated venv, so on a machine without a bare `python` every src edit was reported as a failure (`python: command not found`), which makes a red hook meaningless.
- 2026-11-25: Added `cnt_roll_28`, the mean of `cnt` over the 28 calendar days before d. Available at prediction time: every input day ends before d starts. Validation R^2 ridge 0.6475 -> 0.6839, hgb 0.2185 -> 0.2415 (persistence 0.4507 unchanged). LAG_WARMUP 7 -> 28 shrinks train 348 -> 306 rows, which alone moves the mean baseline -0.93 -> -0.69, so the feature effect was isolated on the same 306 rows: ridge +0.038, hgb +0.049. Kept. Test set not consulted.
- 2026-11-25: `cnt_lag_7` and `cnt_roll_7` now use a literal 7 instead of LAG_WARMUP. They were tied to it, so raising LAG_WARMUP for a 28-day feature would have silently turned `cnt_lag_7` into a 28-day lag with no test failing.
- 2026-11-25: Added `test_weather_encoder_rejects_unknown_labels`. Seeded fault 3 (the encoder's `raise` removed) left all 16 tests and the hook green, because the real data never hits the guard; a relabelled BAD would drop 18 days silently. A guard no test exercises is not a check.
