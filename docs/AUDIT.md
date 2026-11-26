# Audit

## Part 1: `/leakage-auditor` on a teammate's repository

**Repository:** https://github.com/jecDSC/bikeshare-open at commit `bf6fa11`,
cloned to `../teammate-bikeshare-open`. Scope: everything since their unmodified
starter commit `756a72b` (`git diff 756a72b HEAD -- src tests Makefile`).

**How it was run:** the `leakage-auditor` subagent, in a fresh context, with
read and run access only. Before it ran, their diff was read for anything unsafe
to execute (network calls, shell-outs, changed hooks); the only network call is
the starter's download, which is skipped when the CSV exists. Their code ran
with this repo's venv (identical pinned `requirements.txt`).

### Auditor output (verbatim)

> All five checks pass, and I found nothing in 756a72b..bf6fa11 that changes a number or invalidates a conclusion. Rerunning the pipeline reproduces the committed reports/metrics.json and both metrics_target_*.json files exactly.
>
> Scope: `git -C /Users/melvin/Documents/DSC180A/teammate-bikeshare-open diff 756a72b HEAD -- src tests Makefile`. Changed files: src/config.py, src/features.py, src/ingest.py, src/train.py, tests/test_contract.py, tests/test_leakage.py. Makefile is unchanged.
>
> **Runs** (cwd /Users/melvin/Documents/DSC180A/teammate-bikeshare-open, Python /Users/melvin/Documents/DSC180A/bikeshare-open/.venv/bin/python)
> - `-m pytest tests -q` gave `23 passed in 1.44s`.
> - `-m src.features` gave `OK features: 672 rows, 15 features -> features.parquet`. The features are temp, atemp, hum, windspeed, dayofweek, is_weekend, is_holiday, is_workday, month_sin, month_cos, weather_severity, cnt_lag_1, cnt_lag_7, cnt_roll_7, cnt_roll_28.
> - `-m src.train`, then `git diff --stat reports/metrics.json`: empty output, so the regenerated metrics.json matches the committed one byte for byte.
> - `-m src.train --target ratio28` and `-m src.train --target diff1 --valid-only`, then `git status --short`: empty output, so both regenerated metrics_target_*.json files also match what was committed. No checkout was needed and the working tree is clean.
>
> **1. Target leakage: pass**
> - `feature_columns(feats) & FORBIDDEN` is `set()`.
> - New config entries don't let anything leak. REDUNDANT_COLS = {season, mnth, weekday} only adds exclusions to FORBIDDEN. FEATURE_SOURCES is a hand-written allowlist that the contract test checks; it never feeds `feature_columns`.
> - Arithmetic restatement check: OLS of cnt on all 15 features, in-sample over all 672 rows, gives R^2 = 0.8119, not about 1.0. No single feature equals cnt on any row.
> - The transforms don't leak either. `_ratio28` / `_unratio28` divide by and multiply back by cnt_roll_28; `_diff1` / `_undiff1` subtract and add back cnt_lag_1. Both levels are shift(1) features. `OnTransformedTarget.predict` (src/train.py:106-108) inverts back to rides before `r2_score` / `mean_absolute_error` run at lines 150-151.
> - Test I ran: I fitted hgb+ratio28 on train, then set y_valid to 0. The predictions did not change, so the inverse uses only X. The prediction range was 1162 to 8489, above the 2011 maximum of 6043, which is the intended effect.
> - The transforms have no fitted statistics; the level is a per-row lagged feature.
>
> **2. Temporal ordering: pass**
> - cnt_roll_28 is `shift(1).rolling(28).mean()` (src/features.py:83); cnt_roll_7 and cnt_lag_7 now use WEEK=7 instead of LAG_WARMUP.
> - I recomputed cnt_lag_1, cnt_lag_7, cnt_roll_7 and cnt_roll_28 for all 672 rows by calendar-day lookup in the raw CSV: 0 mismatches.
> - The split is by date: train ends 2011-12-31, valid runs 2012-01-01 to 2012-06-30, test starts 2012-07-01. The StandardScaler sits inside the ridge pipeline and is fit only on `train[cols]`.
> - Choosing the transform:
>   - At cce8d69, metrics_target_ratio28.json and metrics_target_diff1.json hold validation scores only. The only line matching `test` is `n_test`, and `"test"` occurs 0 times.
>   - The pre-registration commit 73b3fb3 (22:06:58) picks ratio28 for hgb using validation alone: 0.6754 vs 0.6138 for diff1.
>   - The single test run is af90030 (22:09:30).
>   - DECISIONS.md says the test set had already been seen once, through an earlier audit run, after all feature work was committed. Nothing was tuned on it.
>
> **3. Calendar gaps: pass**
> - `build_features` still runs `reindex_to_calendar` before every shift and roll.
> - `test_roll_28_respects_the_calendar_gap` passes: no rows are kept from 2011-03-11 to 2011-04-07.
> - My independent calendar recompute (check 2) had 0 mismatches.
>
> **4. Silent row loss: pass**
> - 728 raw rows become 672, a loss of 56. That is exactly the asserted bound in `test_only_expected_rows_are_dropped`: 28 + 28 x 1 missing day, with LAG_WARMUP raised to 28. An assert in features.py ties FOUR_WEEKS to LAG_WARMUP.
> - `encode_weather` still raises on unknown labels, and a new test covers that.
> - Raw holiday and workday values are exactly {Y, N}, so `== "Y"` drops nothing.
>
> **5. Baselines: pass**
> - `models_for(target)` returns baseline_mean and baseline_persistence unchanged under every transform, and `test_default_pipeline_is_untransformed` checks this.
> - Every metrics file reports both baselines next to ridge and hgb. Persistence test R^2 is 0.5099, MAE 840.1. hgb with ratio28 scores test R^2 0.6392, MAE 744.2. Ridge under ratio28 falls to test R^2 -0.5069, and that is reported too.
>
> Files reviewed:
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/src/config.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/src/features.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/src/ingest.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/src/train.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/tests/test_contract.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/tests/test_leakage.py
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/docs/DECISIONS.md
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/reports/metrics.json
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/reports/metrics_target_ratio28.json
> - /Users/melvin/Documents/DSC180A/teammate-bikeshare-open/reports/metrics_target_diff1.json

### Re-checked independently

The auditor's report is a claim, not evidence, so the parts that carry its
verdict were re-run here:

```
$ .venv/bin/python -m pytest tests -q                      # in their repo
23 passed in 1.62s
$ .venv/bin/python -m src.train && git diff --stat reports/
                                                           # empty: metrics reproduce
$ git show cce8d69:reports/metrics_target_ratio28.json | grep -c '"test"'
0                                                          # transforms compared on validation only
$ git log --format="%h %ci %s"
af90030 2026-11-25 22:09:30 -0800 Ex4: single test run of ratio28; ...
73b3fb3 2026-11-25 22:06:58 -0800 Ex4: pre-register ratio28 for the single test run ...
```

`OnTransformedTarget.predict` (their `src/train.py:106-108`) applies the inverse
before the scoring calls at lines 150-151, as the auditor says.

### My verdict

I agree with the auditor: **no finding changes a number.** Every committed
metric reproduces from a clean run, the new feature is a correct calendar lag
with the gap handled, the transform is inverted with lagged features only, and
the test set was scored once for Exercise 4 after the choice was recorded.

Two points I would raise with them that do not change a number but bear on
how the result is phrased:

- **Exercise 4's "hgb competes" rests on a 0.015 R^2 margin** (hgb-ratio28 test
  0.6392 vs ridge 0.6239) chosen from two candidate transforms. Their
  `DECISIONS.md` already says this is inside the noise of 184 test days and
  claims "competes", not "wins"; the report should keep that wording.
- **Same-day weather** (`temp`, `atemp`, `hum`, `windspeed`,
  `weather_severity`) is the measured weather of the target day. That is the
  starter's design and is shared by every repo, so it is not their error, but a
  day-ahead forecaster would only have a forecast; the scores are optimistic
  for deployment. This was raised by `agent/audit.py` on this repo (Part 2) and
  applies to theirs equally.

## Part 2: `agent/audit.py` on this repository

```
$ AUDIT_MODEL=opus python agent/audit.py . --out reports/audit.jsonl
audited 1 repos; 0 flagged -> reports/audit.jsonl
```

Run against this repo at `073c3e6` (after Exercises 1 and 2). Result:
`leakage: none`, `temporal_ok: true`, `baseline_reported: true`,
`reproducible: true`, confidence medium, five findings. Full JSON in
`reports/audit.jsonl`. My response to each:

| Finding | Changes a number? | Response |
|---|---|---|
| Same-day weather used as features | No | Agreed. Starter design, kept; stated as a limitation in REPORT.md. |
| Ridge loses to persistence on test MAE (879.3 vs 840.1) | No | Agreed, and it is Exercise 5's question. Reported, not hidden. |
| `leakage_demo.py` uses casual + registered | No | Intentional demo, not in `make all`, never quoted as a result. |
| README said 14 tests; 17 run | No | Fixed: README now says 17. |
| `check_identity` ran on the raw CSV, not the parquet | No | Limitation of the run. The lag features are covered by `test_lags_are_true_calendar_lags`, `test_rolling_mean_excludes_today` and the two `cnt_roll_28` tests. |

**Test-set exposure:** this run read `reports/metrics.json`, which contains test
scores, and quoted them. All feature work (`cnt_roll_28`, kept on a validation
delta) was committed before this run, so nothing was chosen with them in view.
Recorded in `docs/DECISIONS.md`.
