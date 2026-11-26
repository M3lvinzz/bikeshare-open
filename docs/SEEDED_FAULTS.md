# Seeded faults

Each fault was introduced with the agent's Edit tool, so the PostToolUse hook
(`.claude/hooks/check_leakage.sh`) ran exactly as it would on a real mistake.
"Hook stderr" is what the hook returned to the agent; "`make test`" is the
full suite run afterwards. All output below is pasted from the actual runs.
Pass counts are those of the suite at the time (16, then 17 tests); tests
added later raise the total but not the failures shown. Each fault's commit
contains the suite exactly as it was when that output was captured.

## Exercise 1: the new feature's test, before the feature existed

Commit `d78980c` adds two tests for `cnt_roll_28` before `src/features.py`
knows about it. `make test` at that commit:

```
.........FF.....                                                         [100%]
=================================== FAILURES ===================================
E   AttributeError: 'DataFrame' object has no attribute 'cnt_roll_28'. Did you mean '.cnt_roll_7' instead of '.cnt_roll_28'?
E   AssertionError: 21 rows within 28 days after the gap survived: 2011-03-18 .. 2011-04-07
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_roll_28_is_mean_of_previous_28_calendar_days
FAILED tests/test_leakage.py::test_roll_28_never_spans_the_missing_day - Asse...
2 failed, 14 passed in 1.12s
```

Both fail for the right reason: the column does not exist yet, and with only
7-day features the 21 days from 2011-03-18 to 2011-04-07 survive even though
their 28-day window contains the missing 2011-03-10. Commit `13c7e76` makes
both pass (16 passed).

## Fault 1: a forbidden column added to `feature_columns`

**Change** (`src/features.py`, `feature_columns`):

```diff
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
```

**Hook stderr** (fired on the Edit itself):

```
Leakage tests failed after editing /Users/melvin/Documents/DSC180A/bikeshare-open/src/features.py:
feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7  cnt_roll_28
0   2011-01-31     31.0  WINTER  ...     141....142857  4071.321429
671 2012-12-31    731.0  WINTER  ...      920.0  1531.428571  3969.428571

[672 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_28', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 8 passed in 0.50s
```

**`make test`:**

```
E       AssertionError: forbidden columns in feature set: {'casual'}
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 15 passed in 1.40s
```

**What it would have done to the result:** `make train` still ran to
completion and wrote ridge validation R^2 = 0.8259 (honest: 0.6839). Half the
leaked identity is enough to inflate the score by 0.14 with no error.

**Turns to fix:** 1. The hook's message names the column and the test, and the
fix is reverting one line. The hook was silent on the fixing edit.

## Fault 2: `.shift(1)` deleted from `cnt_roll_7`

**Change** (`src/features.py`, `build_features`):

```diff
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(7).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(7).mean()
```

**Hook stderr:**

```
Leakage tests failed after editing /Users/melvin/Documents/DSC180A/bikeshare-open/src/features.py:
[672 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4605.285714285715) == 4724.428571428572 ± 0.00472443
E             
E             comparison failed
E             Obtained: 4605.285714285715
E             Expected: 4724.428571428572 ± 0.00472443

tests/test_leakage.py:57: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 8 passed in 0.29s
```

**`make test`:**

```
E           assert np.float64(4605.285714285715) == 4724.428571428572 ± 0.00472443
E             
E             comparison failed
E             Obtained: 4605.285714285715
E             Expected: 4724.428571428572 ± 0.00472443
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 15 passed in 1.65s
```

**What it would have done to the result:** ridge validation R^2 0.6839 ->
0.7541. Today's count is 1/7 of the feature, a quieter leak than fault 1 but
still a +0.07 "improvement" that the pipeline reports without complaint.

**Turns to fix:** 1 (restore `.shift(1)`).

**Weakness noticed:** the test checks one row (`iloc[200]`) and is wrapped in
`if all(d in by_date.index ...)`, so if that row's window ever contained a
missing day the test would pass without asserting anything. Checking every
row with a complete window, as `test_lags_are_true_calendar_lags` does, would
close that.

## Fault 3: the weather encoder's `raise` removed

**Change** (`src/features.py`, `encode_weather`):

```diff
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
```

**Hook stderr, first attempt: none.** The hook ran and `tests/test_leakage.py`
passed, so it exited 0 and said nothing. **`make test` with the guard gone:**

```
16 passed in 1.56s
```

The suite was green under this fault. Nothing exercised the guard, because the
real data only contains GOOD / MISTY / BAD. What that costs, shown in memory
with BAD relabelled the way a new export might spell it (`data/raw` untouched):

```
rows: 672 with known labels, 654 after BAD -> 'RAIN/SNOW'; 18 days silently dropped, no error
```

`test_only_expected_rows_are_dropped` would catch this only if the relabelled
file were the one on disk; nothing runs it against new data.

**Fix: a test of the guard itself.** Added to `tests/test_leakage.py` so the
hook runs it on every src edit:

```python
def test_weather_encoder_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unmapped weather labels"):
        encode_weather(pd.Series(["GOOD", "RAIN/SNOW"]))
```

**`make test` with the new test and the guard still removed:**

```
E       Failed: DID NOT RAISE ValueError
tests/test_leakage.py:88: Failed
FAILED tests/test_leakage.py::test_weather_encoder_rejects_unknown_labels - F...
1 failed, 16 passed in 1.48s
```

**Hook stderr, fault re-seeded after the test existed:**

```
Leakage tests failed after editing /Users/melvin/Documents/DSC180A/bikeshare-open/src/features.py:
.......F..                                                               [100%]
=================================== FAILURES ===================================
_________________ test_weather_encoder_rejects_unknown_labels __________________

    def test_weather_encoder_rejects_unknown_labels():
        """The real data only has known labels, so nothing else exercises the
        guard: delete the `raise` and every other test stays green while a
        relabelled export silently loses its BAD days to dropna()."""
>       with pytest.raises(ValueError, match="unmapped weather labels"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:88: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_weather_encoder_rejects_unknown_labels - F...
1 failed, 9 passed in 0.29s
```

**Turns to fix:** 3. Seeding it went unnoticed (green hook, green suite). One
turn to show the silent loss, one to write the missing test and see it fail,
one to restore the `raise` (hook silent, 17 passed). This is the fault that
justified a new test rather than just a revert.

## Fault 4: a `cp` into `data/raw/` from Bash

**Change:** a copy of the raw CSV with one extra column, `total_riders` =
`cnt` (a leak with an innocent name), copied over the original from Bash:

```
cp /tmp/bikeshare-ex/bike_daily_tampered.csv data/raw/bike_daily.csv
```

**What happened:** nothing stopped it, and nothing noticed.

- The permission rules `Write(./data/raw/**)` / `Edit(./data/raw/**)` bind the
  Write and Edit tools. A `cp` inside a Bash command is neither.
- The PostToolUse hook matches `Edit|Write` and only acts on `src/*.py`. It
  never ran.
- The pipeline and the suite accepted the file:

```
 M data/raw/bike_daily.csv
394db0d9d65573f4                                   # sha256, was a6bed1577f6e4de5
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
  temp, atemp, hum, windspeed, total_riders, dayofweek, is_weekend, is_holiday, is_workday, month_sin, month_cos, weather_severity, cnt_lag_1, cnt_lag_7, cnt_roll_7, cnt_roll_28
17 passed in 1.27s
ridge valid r2: 0.9999  total_riders in features: True
```

`total_riders` is numeric and not in `FORBIDDEN`, so `feature_columns()` let
it in, and `test_no_forbidden_features` cannot see it: it only checks names
that are already forbidden. Ridge validation R^2 went to 0.9999 with every
check green. Restored with `git checkout -- data/raw/bike_daily.csv` (sha256
back to `a6bed1577f6e4de5...`).

**What would have caught it:**

- **A hash check on the raw file (implemented).** `load_raw()` now compares
  the file's sha256 with `RAW_SHA256` in `src/config.py` and raises on any
  difference. Every stage and every test goes through `load_raw()`, so a
  changed file stops `make data`, `make features`, `make train` and `make test`
  alike, however it was changed. Diff:

```diff
# src/config.py
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"

# src/ingest.py
 def load_raw() -> pd.DataFrame:
+    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
+    if digest != RAW_SHA256:
+        raise ValueError(
+            f"{RAW.name} has changed (sha256 {digest[:16]}..., expected "
+            f"{RAW_SHA256[:16]}...). Raw data is immutable: restore it with "
+            f"`git checkout -- {RAW.relative_to(RAW.parents[2])}`."
+        )
```

  Run against the tampered copy (pointed at directly, since a second `cp` into
  `data/raw/` was refused by the session's permissions):

```
ValueError: bike_daily_tampered.csv has changed (sha256 394db0d9d65573f4..., expected a6bed1577f6e4de5...). Raw data is immutable: restore it with `git checkout -- tmp/bikeshare-ex/bike_daily_tampered.csv`.
```

  (The odd restore path is because `RAW` was pointed at `/tmp` for this run; for
  the real file it reads `git checkout -- data/raw/bike_daily.csv`.)

- **A PreToolUse hook on Bash** that exits 2 for any command mentioning
  `data/raw`. It blocks the write before it happens instead of detecting it
  afterwards, but it is pattern matching on command text (`cd data && cp ... raw/`
  gets past a naive one).
- **A filesystem permission:** `chmod -w data/raw/bike_daily.csv`. Stops `cp`
  outright, regardless of tool.
- **The unclassified-column test** (`tests/test_contract.py`, required
  separately) would also flag `total_riders` as a column in the raw file that
  `src/config.py` does not classify. It catches this particular change, but not
  edits to values, which the hash does.

**Turns to fix:** 1 to restore from git, 1 more to add the hash check.

**A bonus catch:** while the hash check was being added, the Edit to
`src/ingest.py` landed before its `import hashlib` did. The hook fired on that
intermediate state (`NameError: name 'hashlib' is not defined`, 9 errors in
`tests/test_leakage.py`), which is the hook doing its job on an unplanned
mistake rather than a seeded one.
