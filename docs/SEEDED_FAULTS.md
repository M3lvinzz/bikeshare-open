# Seeded faults

Each fault was introduced with the agent's Edit tool, so the PostToolUse hook
(`.claude/hooks/check_leakage.sh`) ran exactly as it would on a real mistake.
"Hook stderr" is what the hook returned to the agent; "`make test`" is the
full suite run afterwards. All output below is pasted from the actual runs.

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
