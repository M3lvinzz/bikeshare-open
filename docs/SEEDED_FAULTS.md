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
