---
name: ablation
description: Drop one feature, refit on the same rows, and report the validation R^2 delta. Use when someone asks whether a feature earns its place, wants to remove or justify a feature, or says "ablate X".
allowed-tools: Read, Edit, Bash(python -m src.ablation*), Bash(make*)
---

# Current feature set

!`python -c "import pandas as pd; from src.features import feature_columns; print(', '.join(feature_columns(pd.read_parquet('data/processed/features.parquet'))))"`

# Ablating: $ARGUMENTS

1. **Check the name.** `$ARGUMENTS` must be one of the features listed above.
   If it is a raw column or a typo, say so and stop.

2. **Run on the same rows.** `make features`, then
   `python -m src.ablation $ARGUMENTS`. Do not delete the feature from
   `src/features.py` and rebuild: that can change LAG_WARMUP and the training
   rows, and the delta would mix a sample effect into the feature effect.

3. **Report validation only.** Give ridge and hgb R^2 and MAE with and without
   the feature, from the printed output. Never score or quote the test split
   for this; it is touched once, at the end of the project.

4. **Judge it.** If neither model's validation R^2 drops by at least 0.01 when
   the feature is removed, say plainly that the feature is inside the noise of
   182 validation days and recommend dropping it. If one model gains and the
   other loses, say which, and do not average them away.

5. **Record the decision** as one dated line at the end of `docs/DECISIONS.md`:
   the feature, both deltas, and keep or drop. Removing a feature is a
   change to `src/features.py` through `/new-feature`'s rules in reverse, done
   only after the decision is recorded.
