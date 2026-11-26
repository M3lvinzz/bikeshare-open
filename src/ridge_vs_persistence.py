"""Exercise 5: is ridge worth deploying over "yesterday again"?

Reproduces every number in REPORT.md's Exercise 5 paragraph. Analysis only:
it fits the committed ridge on train exactly as src.train does and never
writes reports/metrics.json. The bias correction is estimated on validation
alone and then applied to test, so test chooses nothing.

Run: python -m src.ridge_vs_persistence
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

# A day where demand moved more than this from yesterday is a "swing" day:
# the days where persistence is wrong by construction.
SWING = 1500


def compare() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    ridge = clone(MODELS["ridge"]).fit(train[cols], train[TARGET])

    valid_bias = float((ridge.predict(valid[cols]) - valid[TARGET]).mean())

    y = test[TARGET].to_numpy(dtype=float)
    e_ridge = ridge.predict(test[cols]) - y
    e_corr = e_ridge - valid_bias
    e_pers = test["cnt_lag_1"].to_numpy(dtype=float) - y
    swing = np.abs(y - test["cnt_lag_1"].to_numpy(dtype=float)) > SWING
    a_r, a_c, a_p = np.abs(e_ridge), np.abs(e_corr), np.abs(e_pers)
    return {
        "valid_bias": valid_bias,
        "test_bias_ridge": float(e_ridge.mean()),
        "test_bias_persistence": float(e_pers.mean()),
        "mae_ridge": float(a_r.mean()),
        "mae_ridge_corrected": float(a_c.mean()),
        "mae_persistence": float(a_p.mean()),
        "median_ridge": float(np.median(a_r)),
        "median_ridge_corrected": float(np.median(a_c)),
        "median_persistence": float(np.median(a_p)),
        "n_swing": int(swing.sum()),
        "swing_mae_ridge": float(a_r[swing].mean()),
        "swing_mae_persistence": float(a_p[swing].mean()),
        "calm_mae_ridge": float(a_r[~swing].mean()),
        "calm_mae_persistence": float(a_p[~swing].mean()),
        "over2000_ridge": int((a_r > 2000).sum()),
        "over2000_persistence": int((a_p > 2000).sum()),
        "closer_ridge": float((a_r < a_p).mean()),
        "closer_ridge_corrected": float((a_c < a_p).mean()),
    }


if __name__ == "__main__":
    r = compare()
    for k, v in r.items():
        print(f"  {k:24s} {v:10.3f}" if isinstance(v, float) else f"  {k:24s} {v:10d}")
    print(
        f"OK ridge_vs_persistence: test MAE ridge {r['mae_ridge']:.1f}, "
        f"corrected {r['mae_ridge_corrected']:.1f}, persistence {r['mae_persistence']:.1f}"
    )
