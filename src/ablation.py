"""Drop one feature, refit ridge and hgb on the same rows, report the
validation delta. Validation only: the test split is never scored.

Same rows matters. Removing a long-window feature and rebuilding would also
change LAG_WARMUP and the training set, mixing a sample effect into the
feature effect (raising LAG_WARMUP to 28 alone moved the mean baseline by
0.24 R^2). This module refits on the existing table with one column left out.

Run: python -m src.ablation <feature>
"""
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

# Below this, a validation R^2 change is inside the noise of 182 days.
NOISE_FLOOR = 0.01


def ablate(feature: str) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    if feature not in cols:
        raise SystemExit(f"{feature!r} is not a model feature; features are: {', '.join(cols)}")
    without = [c for c in cols if c != feature]
    train, valid, _ = temporal_split(df)

    out = {"feature": feature, "n_train": len(train), "n_valid": len(valid)}
    for name in ("ridge", "hgb"):
        scores = {}
        for label, use in (("with", cols), ("without", without)):
            pred = clone(MODELS[name]).fit(train[use], train[TARGET]).predict(valid[use])
            scores[label] = (r2_score(valid[TARGET], pred), mean_absolute_error(valid[TARGET], pred))
        out[name] = {
            "r2_with": round(scores["with"][0], 4),
            "r2_without": round(scores["without"][0], 4),
            "r2_delta": round(scores["with"][0] - scores["without"][0], 4),
            "mae_with": round(scores["with"][1], 1),
            "mae_without": round(scores["without"][1], 1),
        }
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.ablation <feature>")
    res = ablate(sys.argv[1])
    for name in ("ridge", "hgb"):
        r = res[name]
        print(
            f"  {name:5s} valid R^2 {r['r2_without']:.4f} -> {r['r2_with']:.4f} "
            f"(delta {r['r2_delta']:+.4f}), MAE {r['mae_without']:.1f} -> {r['mae_with']:.1f}"
        )
    best = max(res["ridge"]["r2_delta"], res["hgb"]["r2_delta"])
    verdict = "worth keeping" if best >= NOISE_FLOOR else "inside the noise"
    print(f"OK ablation: {res['feature']} on {res['n_train']} train / {res['n_valid']} valid rows, "
          f"best delta {best:+.4f} ({verdict})")
