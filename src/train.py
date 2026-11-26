"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train
     python -m src.train --target ratio7 --valid-only   # Exercise 4

`make train` (no flags) is the committed pipeline. --target writes to
reports/metrics_target_<name>.json and never touches reports/metrics.json.
"""
import argparse
import json

import numpy as np

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, REPORTS, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


# --- Exercise 4: target transformations ---------------------------------
# Trees cannot predict above the largest target they saw, and 2012 rides sit
# above 2011's ceiling. Each transform re-expresses cnt relative to a level
# known at prediction time, so a tree only has to learn a bounded quantity.
# Every entry is (forward(y, X), inverse(pred, X)); the inverse may read only
# lagged features in X, never the true target. tests/test_target.py checks.
TARGET_TRANSFORMS = {
    # Control: compresses scale but keeps the ceiling (log of 2011's max).
    "log": (
        lambda y, X: np.log1p(y),
        lambda p, X: np.expm1(p),
    ),
    # Ride count as a multiple of the past week's mean.
    "ratio7": (
        lambda y, X: np.asarray(y, dtype=float) / X["cnt_roll_7"].to_numpy(dtype=float),
        lambda p, X: np.asarray(p, dtype=float) * X["cnt_roll_7"].to_numpy(dtype=float),
    ),
    # Change from the same weekday last week.
    "diff7": (
        lambda y, X: np.asarray(y, dtype=float) - X["cnt_lag_7"].to_numpy(dtype=float),
        lambda p, X: np.asarray(p, dtype=float) + X["cnt_lag_7"].to_numpy(dtype=float),
    ),
}


class TransformedTarget(BaseEstimator, RegressorMixin):
    """Fit `model` on a transformed target; predict back in rides.

    sklearn's TransformedTargetRegressor cannot do this: its transform sees
    only y, and these need a lagged level from X."""

    def __init__(self, model, transform: str):
        self.model = model
        self.transform = transform

    def fit(self, X, y):
        forward, _ = TARGET_TRANSFORMS[self.transform]
        self.model_ = clone(self.model).fit(X, forward(y, X))
        return self

    def predict(self, X):
        _, inverse = TARGET_TRANSFORMS[self.transform]
        return inverse(self.model_.predict(X), X)


def models_for(target: str = "none") -> dict:
    """Baselines are never transformed; ridge and hgb are."""
    if target == "none":
        return MODELS
    return {
        name: m if name.startswith("baseline") else TransformedTarget(m, target)
        for name, m in MODELS.items()
    }


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score(target: str = "none", valid_only: bool = False) -> dict:
    """Defaults reproduce the committed pipeline. valid_only never scores
    the test split; transforms are compared that way."""
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    splits = [("valid", valid)] if valid_only else [("valid", valid), ("test", test)]

    results = {} if target == "none" else {"target": target}
    results.update({
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    })
    for name, model in models_for(target).items():
        model.fit(train[cols], train[TARGET])
        results[name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split, part in splits
        }
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="none", choices=["none", *TARGET_TRANSFORMS])
    ap.add_argument("--valid-only", action="store_true")
    ns = ap.parse_args()
    if ns.target == "none" and ns.valid_only:
        ap.error("--valid-only is for comparing transforms; make train scores both splits")
    res = fit_and_score(ns.target, ns.valid_only)
    out = METRICS if ns.target == "none" else REPORTS / f"metrics_target_{ns.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
