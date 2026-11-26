"""Exercise 4: target transformations must be honest round trips.

A transformed model is scored in rides, so every transform needs an exact
inverse that uses only features (lagged demand), never the true target.
"""
import numpy as np
import pytest

from src.config import TARGET
from src.features import build_features
from src.ingest import load_raw
from src.train import MODELS, TARGET_TRANSFORMS, fit_and_score, models_for


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


@pytest.mark.parametrize("name", sorted(TARGET_TRANSFORMS))
def test_transform_round_trips_to_rides(feats, name):
    forward, inverse = TARGET_TRANSFORMS[name]
    y = feats[TARGET].to_numpy(dtype=float)
    back = inverse(forward(y, feats), feats)
    assert np.allclose(back, y), f"{name} does not invert exactly"


@pytest.mark.parametrize("name", sorted(TARGET_TRANSFORMS))
def test_inverse_ignores_the_true_target(feats, name):
    """Scrambling cnt must not change what the inverse produces: the level
    it multiplies or adds back has to come from lagged features."""
    _, inverse = TARGET_TRANSFORMS[name]
    pred = np.ones(len(feats))
    scrambled = feats.assign(**{TARGET: feats[TARGET].sample(frac=1, random_state=0).to_numpy()})
    assert np.allclose(inverse(pred, feats), inverse(pred, scrambled))


def test_default_pipeline_is_untransformed():
    assert models_for("none") is MODELS


def test_baselines_are_never_transformed():
    for name in TARGET_TRANSFORMS:
        models = models_for(name)
        assert models["baseline_mean"] is MODELS["baseline_mean"]
        assert models["baseline_persistence"] is MODELS["baseline_persistence"]


def test_valid_only_never_scores_test():
    res = fit_and_score("ratio7", valid_only=True)
    for model in ("baseline_mean", "baseline_persistence", "ridge", "hgb"):
        assert set(res[model]) == {"valid"}
