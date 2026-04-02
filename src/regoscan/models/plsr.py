"""Baseline mineral classifier + ilmenite regressor.

Two stacked sklearn models on the canonical (preprocessed spectrum +
LED + LIF + hand-crafted features) feature vector:

1. **PLSR** for the continuous ilmenite-fraction regression target.
2. **RandomForest** for the 5-way mineral class. RF is more robust on
   small datasets than a softmax-on-PLS scheme and gets us off the ground
   without hyperparameter sweeps.

Both share the same input vector and the same fit/predict surface so
``train.py`` and ``evaluate.py`` can treat them as one bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier

from regoscan.datasets import NumpyBundle
from regoscan.features import compute_features
from regoscan.preprocess import (
    apply_standardise,
    savgol_smooth,
    standardise,
)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def build_baseline_features(bundle: NumpyBundle) -> np.ndarray:
    """Smooth spectra, then concat [smoothed-spectrum | LED | LIF | hand-crafted].

    The baseline doesn't see raw counts — only the same preprocessed
    representation we'd ship to the CNN, plus a small set of expert
    features so the linear PLS model has something to chew on.
    """
    spec_smoothed = savgol_smooth(bundle.spectra, window_length=11, polyorder=3)
    expert = compute_features(spec_smoothed, bundle.leds, bundle.lif)
    return np.concatenate(
        [
            spec_smoothed,
            bundle.leds,
            bundle.lif.reshape(-1, 1),
            expert,
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Bundle of two sklearn models with one fit/predict surface
# ---------------------------------------------------------------------------


@dataclass
class BaselineBundle:
    rf: RandomForestClassifier
    plsr: PLSRegression
    feat_mean: np.ndarray
    feat_std: np.ndarray

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = apply_standardise(X, self.feat_mean, self.feat_std)
        cls = self.rf.predict(Xs)
        ilm = self.plsr.predict(Xs).ravel()
        ilm = np.clip(ilm, 0.0, 1.0)
        return cls.astype(np.int64), ilm.astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = apply_standardise(X, self.feat_mean, self.feat_std)
        return self.rf.predict_proba(Xs)


def fit_baseline(
    train_bundle: NumpyBundle,
    *,
    n_components: int = 8,
    n_estimators: int = 200,
    seed: int = 0,
) -> BaselineBundle:
    X = build_baseline_features(train_bundle)
    y_cls = train_bundle.class_idx
    y_ilm = train_bundle.ilmenite

    Xs, mean, std = standardise(X, axis=0)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(Xs, y_cls)

    n_components_eff = max(1, min(n_components, Xs.shape[0] - 1, Xs.shape[1]))
    plsr = PLSRegression(n_components=n_components_eff, scale=False)
    plsr.fit(Xs, y_ilm)

    return BaselineBundle(rf=rf, plsr=plsr, feat_mean=mean, feat_std=std)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_baseline(bundle: BaselineBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_baseline(path: str | Path) -> BaselineBundle:
    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = [
    "build_baseline_features",
    "BaselineBundle",
    "fit_baseline",
    "save_baseline",
    "load_baseline",
]
