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
from regoscan.features import FEATURE_NAMES, compute_features
from regoscan.preprocess import (
    apply_standardise,
    savgol_smooth,
    standardise,
)
from regoscan.schema import LED_COLS, SPEC_COLS


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


def baseline_feature_names() -> list[str]:
    """Return ordered feature names matching columns of :func:`build_baseline_features`.

    Layout: [spec_000..spec_287 | led_385..led_940 | lif_450lp | expert features].
    """
    names: list[str] = list(SPEC_COLS) + list(LED_COLS) + ["lif_450lp"]
    names.extend(FEATURE_NAMES)
    return names


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
# Feature importance via permutation
# ---------------------------------------------------------------------------


def compute_feature_importance(
    bundle_obj: BaselineBundle,
    val_bundle: NumpyBundle,
    *,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, object]:
    """Compute permutation importance of the RF classifier on validation data.

    Parameters
    ----------
    bundle_obj : BaselineBundle
        Trained baseline bundle.
    val_bundle : NumpyBundle
        Validation data (must not overlap with training data).
    n_repeats : int
        Number of permutation repeats.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``feature_names``, ``importances_mean``, ``importances_std``.
        Values are lists aligned to the feature columns.
    """
    from sklearn.inspection import permutation_importance

    X_val = build_baseline_features(val_bundle)
    X_val_s = apply_standardise(X_val, bundle_obj.feat_mean, bundle_obj.feat_std)
    y_val = val_bundle.class_idx

    result = permutation_importance(
        bundle_obj.rf,
        X_val_s,
        y_val,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )

    names = baseline_feature_names()
    return {
        "feature_names": names,
        "importances_mean": result.importances_mean.tolist(),
        "importances_std": result.importances_std.tolist(),
    }


def plot_feature_importance(
    importance_data: dict[str, object],
    out_path: str | Path,
    *,
    top_n: int = 20,
) -> Path:
    """Save a horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importance_data : dict
        Output of :func:`compute_feature_importance`.
    out_path : str or Path
        Destination PNG path.
    top_n : int
        Number of features to display.

    Returns
    -------
    Path
        The saved file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    names = np.array(importance_data["feature_names"])
    means = np.array(importance_data["importances_mean"])
    stds = np.array(importance_data["importances_std"])

    # Select top_n by mean importance (descending)
    order = np.argsort(means)[::-1][:top_n]
    # Reverse for horizontal bar chart (top feature on top)
    order = order[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * top_n)), dpi=140)
    ax.barh(
        np.arange(len(order)),
        means[order],
        xerr=stds[order],
        color="#00d1ff",
        edgecolor="#0a3b4a",
        capsize=3,
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(names[order], fontsize=8)
    ax.set_xlabel("Mean decrease in accuracy (permutation importance)")
    ax.set_title(f"Top {len(order)} feature importances (RF classifier)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


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
    "baseline_feature_names",
    "BaselineBundle",
    "fit_baseline",
    "save_baseline",
    "load_baseline",
    "compute_feature_importance",
    "plot_feature_importance",
]
