"""Read/write the canonical Regoscan measurement CSV.

Everything that touches a CSV must go through this module so the schema
contract is enforced in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from regoscan.schema import (
    ALL_COLUMNS,
    LED_COLS,
    LIF_COL,
    Measurement,
    MINERAL_CLASSES,
    N_LED,
    N_SPEC,
    PACKING_DENSITIES,
    SPEC_COLS,
)


class SchemaError(ValueError):
    """Raised when a CSV does not match the canonical Regoscan schema."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate a DataFrame against the canonical schema.

    Cheap structural checks (column names, dtypes, value ranges). Does NOT
    instantiate :class:`~regoscan.schema.Measurement` for every row — that
    would be slow on large datasets.
    """
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing[:8]}...")

    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        raise SchemaError(f"unexpected columns present: {extra[:8]}...")

    if len(df) == 0:
        return

    bad_class = set(df["mineral_class"].unique()) - set(MINERAL_CLASSES)
    if bad_class:
        raise SchemaError(f"invalid mineral_class values: {sorted(bad_class)}")

    bad_pack = set(df["packing_density"].unique()) - set(PACKING_DENSITIES)
    if bad_pack:
        raise SchemaError(f"invalid packing_density values: {sorted(bad_pack)}")

    if (df["ilmenite_fraction"] < 0).any() or (df["ilmenite_fraction"] > 1).any():
        raise SchemaError("ilmenite_fraction outside [0, 1]")

    if (df["integration_time_ms"] <= 0).any():
        raise SchemaError("integration_time_ms must be > 0")

    spec_block = df[list(SPEC_COLS)].to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(spec_block)):
        raise SchemaError("spec_* columns contain NaN/Inf")

    led_block = df[list(LED_COLS)].to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(led_block)):
        raise SchemaError("led_* columns contain NaN/Inf")

    lif_vals = df[LIF_COL].to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(lif_vals)):
        raise SchemaError(f"{LIF_COL} contains NaN/Inf")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_measurements_csv(path: str | Path) -> pd.DataFrame:
    """Read a Regoscan CSV from disk and validate it.

    Returns a DataFrame whose columns are exactly :data:`ALL_COLUMNS`, in
    canonical order, with numeric blocks dtyped as ``float64``/``int64``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    validate_dataframe(df)

    df = df[list(ALL_COLUMNS)].copy()
    df["integration_time_ms"] = df["integration_time_ms"].astype(np.int64)
    df["ilmenite_fraction"] = df["ilmenite_fraction"].astype(np.float64)
    df["ambient_temp_c"] = df["ambient_temp_c"].astype(np.float64)
    for c in SPEC_COLS:
        df[c] = df[c].astype(np.float64)
    for c in LED_COLS:
        df[c] = df[c].astype(np.float64)
    df[LIF_COL] = df[LIF_COL].astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_measurements_csv(
    measurements: Iterable[Measurement] | pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write measurements to disk as a canonical CSV.

    Accepts either an iterable of :class:`Measurement` instances or an
    already-built DataFrame. In both cases the output is validated before
    being written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(measurements, pd.DataFrame):
        df = measurements[list(ALL_COLUMNS)].copy()
    else:
        rows = [m.to_row() for m in measurements]
        df = pd.DataFrame(rows, columns=list(ALL_COLUMNS))

    validate_dataframe(df)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Convenience extraction helpers
# ---------------------------------------------------------------------------


def extract_spectra(df: pd.DataFrame) -> np.ndarray:
    """Return the (N, 288) spectrometer block as a contiguous float64 array."""
    return np.ascontiguousarray(
        df[list(SPEC_COLS)].to_numpy(dtype=np.float64, copy=True)
    )


def extract_leds(df: pd.DataFrame) -> np.ndarray:
    """Return the (N, 12) LED narrowband block."""
    return np.ascontiguousarray(
        df[list(LED_COLS)].to_numpy(dtype=np.float64, copy=True)
    )


def extract_lif(df: pd.DataFrame) -> np.ndarray:
    """Return the (N,) LIF photodiode column."""
    return np.ascontiguousarray(df[LIF_COL].to_numpy(dtype=np.float64, copy=True))


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the (N, 301) concat of [spec | led | lif].

    This is the canonical raw feature matrix consumed by training/inference.
    Preprocessing is applied on top of this; the column ordering matches
    ``[*SPEC_COLS, *LED_COLS, LIF_COL]``.
    """
    spec = extract_spectra(df)
    led = extract_leds(df)
    lif = extract_lif(df).reshape(-1, 1)
    return np.concatenate([spec, led, lif], axis=1)


def extract_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(class_indices, ilmenite_fraction)`` arrays."""
    from regoscan.schema import CLASS_TO_INDEX

    class_idx = df["mineral_class"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    ilm = df["ilmenite_fraction"].to_numpy(dtype=np.float64)
    return class_idx, ilm


__all__ = [
    "SchemaError",
    "validate_dataframe",
    "read_measurements_csv",
    "write_measurements_csv",
    "extract_spectra",
    "extract_leds",
    "extract_lif",
    "extract_feature_matrix",
    "extract_labels",
]
