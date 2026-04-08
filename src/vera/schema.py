"""Canonical measurement schema for VERA.

This module is the **contract** between the (future) hardware acquisition
layer and the entire software stack. Every other module that touches a
measurement reads or writes through this schema.

Locked. Do not change column names without bumping ``SCHEMA_VERSION``.

The 288 spectrometer channels span 340–850 nm on a Hamamatsu C12880MA
(~1.78 nm/px). Because that step is non-integer, columns are named by index
(``spec_000 .. spec_287``) and the actual wavelength grid is exposed as
:data:`WAVELENGTHS`.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Spectrometer grid (Hamamatsu C12880MA)
# ---------------------------------------------------------------------------

N_SPEC: Final[int] = 288
SPEC_LAMBDA_MIN_NM: Final[float] = 340.0
SPEC_LAMBDA_MAX_NM: Final[float] = 850.0
WAVELENGTHS: Final[np.ndarray] = np.linspace(
    SPEC_LAMBDA_MIN_NM, SPEC_LAMBDA_MAX_NM, N_SPEC, dtype=np.float64
)
SPEC_COLS: Final[tuple[str, ...]] = tuple(f"spec_{i:03d}" for i in range(N_SPEC))

# ---------------------------------------------------------------------------
# LED narrowband channels (one reflectance value per LED)
# ---------------------------------------------------------------------------

LED_WAVELENGTHS_NM: Final[tuple[int, ...]] = (
    385, 405, 450, 500, 525, 590, 625, 660, 730, 780, 850, 940,
)
LED_COLS: Final[tuple[str, ...]] = tuple(f"led_{w}" for w in LED_WAVELENGTHS_NM)
N_LED: Final[int] = len(LED_COLS)

# ---------------------------------------------------------------------------
# LIF channel
# ---------------------------------------------------------------------------

LIF_COL: Final[str] = "lif_450lp"  # 405 nm excitation, 450 nm longpass

# ---------------------------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------------------------

META_COLS: Final[tuple[str, ...]] = (
    "sample_id",
    "measurement_id",
    "timestamp",
    "mineral_class",
    "ilmenite_fraction",
    "integration_time_ms",
    "ambient_temp_c",
    "packing_density",
)

# ---------------------------------------------------------------------------
# Mineral class label space
# ---------------------------------------------------------------------------

MINERAL_CLASSES: Final[tuple[str, ...]] = (
    "ilmenite_rich",
    "olivine_rich",
    "pyroxene_rich",
    "anorthositic",
    "mixed",
)
N_CLASSES: Final[int] = len(MINERAL_CLASSES)
CLASS_TO_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(MINERAL_CLASSES)}
INDEX_TO_CLASS: Final[dict[int, str]] = {i: c for c, i in CLASS_TO_INDEX.items()}

PACKING_DENSITIES: Final[tuple[str, ...]] = ("loose", "medium", "packed")

# ---------------------------------------------------------------------------
# Full ordered column list
# ---------------------------------------------------------------------------

ALL_COLUMNS: Final[tuple[str, ...]] = (
    *META_COLS,
    *SPEC_COLS,
    *LED_COLS,
    LIF_COL,
)

N_FEATURES_TOTAL: Final[int] = N_SPEC + N_LED + 1  # 288 + 12 + 1 = 301


# ---------------------------------------------------------------------------
# Pydantic model for a single measurement row
# ---------------------------------------------------------------------------


PackingDensity = Literal["loose", "medium", "packed"]
MineralClass = Literal[
    "ilmenite_rich", "olivine_rich", "pyroxene_rich", "anorthositic", "mixed"
]


class Measurement(BaseModel):
    """One row of the canonical CSV.

    Validates types, ranges, and array lengths. Use this at the boundary —
    e.g., when ingesting a new acquisition or sanity-checking a synthetic
    dataset — not on every row of a hot inner loop.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    sample_id: str = Field(min_length=1)
    measurement_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    mineral_class: MineralClass
    ilmenite_fraction: float = Field(ge=0.0, le=1.0)
    integration_time_ms: int = Field(gt=0)
    ambient_temp_c: float
    packing_density: PackingDensity

    spec: list[float] = Field(min_length=N_SPEC, max_length=N_SPEC)
    led: list[float] = Field(min_length=N_LED, max_length=N_LED)
    lif_450lp: float

    @field_validator("spec", "led")
    @classmethod
    def _all_finite(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("spectrum / LED values must all be finite")
        return v

    @field_validator("lif_450lp")
    @classmethod
    def _lif_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("lif_450lp must be finite")
        return v

    def to_row(self) -> dict[str, object]:
        """Flatten this measurement to a dict matching :data:`ALL_COLUMNS`."""
        row: dict[str, object] = {
            "sample_id": self.sample_id,
            "measurement_id": self.measurement_id,
            "timestamp": self.timestamp,
            "mineral_class": self.mineral_class,
            "ilmenite_fraction": float(self.ilmenite_fraction),
            "integration_time_ms": int(self.integration_time_ms),
            "ambient_temp_c": float(self.ambient_temp_c),
            "packing_density": self.packing_density,
        }
        for col, val in zip(SPEC_COLS, self.spec):
            row[col] = float(val)
        for col, val in zip(LED_COLS, self.led):
            row[col] = float(val)
        row[LIF_COL] = float(self.lif_450lp)
        return row

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "Measurement":
        """Inverse of :meth:`to_row` — build a model from a dict-shaped row."""
        spec = [float(row[c]) for c in SPEC_COLS]
        led = [float(row[c]) for c in LED_COLS]
        return cls(
            sample_id=str(row["sample_id"]),
            measurement_id=str(row["measurement_id"]),
            timestamp=str(row["timestamp"]),
            mineral_class=str(row["mineral_class"]),  # type: ignore[arg-type]
            ilmenite_fraction=float(row["ilmenite_fraction"]),
            integration_time_ms=int(row["integration_time_ms"]),
            ambient_temp_c=float(row["ambient_temp_c"]),
            packing_density=str(row["packing_density"]),  # type: ignore[arg-type]
            spec=spec,
            led=led,
            lif_450lp=float(row[LIF_COL]),
        )


__all__ = [
    "SCHEMA_VERSION",
    "N_SPEC",
    "SPEC_LAMBDA_MIN_NM",
    "SPEC_LAMBDA_MAX_NM",
    "WAVELENGTHS",
    "SPEC_COLS",
    "LED_WAVELENGTHS_NM",
    "LED_COLS",
    "N_LED",
    "LIF_COL",
    "META_COLS",
    "MINERAL_CLASSES",
    "N_CLASSES",
    "CLASS_TO_INDEX",
    "INDEX_TO_CLASS",
    "PACKING_DENSITIES",
    "ALL_COLUMNS",
    "N_FEATURES_TOTAL",
    "Measurement",
    "MineralClass",
    "PackingDensity",
]
