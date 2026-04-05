"""Lightweight ONNX-runtime inference for the Regoscan web/serverless layer.

This module is the **only** place that talks to onnxruntime. Both the
local FastAPI dev server (``apps/api.py``) and the Vercel serverless
handler (``web/api/predict.py``) import from here so there is exactly
one inference code path.

We deliberately avoid importing torch from here — torch is too large to
fit in Vercel's serverless Python runtime. The trained .onnx file is the
portable artefact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from regoscan.schema import (
    LED_WAVELENGTHS_NM,
    MINERAL_CLASSES,
    N_FEATURES_TOTAL,
    N_LED,
    N_SPEC,
    WAVELENGTHS,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Thin wrapper over an onnxruntime ``InferenceSession``.

    The Regoscan ResNet exports two outputs: ``logits`` (shape ``(B, 5)``)
    and ``ilmenite`` (shape ``(B,)`` already passed through sigmoid).
    """

    def __init__(self, onnx_path: str | Path) -> None:
        # Lazy import so the module is still importable in environments
        # where onnxruntime isn't installed (tests that don't touch the
        # API path, doc generation, etc.).
        import onnxruntime as ort  # noqa: WPS433

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self._path = onnx_path
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        # Output ordering is fixed by quantize.export_onnx: logits, ilmenite.
        self._logits_name = outputs[0].name
        self._ilm_name = outputs[1].name

    @property
    def version(self) -> str:
        return f"regoscan-resnet-{self._path.stem}"

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Run inference on a single 301-vector or a batch of them.

        Returns a dict with ``probabilities`` (length 5), ``class_index``,
        and ``ilmenite_fraction``. For batched inputs, the caller is
        responsible for indexing — this helper assumes single-sample use.
        """
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            if x.shape[0] != N_FEATURES_TOTAL:
                raise ValueError(
                    f"expected {N_FEATURES_TOTAL} features, got {x.shape[0]}"
                )
            x = x.reshape(1, 1, N_FEATURES_TOTAL)
        elif x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        elif x.ndim == 3:
            pass
        else:
            raise ValueError(f"unsupported feature ndim={x.ndim}")

        logits, ilm = self._session.run(
            [self._logits_name, self._ilm_name], {self._input_name: x}
        )
        probs = _softmax(logits[0])
        return {
            "probabilities": probs,
            "class_index": int(np.argmax(probs)),
            "ilmenite_fraction": float(ilm[0]),
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Demo / endmember helpers
# ---------------------------------------------------------------------------
#
# These keep the web demo self-contained: the frontend can request a
# synthetic spectrum without any CSV upload, and the reference plot can
# render the four endmembers as faint baselines.


_DEFAULT_ENDMEMBERS_NPZ = (
    Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "usgs_endmembers.npz"
)


def synth_demo_features(seed: int | None = None) -> dict[str, Any]:
    """Generate one random measurement using the project's synth engine.

    Imported lazily so this module stays cheap if a caller only wants the
    InferenceEngine.
    """
    from regoscan.synth import (  # noqa: WPS433
        Endmembers,
        NoiseConfig,
        fractions_for_class,
        load_endmembers,
        synth_measurement,
    )

    rng = np.random.default_rng(seed)
    cls = str(rng.choice(MINERAL_CLASSES))

    if _DEFAULT_ENDMEMBERS_NPZ.exists():
        endmembers = load_endmembers(_DEFAULT_ENDMEMBERS_NPZ)
    else:
        endmembers = _toy_endmembers()

    fractions = fractions_for_class(cls, rng)
    measurement = synth_measurement(
        sample_id="demo",
        mineral_class=cls,
        fractions=fractions,
        endmembers=endmembers,
        rng=rng,
        noise=NoiseConfig(),
    )

    spec = np.asarray(measurement.spec, dtype=np.float32)
    led = np.asarray(measurement.led, dtype=np.float32)
    lif = float(measurement.lif_450lp)
    features = np.concatenate([spec, led, np.asarray([lif], dtype=np.float32)])

    return {
        "features": features,
        "spec": spec,
        "led": led,
        "lif_450lp": lif,
        "true_class": cls,
        "true_ilmenite_fraction": float(measurement.ilmenite_fraction),
    }


def load_endmembers_payload() -> dict[str, Any]:
    """Return endmember spectra as plain JSON for the frontend chart."""
    from regoscan.synth import load_endmembers, ENDMEMBER_NAMES  # noqa: WPS433

    if not _DEFAULT_ENDMEMBERS_NPZ.exists():
        raise FileNotFoundError(
            f"endmember cache not found at {_DEFAULT_ENDMEMBERS_NPZ}; "
            "run scripts/download_usgs.py"
        )
    em = load_endmembers(_DEFAULT_ENDMEMBERS_NPZ)
    return {
        "wavelengths_nm": [float(w) for w in em.wavelengths_nm],
        "led_wavelengths_nm": list(LED_WAVELENGTHS_NM),
        "endmembers": [
            {
                "name": ENDMEMBER_NAMES[i],
                "spectrum": [float(v) for v in em.spectra[i]],
            }
            for i in range(em.n_endmembers)
        ],
        "source": em.source,
    }


def _toy_endmembers():
    """Trivial linear endmembers used as a fallback for self-test."""
    from regoscan.synth import Endmembers  # noqa: WPS433

    lam = WAVELENGTHS
    x = (lam - lam.min()) / (lam.max() - lam.min())
    olivine = 0.20 + 0.60 * x
    pyroxene = 0.15 + 0.50 * x
    anorthite = 0.55 + 0.30 * x
    ilmenite = 0.05 + 0.05 * x
    spectra = np.stack([olivine, pyroxene, anorthite, ilmenite], axis=0)
    return Endmembers(wavelengths_nm=lam, spectra=spectra, source="toy")
