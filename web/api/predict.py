"""Vercel Python serverless function for Regoscan inference.

This file is the **production** entry point. Vercel mounts every .py file
under ``web/api/`` as a serverless function and supports ASGI apps (like
FastAPI) via the ``app`` symbol.

Routes (relative to the function mount):
    GET  /api/predict/healthz
    GET  /api/predict/meta
    POST /api/predict
    POST /api/predict/demo

The Next.js client uses a same-origin rewrite (see vercel.json), so the
browser only ever talks to the same domain — no CORS dance.

Keep this file self-contained: do not import from the in-repo
``regoscan`` package, because Vercel only ships ``web/`` to the
serverless runtime. The trained model lives at ``web/api/model.onnx``
and is committed alongside this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schema constants — duplicated from regoscan.schema so this module is
# self-contained for the Vercel runtime.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
N_SPEC = 288
N_LED = 12
N_FEATURES_TOTAL = N_SPEC + N_LED + 1  # 301

WAVELENGTHS_NM: list[float] = list(np.linspace(340.0, 850.0, N_SPEC))
LED_WAVELENGTHS_NM: list[int] = [
    385, 405, 450, 500, 525, 590, 625, 660, 730, 780, 850, 940,
]
MINERAL_CLASSES: list[str] = [
    "ilmenite_rich",
    "olivine_rich",
    "pyroxene_rich",
    "anorthositic",
    "mixed",
]


# ---------------------------------------------------------------------------
# Model loading (cold start)
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(__file__).parent / "model.onnx"
_SESSION: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        if not _MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"model.onnx missing at {_MODEL_PATH}; export it with "
                    "`uv run python -m regoscan.quantize --run runs/cnn_v2 "
                    "--out web/api/model.onnx` and redeploy"
                ),
            )
        _SESSION = ort.InferenceSession(
            str(_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _SESSION


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _run_inference(features: np.ndarray) -> dict[str, Any]:
    sess = _get_session()
    x = features.astype(np.float32).reshape(1, 1, N_FEATURES_TOTAL)
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    logits, ilm = sess.run(out_names, {input_name: x})
    probs = _softmax(logits[0])
    cls_idx = int(np.argmax(probs))
    return {
        "predicted_class": MINERAL_CLASSES[cls_idx],
        "predicted_class_index": cls_idx,
        "probabilities": [
            {"name": MINERAL_CLASSES[i], "probability": float(p)}
            for i, p in enumerate(probs)
        ],
        "ilmenite_fraction": float(ilm[0]),
        "confidence": float(probs[cls_idx]),
        "model_version": "regoscan-resnet-v2",
    }


# ---------------------------------------------------------------------------
# Synthetic demo generator (so the UI can scan without a CSV upload)
#
# This is a simplified, self-contained version of regoscan.synth — just
# enough physics to produce a believable spectrum for the four endmembers
# and let the trained model classify it.
# ---------------------------------------------------------------------------


_LAM = np.linspace(340.0, 850.0, N_SPEC)
_LAM_NORM = (_LAM - _LAM.min()) / (_LAM.max() - _LAM.min())

# Linear endmembers — the same toy fallback used in regoscan.inference.
_ENDMEMBERS = np.stack(
    [
        0.20 + 0.60 * _LAM_NORM,  # olivine
        0.15 + 0.50 * _LAM_NORM,  # pyroxene
        0.55 + 0.30 * _LAM_NORM,  # anorthite
        0.05 + 0.05 * _LAM_NORM,  # ilmenite
    ]
)

_LIF_EFFICIENCY = {"olivine": 0.30, "pyroxene": 0.40, "anorthite": 0.85, "ilmenite": 0.0}
_END_NAMES = ("olivine", "pyroxene", "anorthite", "ilmenite")


def _fractions_for_class(klass: str, rng: np.random.Generator) -> np.ndarray:
    f = np.zeros(4, dtype=np.float64)
    if klass == "olivine_rich":
        f[0] = rng.uniform(0.55, 0.85); f[1] = rng.uniform(0.05, 0.20)
        f[2] = rng.uniform(0.05, 0.20); f[3] = rng.uniform(0.0, 0.05)
    elif klass == "pyroxene_rich":
        f[1] = rng.uniform(0.55, 0.85); f[0] = rng.uniform(0.05, 0.20)
        f[2] = rng.uniform(0.05, 0.20); f[3] = rng.uniform(0.0, 0.05)
    elif klass == "anorthositic":
        f[2] = rng.uniform(0.65, 0.90); f[0] = rng.uniform(0.02, 0.15)
        f[1] = rng.uniform(0.02, 0.15); f[3] = rng.uniform(0.0, 0.05)
    elif klass == "ilmenite_rich":
        f[3] = rng.uniform(0.35, 0.65); f[0] = rng.uniform(0.05, 0.25)
        f[1] = rng.uniform(0.05, 0.25); f[2] = rng.uniform(0.05, 0.20)
    else:  # mixed
        raw = rng.uniform(0.1, 0.4, size=4)
        f = raw
    return f / f.sum()


def _synth_demo(seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    klass = str(rng.choice(MINERAL_CLASSES))
    fractions = _fractions_for_class(klass, rng)

    # Linear mixture + multiplicative gain + polynomial baseline + shot noise
    base = (fractions[:, None] * _ENDMEMBERS).sum(axis=0)
    gain = 1.0 + rng.normal(0.0, 0.01, size=N_SPEC)
    spec = base * gain
    poly = np.polyval(rng.normal(0.0, 0.01, size=3), _LAM_NORM)
    spec = spec + poly
    intensity = float(rng.uniform(0.85, 1.05))
    spec = spec * intensity
    spec = spec + rng.normal(0.0, 0.005, size=N_SPEC)
    spec = np.clip(spec, 0.0, 1.5)

    # LED narrowband sampling — pick the nearest spectrometer pixel.
    led = np.array(
        [spec[int(np.argmin(np.abs(_LAM - lw)))] for lw in LED_WAVELENGTHS_NM]
    ) + rng.normal(0.0, 0.005, size=N_LED)
    led = np.clip(led, 0.0, 1.5)

    # LIF: efficiency-weighted, quenched by ilmenite.
    eff = sum(float(fractions[i]) * _LIF_EFFICIENCY[name] for i, name in enumerate(_END_NAMES))
    quench = (1.0 - float(fractions[3])) ** 1.5
    lif = float(max(eff * quench * intensity + rng.normal(0.0, 0.01), 0.0))

    return {
        "spec": spec.astype(np.float32),
        "led": led.astype(np.float32),
        "lif_450lp": lif,
        "true_class": klass,
        "true_ilmenite_fraction": float(fractions[3]),
    }


# ---------------------------------------------------------------------------
# FastAPI app — exported as `app` for Vercel's ASGI runtime.
# ---------------------------------------------------------------------------

app = FastAPI(title="Regoscan API (Vercel)", version="0.2.0")


class SpectrumRequest(BaseModel):
    spec: list[float] = Field(min_length=N_SPEC, max_length=N_SPEC)
    led: list[float] = Field(min_length=N_LED, max_length=N_LED)
    lif_450lp: float


@app.get("/api/predict/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _MODEL_PATH.exists(),
        "schema_version": SCHEMA_VERSION,
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "class_names": MINERAL_CLASSES,
        "wavelengths_nm": WAVELENGTHS_NM,
        "led_wavelengths_nm": LED_WAVELENGTHS_NM,
        "n_features_total": N_FEATURES_TOTAL,
        "model_loaded": _MODEL_PATH.exists(),
        "model_sha256": None,
        "model_run_dir": "vercel:/api",
    }


@app.post("/api/predict")
def predict(req: SpectrumRequest) -> dict[str, Any]:
    features = np.concatenate(
        [
            np.asarray(req.spec, dtype=np.float32),
            np.asarray(req.led, dtype=np.float32),
            np.asarray([req.lif_450lp], dtype=np.float32),
        ]
    )
    return _run_inference(features)


@app.post("/api/predict/demo")
def predict_demo(seed: int | None = None) -> dict[str, Any]:
    demo = _synth_demo(seed=seed)
    features = np.concatenate(
        [demo["spec"], demo["led"], np.asarray([demo["lif_450lp"]], dtype=np.float32)]
    )
    result = _run_inference(features)
    result["spec"] = demo["spec"].tolist()
    result["led"] = demo["led"].tolist()
    result["lif_450lp"] = demo["lif_450lp"]
    result["true_class"] = demo["true_class"]
    result["true_ilmenite_fraction"] = demo["true_ilmenite_fraction"]
    return result
