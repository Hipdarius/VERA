"""Tests for the FastAPI REST endpoints (apps/api.py).

Uses FastAPI's ``TestClient`` which wraps ``requests`` internally.
Model-dependent tests are skipped when the ONNX checkpoint is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable, matching the pattern in apps/api.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODEL_PATH = _PROJECT_ROOT / "runs" / "cnn_v2" / "model.onnx"
_model_exists = _MODEL_PATH.exists()

try:
    import onnxruntime  # noqa: F401
    _has_ort = True
except ImportError:
    _has_ort = False

requires_model = pytest.mark.skipif(
    not (_model_exists and _has_ort),
    reason="trained ONNX model or onnxruntime not available",
)

from regoscan.schema import (  # noqa: E402
    MINERAL_CLASSES,
    N_FEATURES_TOTAL,
    N_LED,
    N_SPEC,
    SCHEMA_VERSION,
    WAVELENGTHS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app.

    Uses ``scope="module"`` so the (potentially expensive) model load
    happens only once per test module.
    """
    from fastapi.testclient import TestClient

    # Import the app — this triggers module-level model loading in apps/api.py.
    # We add the project root so `apps.api` resolves as a dotted module.
    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from apps.api import app  # noqa: E402

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


class TestHealthz:
    """Liveness probe."""

    def test_returns_200(self, client) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_has_model_loaded_field(self, client) -> None:
        body = client.get("/healthz").json()
        assert "model_loaded" in body
        assert isinstance(body["model_loaded"], bool)

    def test_has_schema_version(self, client) -> None:
        body = client.get("/healthz").json()
        assert body["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# GET /api/meta
# ---------------------------------------------------------------------------


class TestMeta:
    """Schema / configuration metadata."""

    def test_returns_200(self, client) -> None:
        resp = client.get("/api/meta")
        assert resp.status_code == 200

    def test_schema_version(self, client) -> None:
        body = client.get("/api/meta").json()
        assert body["schema_version"] == SCHEMA_VERSION

    def test_wavelengths_count(self, client) -> None:
        body = client.get("/api/meta").json()
        assert len(body["wavelengths_nm"]) == N_SPEC  # 288

    def test_n_features_total(self, client) -> None:
        body = client.get("/api/meta").json()
        assert body["n_features_total"] == N_FEATURES_TOTAL  # 301

    def test_class_names(self, client) -> None:
        body = client.get("/api/meta").json()
        assert body["class_names"] == list(MINERAL_CLASSES)
        assert len(body["class_names"]) == 5

    def test_led_wavelengths(self, client) -> None:
        body = client.get("/api/meta").json()
        assert len(body["led_wavelengths_nm"]) == N_LED  # 12


# ---------------------------------------------------------------------------
# POST /api/predict/demo
# ---------------------------------------------------------------------------


@requires_model
class TestPredictDemo:
    """Demo endpoint that synthesises a spectrum and predicts it."""

    def test_returns_200(self, client) -> None:
        resp = client.post("/api/predict/demo")
        assert resp.status_code == 200

    def test_response_has_expected_fields(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert "predicted_class" in body
        assert "confidence" in body
        assert "probabilities" in body
        assert "ilmenite_fraction" in body
        assert "spec" in body
        assert "led" in body
        assert "lif_450lp" in body

    def test_spec_and_led_shapes(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert len(body["spec"]) == N_SPEC
        assert len(body["led"]) == N_LED

    def test_probabilities_count(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert len(body["probabilities"]) == 5

    def test_predicted_class_is_valid(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert body["predicted_class"] in MINERAL_CLASSES

    def test_confidence_in_range(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_ilmenite_fraction_in_range(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert 0.0 <= body["ilmenite_fraction"] <= 1.0

    def test_seed_produces_deterministic_result(self, client) -> None:
        b1 = client.post("/api/predict/demo?seed=42").json()
        b2 = client.post("/api/predict/demo?seed=42").json()
        assert b1["predicted_class"] == b2["predicted_class"]
        assert b1["spec"] == b2["spec"]

    def test_true_class_field_present(self, client) -> None:
        body = client.post("/api/predict/demo").json()
        assert body["true_class"] in MINERAL_CLASSES
        assert 0.0 <= body["true_ilmenite_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# POST /api/predict  (full-feature)
# ---------------------------------------------------------------------------


@requires_model
class TestPredict:
    """Full-feature inference endpoint."""

    @pytest.fixture()
    def valid_payload(self) -> dict:
        rng = np.random.default_rng(0)
        return {
            "spec": rng.uniform(0.1, 0.9, size=N_SPEC).tolist(),
            "led": rng.uniform(0.1, 0.9, size=N_LED).tolist(),
            "lif_450lp": 0.35,
        }

    def test_returns_200_with_valid_payload(self, client, valid_payload) -> None:
        resp = client.post("/api/predict", json=valid_payload)
        assert resp.status_code == 200

    def test_response_structure(self, client, valid_payload) -> None:
        body = client.post("/api/predict", json=valid_payload).json()
        assert "predicted_class" in body
        assert "predicted_class_index" in body
        assert "probabilities" in body
        assert "ilmenite_fraction" in body
        assert "confidence" in body
        assert "model_version" in body

    def test_predicted_class_valid(self, client, valid_payload) -> None:
        body = client.post("/api/predict", json=valid_payload).json()
        assert body["predicted_class"] in MINERAL_CLASSES

    def test_confidence_in_range(self, client, valid_payload) -> None:
        body = client.post("/api/predict", json=valid_payload).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_ilmenite_fraction_in_range(self, client, valid_payload) -> None:
        body = client.post("/api/predict", json=valid_payload).json()
        assert 0.0 <= body["ilmenite_fraction"] <= 1.0

    def test_probabilities_sum_to_one(self, client, valid_payload) -> None:
        body = client.post("/api/predict", json=valid_payload).json()
        total = sum(p["probability"] for p in body["probabilities"])
        assert total == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# POST /api/predict — validation / error paths
# ---------------------------------------------------------------------------


class TestPredictValidation:
    """Error handling for malformed requests (no model needed for 422s)."""

    def test_empty_body_returns_422(self, client) -> None:
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 422

    def test_missing_spec_returns_422(self, client) -> None:
        payload = {
            "led": [0.5] * N_LED,
            "lif_450lp": 0.3,
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_wrong_spec_length_returns_422(self, client) -> None:
        payload = {
            "spec": [0.5] * 100,  # wrong length
            "led": [0.5] * N_LED,
            "lif_450lp": 0.3,
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_wrong_led_length_returns_422(self, client) -> None:
        payload = {
            "spec": [0.5] * N_SPEC,
            "led": [0.5] * 3,  # wrong length
            "lif_450lp": 0.3,
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_lif_returns_422(self, client) -> None:
        payload = {
            "spec": [0.5] * N_SPEC,
            "led": [0.5] * N_LED,
            # lif_450lp missing
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_non_numeric_spec_returns_422(self, client) -> None:
        payload = {
            "spec": ["abc"] * N_SPEC,
            "led": [0.5] * N_LED,
            "lif_450lp": 0.3,
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/endmembers
# ---------------------------------------------------------------------------


_ENDMEMBERS_NPZ = _PROJECT_ROOT / "data" / "cache" / "usgs_endmembers.npz"

requires_endmembers = pytest.mark.skipif(
    not _ENDMEMBERS_NPZ.exists(),
    reason="endmember cache not found",
)


@requires_endmembers
class TestEndmembers:
    """Endmember spectra endpoint."""

    def test_returns_200(self, client) -> None:
        resp = client.get("/api/endmembers")
        assert resp.status_code == 200

    def test_has_wavelengths_and_endmembers(self, client) -> None:
        body = client.get("/api/endmembers").json()
        assert "wavelengths_nm" in body
        assert "endmembers" in body

    def test_four_endmember_spectra(self, client) -> None:
        body = client.get("/api/endmembers").json()
        assert len(body["endmembers"]) == 4

    def test_each_endmember_has_name_and_spectrum(self, client) -> None:
        body = client.get("/api/endmembers").json()
        n_wavelengths = len(body["wavelengths_nm"])
        for em in body["endmembers"]:
            assert "name" in em
            assert "spectrum" in em
            assert len(em["spectrum"]) == n_wavelengths
