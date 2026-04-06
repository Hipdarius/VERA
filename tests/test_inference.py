"""Tests for the ONNX inference engine and demo helpers.

Covers the single inference code path shared by the FastAPI server and
the Vercel serverless handler.  Model-dependent tests are skipped when
``runs/cnn_v2/model.onnx`` is not present on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from regoscan.schema import (
    MINERAL_CLASSES,
    N_FEATURES_TOTAL,
    N_LED,
    N_SPEC,
)

# ---------------------------------------------------------------------------
# Paths & skip markers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "runs" / "cnn_v2" / "model.onnx"

_model_exists = _MODEL_PATH.exists()
requires_model = pytest.mark.skipif(
    not _model_exists,
    reason="trained ONNX model not found at runs/cnn_v2/model.onnx",
)

try:
    import onnxruntime  # noqa: F401
    _has_ort = True
except ImportError:
    _has_ort = False

requires_ort = pytest.mark.skipif(
    not _has_ort,
    reason="onnxruntime not installed",
)


# ---------------------------------------------------------------------------
# _softmax helper (pure numpy, no model needed)
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Unit tests for the internal _softmax helper."""

    def test_softmax_sums_to_one(self) -> None:
        from regoscan.inference import _softmax

        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        probs = _softmax(logits)
        assert probs.shape == (5,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_softmax_manual_calculation(self) -> None:
        """Compare against a hand-computed softmax for a simple case."""
        from regoscan.inference import _softmax

        logits = np.array([0.0, 1.0])
        probs = _softmax(logits)
        # softmax([0, 1]) = [1/(1+e), e/(1+e)]
        e = np.e
        expected = np.array([1.0 / (1.0 + e), e / (1.0 + e)])
        np.testing.assert_allclose(probs, expected, atol=1e-7)

    def test_softmax_uniform_logits(self) -> None:
        from regoscan.inference import _softmax

        probs = _softmax(np.array([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)

    def test_softmax_large_logits_no_overflow(self) -> None:
        """Numerical stability: large logits should not produce NaN/Inf."""
        from regoscan.inference import _softmax

        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(logits)
        assert np.all(np.isfinite(probs))
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_softmax_negative_logits(self) -> None:
        from regoscan.inference import _softmax

        probs = _softmax(np.array([-10.0, -9.0, -8.0]))
        assert np.all(probs > 0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# synth_demo_features (no model needed)
# ---------------------------------------------------------------------------


class TestSynthDemoFeatures:
    """Tests for the demo spectrum synthesiser."""

    def test_returns_correct_feature_shape(self) -> None:
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=42)
        features = demo["features"]
        assert features.shape == (N_FEATURES_TOTAL,), (
            f"expected ({N_FEATURES_TOTAL},), got {features.shape}"
        )

    def test_spec_led_lif_sizes(self) -> None:
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=42)
        assert demo["spec"].shape == (N_SPEC,)
        assert demo["led"].shape == (N_LED,)
        assert isinstance(demo["lif_450lp"], float)

    def test_features_are_finite(self) -> None:
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=99)
        assert np.all(np.isfinite(demo["features"]))

    def test_true_class_is_valid(self) -> None:
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=7)
        assert demo["true_class"] in MINERAL_CLASSES

    def test_true_ilmenite_fraction_in_range(self) -> None:
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=7)
        assert 0.0 <= demo["true_ilmenite_fraction"] <= 1.0

    def test_different_seeds_produce_different_spectra(self) -> None:
        from regoscan.inference import synth_demo_features

        d1 = synth_demo_features(seed=1)
        d2 = synth_demo_features(seed=2)
        assert not np.array_equal(d1["features"], d2["features"]), (
            "different seeds must produce different spectra"
        )

    def test_same_seed_is_deterministic(self) -> None:
        from regoscan.inference import synth_demo_features

        d1 = synth_demo_features(seed=123)
        d2 = synth_demo_features(seed=123)
        np.testing.assert_array_equal(d1["features"], d2["features"])

    def test_feature_vector_is_concat_of_parts(self) -> None:
        """features == concat(spec, led, [lif])."""
        from regoscan.inference import synth_demo_features

        demo = synth_demo_features(seed=50)
        expected = np.concatenate([
            demo["spec"],
            demo["led"],
            np.array([demo["lif_450lp"]], dtype=np.float32),
        ])
        np.testing.assert_array_equal(demo["features"], expected)


# ---------------------------------------------------------------------------
# load_endmembers_payload
# ---------------------------------------------------------------------------


_ENDMEMBERS_NPZ = _PROJECT_ROOT / "data" / "cache" / "usgs_endmembers.npz"

requires_endmembers = pytest.mark.skipif(
    not _ENDMEMBERS_NPZ.exists(),
    reason="endmember cache not found at data/cache/usgs_endmembers.npz",
)


@requires_endmembers
class TestLoadEndmembersPayload:
    """Tests for the endmember payload helper."""

    def test_returns_expected_keys(self) -> None:
        from regoscan.inference import load_endmembers_payload

        payload = load_endmembers_payload()
        assert "wavelengths_nm" in payload
        assert "led_wavelengths_nm" in payload
        assert "endmembers" in payload
        assert "source" in payload

    def test_endmembers_have_name_and_spectrum(self) -> None:
        from regoscan.inference import load_endmembers_payload

        payload = load_endmembers_payload()
        for em in payload["endmembers"]:
            assert "name" in em
            assert "spectrum" in em
            assert len(em["spectrum"]) == len(payload["wavelengths_nm"])

    def test_four_endmembers(self) -> None:
        from regoscan.inference import load_endmembers_payload

        payload = load_endmembers_payload()
        assert len(payload["endmembers"]) == 4


# ---------------------------------------------------------------------------
# InferenceEngine — model required
# ---------------------------------------------------------------------------


@requires_ort
@requires_model
class TestInferenceEngine:
    """Integration tests that load the real ONNX model."""

    @pytest.fixture()
    def engine(self):
        from regoscan.inference import InferenceEngine
        return InferenceEngine(_MODEL_PATH)

    def test_load_model(self, engine) -> None:
        """Engine initialises without error and has a version string."""
        assert engine.version.startswith("regoscan-resnet-")

    def test_predict_output_shape(self, engine) -> None:
        rng = np.random.default_rng(0)
        features = rng.uniform(0, 1, size=N_FEATURES_TOTAL).astype(np.float32)
        result = engine.predict(features)
        assert result["probabilities"].shape == (5,)
        assert isinstance(result["class_index"], int)
        assert isinstance(result["ilmenite_fraction"], float)

    def test_probabilities_sum_to_one(self, engine) -> None:
        features = np.random.default_rng(1).uniform(0, 1, size=N_FEATURES_TOTAL).astype(np.float32)
        result = engine.predict(features)
        assert result["probabilities"].sum() == pytest.approx(1.0, abs=1e-5)

    def test_probabilities_non_negative(self, engine) -> None:
        features = np.random.default_rng(2).uniform(0, 1, size=N_FEATURES_TOTAL).astype(np.float32)
        result = engine.predict(features)
        assert np.all(result["probabilities"] >= 0)

    def test_ilmenite_fraction_in_range(self, engine) -> None:
        features = np.random.default_rng(3).uniform(0, 1, size=N_FEATURES_TOTAL).astype(np.float32)
        result = engine.predict(features)
        assert 0.0 <= result["ilmenite_fraction"] <= 1.0

    def test_class_index_valid(self, engine) -> None:
        features = np.random.default_rng(4).uniform(0, 1, size=N_FEATURES_TOTAL).astype(np.float32)
        result = engine.predict(features)
        assert 0 <= result["class_index"] < len(MINERAL_CLASSES)

    def test_batch_inference_2d(self, engine) -> None:
        """Passing a (B, 301) array should succeed."""
        rng = np.random.default_rng(10)
        batch = rng.uniform(0, 1, size=(4, N_FEATURES_TOTAL)).astype(np.float32)
        # predict is documented for single-sample convenience but accepts
        # batched input — the logits/ilm output will be batched and the
        # helper indexes [0], so we just verify no crash and shape.
        result = engine.predict(batch)
        assert result["probabilities"].shape == (5,)

    def test_wrong_feature_count_raises(self, engine) -> None:
        bad = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="expected 301 features"):
            engine.predict(bad)


@requires_ort
class TestInferenceEngineErrors:
    """Edge-case and error-path tests (no model file needed for some)."""

    def test_missing_model_raises_file_not_found(self, tmp_path: Path) -> None:
        from regoscan.inference import InferenceEngine

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            InferenceEngine(tmp_path / "nonexistent.onnx")

    def test_unsupported_ndim_raises(self) -> None:
        """4-D input should raise ValueError."""
        if not _model_exists:
            pytest.skip("model not available")
        from regoscan.inference import InferenceEngine

        engine = InferenceEngine(_MODEL_PATH)
        bad = np.zeros((1, 1, 1, N_FEATURES_TOTAL), dtype=np.float32)
        with pytest.raises(ValueError, match="unsupported feature ndim"):
            engine.predict(bad)
