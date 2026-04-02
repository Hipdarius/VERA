"""Regoscan Streamlit dashboard.

Reads a Regoscan CSV from disk, lets the user pick a measurement, plots
the spectrum + LED + LIF channels, and runs inference using either the
PLSR baseline or the trained CNN if a run directory is selected.

This dashboard is **read-only** with respect to hardware. There is no
serial code anywhere — measurements always come from the canonical CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sure the in-repo `regoscan` package resolves when launched via
# `streamlit run apps/dashboard.py` from the project root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regoscan.datasets import to_bundle  # noqa: E402
from regoscan.io_csv import (  # noqa: E402
    SchemaError,
    extract_leds,
    extract_lif,
    extract_spectra,
    read_measurements_csv,
)
from regoscan.models.plsr import build_baseline_features, load_baseline  # noqa: E402
from regoscan.schema import (  # noqa: E402
    INDEX_TO_CLASS,
    LED_WAVELENGTHS_NM,
    MINERAL_CLASSES,
    WAVELENGTHS,
)


st.set_page_config(page_title="Regoscan", layout="wide")
st.title("Regoscan — VIS/NIR + 405 nm LIF probe")
st.caption(
    "Lunar regolith mineral classifier. Hardware not connected — load a "
    "canonical CSV from disk to explore measurements and run inference."
)


# ---------------------------------------------------------------------------
# Sidebar — data + model selection
# ---------------------------------------------------------------------------


st.sidebar.header("Data")
default_csv = ROOT / "data" / "synth_v1.csv"
csv_path = st.sidebar.text_input(
    "Measurement CSV",
    value=str(default_csv) if default_csv.exists() else "",
    help="Absolute path to a Regoscan-schema CSV.",
)

st.sidebar.header("Model (optional)")
default_run = ROOT / "runs" / "plsr"
run_dir_str = st.sidebar.text_input(
    "PLSR run directory",
    value=str(default_run) if (default_run / "model.pkl").exists() else "",
    help="Run directory containing model.pkl produced by `regoscan.train --model plsr`.",
)


@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return read_measurements_csv(path)


@st.cache_resource(show_spinner=False)
def _load_baseline(path: str):
    return load_baseline(path)


if not csv_path:
    st.info("Enter a CSV path in the sidebar to begin.")
    st.stop()

try:
    df = _load_csv(csv_path)
except (FileNotFoundError, SchemaError) as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df)} measurements")

# ---------------------------------------------------------------------------
# Sample/measurement picker
# ---------------------------------------------------------------------------

samples = sorted(df["sample_id"].unique().tolist())
sel_sample = st.sidebar.selectbox("sample_id", samples)
sub = df[df["sample_id"] == sel_sample].reset_index(drop=True)
st.sidebar.write(f"{len(sub)} measurements for this sample")

mid_options = sub["measurement_id"].tolist()
sel_mid = st.sidebar.selectbox("measurement_id", mid_options)
row = sub[sub["measurement_id"] == sel_mid].iloc[0]

# ---------------------------------------------------------------------------
# Top metadata strip
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("True class", row["mineral_class"])
c2.metric("Ilmenite frac (truth)", f"{row['ilmenite_fraction']:.3f}")
c3.metric("Integration (ms)", int(row["integration_time_ms"]))
c4.metric("Packing", row["packing_density"])

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

spec = extract_spectra(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0]
leds = extract_leds(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0]
lif = float(extract_lif(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0])

st.subheader("Spectrometer (340–850 nm, 288 channels)")
spec_df = pd.DataFrame({"wavelength_nm": WAVELENGTHS, "reflectance": spec})
st.line_chart(spec_df.set_index("wavelength_nm"))

c5, c6 = st.columns(2)
with c5:
    st.subheader("LED narrowband")
    led_df = pd.DataFrame(
        {"led_nm": list(LED_WAVELENGTHS_NM), "reflectance": leds}
    ).set_index("led_nm")
    st.bar_chart(led_df)
with c6:
    st.subheader("LIF (450 nm LP, 405 nm ex)")
    st.metric("photodiode", f"{lif:.3f}")
    st.caption("Quenched by ilmenite; bright on plagioclase.")

# ---------------------------------------------------------------------------
# Inference (PLSR baseline only — CNN inference would require torch import
# at module top, which slows the dashboard cold-start. Add it if needed.)
# ---------------------------------------------------------------------------

st.subheader("Inference")
if not run_dir_str:
    st.info("Pick a PLSR run directory in the sidebar to run inference.")
else:
    model_path = Path(run_dir_str) / "model.pkl"
    if not model_path.exists():
        st.warning(f"No model.pkl at {model_path}")
    else:
        try:
            bb = _load_baseline(str(model_path))
            bundle = to_bundle(sub[sub["measurement_id"] == sel_mid])
            X = build_baseline_features(bundle)
            pred_cls, pred_ilm = bb.predict(X)
            proba = bb.predict_proba(X)[0]
            top_class = INDEX_TO_CLASS[int(pred_cls[0])]
            top_ilm = float(pred_ilm[0])
            cc1, cc2 = st.columns(2)
            cc1.metric("Predicted class", top_class)
            cc2.metric("Predicted ilmenite frac", f"{top_ilm:.3f}")
            proba_df = pd.DataFrame(
                {"class": list(MINERAL_CLASSES), "probability": proba}
            ).set_index("class")
            st.bar_chart(proba_df)
        except Exception as e:  # pragma: no cover - dashboard UX
            st.error(f"Inference failed: {e}")

st.caption(
    "Schema is locked in `regoscan.schema`. When real hardware comes online "
    "the CSV format won't change — measurements just start arriving from the "
    "device instead of the synthetic generator."
)
