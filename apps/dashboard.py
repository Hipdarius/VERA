"""
Regoscan Dashboard (Streamlit).

Lets us explore the canonical CSV measurements, plot the spectra, and run
inference using the trained PLSR baseline.

Note: Serial/Hardware code isn't here yet. Measurements are read from 
static files for now.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Path resolution for the local 'regoscan' package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regoscan.datasets import to_bundle 
from regoscan.io_csv import (
    SchemaError,
    extract_leds,
    extract_lif,
    extract_spectra,
    read_measurements_csv,
)
from regoscan.models.plsr import build_baseline_features, load_baseline 
from regoscan.schema import (
    INDEX_TO_CLASS,
    LED_WAVELENGTHS_NM,
    MINERAL_CLASSES,
    WAVELENGTHS,
)

st.set_page_config(page_title="Regoscan", layout="wide")
st.title("Regoscan — Optical Probe Dashboard")
st.caption("Mineral classification and spectral explorer. Load a CSV to begin.")

# Sidebar - data loading
st.sidebar.header("Data Loading")
default_csv = ROOT / "data" / "synth_v1.csv"
csv_path = st.sidebar.text_input(
    "Path to CSV",
    value=str(default_csv) if default_csv.exists() else "",
)

st.sidebar.header("Model Selection")
default_run = ROOT / "runs" / "plsr"
run_dir_str = st.sidebar.text_input(
    "PLSR Run Dir",
    value=str(default_run) if (default_run / "model.pkl").exists() else "",
)

@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return read_measurements_csv(path)

@st.cache_resource(show_spinner=False)
def _load_baseline(path: str):
    return load_baseline(path)

if not csv_path:
    st.info("Please provide a CSV path in the sidebar.")
    st.stop()

try:
    df = _load_csv(csv_path)
    st.sidebar.success(f"Found {len(df)} measurements.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sample & measurement selection
samples = sorted(df["sample_id"].unique().tolist())
sel_sample = st.sidebar.selectbox("Select Sample", samples)
sub = df[df["sample_id"] == sel_sample].reset_index(drop=True)

mid_options = sub["measurement_id"].tolist()
sel_mid = st.sidebar.selectbox("Select Measurement", mid_options)
row = sub[sub["measurement_id"] == sel_mid].iloc[0]

# Display metadata
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ground Truth", row["mineral_class"])
c2.metric("Ilm Frac (Truth)", f"{row['ilmenite_fraction']:.3f}")
c3.metric("Int. Time (ms)", int(row["integration_time_ms"]))
c4.metric("Packing", row["packing_density"])

# Spectral plots
spec = extract_spectra(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0]
leds = extract_leds(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0]
lif = float(extract_lif(sub.iloc[[sub.index[sub["measurement_id"] == sel_mid][0]]])[0])

st.write("---")
st.subheader("Spectrometer Data (340–850 nm)")
spec_df = pd.DataFrame({"wavelength_nm": WAVELENGTHS, "reflectance": spec})
st.line_chart(spec_df.set_index("wavelength_nm"))

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("LED Reflection")
    led_df = pd.DataFrame(
        {"led_nm": list(LED_WAVELENGTHS_NM), "reflectance": leds}
    ).set_index("led_nm")
    st.bar_chart(led_df)
with col_r:
    st.subheader("LIF Signal (450nm LP)")
    st.metric("PD Value", f"{lif:.3f}")
    st.caption("Lower value indicates higher ilmenite quenching.")

# Inference section
st.write("---")
st.subheader("Real-Time Inference")
if not run_dir_str:
    st.info("Pick a PLSR run directory in the sidebar to see predictions.")
else:
    model_path = Path(run_dir_str) / "model.pkl"
    if model_path.exists():
        try:
            bb = _load_baseline(str(model_path))
            bundle = to_bundle(sub[sub["measurement_id"] == sel_mid])
            X = build_baseline_features(bundle)
            pred_cls, pred_ilm = bb.predict(X)
            proba = bb.predict_proba(X)[0]
            
            top_class = INDEX_TO_CLASS[int(pred_cls[0])]
            top_ilm = float(pred_ilm[0])
            
            inf1, inf2 = st.columns(2)
            inf1.metric("Predicted Class", top_class)
            inf2.metric("Predicted Ilm Frac", f"{top_ilm:.3f}")
            
            proba_df = pd.DataFrame(
                {"class": list(MINERAL_CLASSES), "probability": proba}
            ).set_index("class")
            st.bar_chart(proba_df)
        except Exception as e:
            st.error(f"Inference failed: {e}")
    else:
        st.warning("No model.pkl found in the selected directory.")
