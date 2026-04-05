#!/usr/bin/env python3
"""
Regoscan Hardware Bridge. 

Listens for incoming JSON measurements from the ESP32 (or mock),
validates against schema.py, runs real-time ONNX inference, 
and logs results to data/real_v1.csv.
"""

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure the in-repo src/ is importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regoscan.schema import (
    ALL_COLUMNS,
    N_SPEC,
    N_LED,
    SCHEMA_VERSION,
    Measurement,
    MINERAL_CLASSES,
)
from regoscan.inference import InferenceEngine

def _resolve_model_path() -> Path | None:
    """Finds the trained ONNX model in common run directories."""
    candidates = [
        ROOT / "runs" / "cnn_v2" / "model.onnx",
        ROOT / "runs" / "cnn_run" / "model.onnx",
        ROOT / "web" / "api" / "model.onnx"
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def main():
    parser = argparse.ArgumentParser(description="Regoscan Serial Bridge")
    parser.add_argument("--sample-id", default="UNKNOWN_SAMPLE", help="Sample ID to log")
    parser.add_argument("--packing", default="medium", choices=["loose", "medium", "packed"], help="Packing density")
    parser.add_argument("--temp", type=float, default=22.0, help="Ambient temperature in Celsius")
    parser.add_argument("--out", default="data/real_v1.csv", help="CSV path to append to")
    parser.add_argument("--model", help="Override path to ONNX model")
    
    args = parser.parse_args()

    print(f"Regoscan Bridge Started. Schema {SCHEMA_VERSION}", file=sys.stderr)
    
    # 1. Load Model
    model_path = Path(args.model) if args.model else _resolve_model_path()
    if model_path is None or not model_path.exists():
        print(f"[err] ONNX model not found. Checked: {model_path}", file=sys.stderr)
        print("Please train a model or export ONNX to web/api/model.onnx first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"[ok] Loading inference engine: {model_path.name}", file=sys.stderr)
    engine = InferenceEngine(model_path)
    
    # 2. Setup CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    
    # 3. Main Loop (stdin)
    print("Bridge Listening... (Pipe mock_esp32.py output into this)", file=sys.stderr)
    
    try:
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            if new_file:
                writer.writeheader()
            
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    raw = json.loads(line)
                    if raw.get("type") != "measurement":
                        continue
                    
                    # A. Data Processing (Dark Subtraction)
                    # spec = broadband - dark
                    broadband = np.asarray(raw["broadband_spectrum"], dtype=np.float32)
                    dark = np.asarray(raw["dark_spectrum"], dtype=np.float32)
                    spec = broadband - dark
                    
                    led = np.asarray(raw["narrowband_leds"], dtype=np.float32)
                    lif = float(raw["lif_value"])
                    
                    # B. Feature Extraction for Inference
                    features = np.concatenate([spec, led, np.asarray([lif], dtype=np.float32)])
                    
                    # C. Run Prediction
                    pred = engine.predict(features)
                    klass = MINERAL_CLASSES[pred["class_index"]]
                    ilm = pred["ilmenite_fraction"]
                    
                    print(f"===> Prediction: {klass} ({pred['probabilities'][pred['class_index']]:.2f} conf) | Ilmenite: {ilm:.1%}")
                    
                    # D. Validation & CSV Export
                    # Use schema.py Measurement to ensure integrity
                    measurement = Measurement(
                        sample_id=args.sample_id,
                        measurement_id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        mineral_class=klass, # Log predicted class as the ground truth label for this live session
                        ilmenite_fraction=ilm,
                        integration_time_ms=int(raw["integration_time_ms"]),
                        ambient_temp_c=args.temp,
                        packing_density=args.packing,
                        spec=[float(x) for x in spec],
                        led=[float(x) for x in led],
                        lif_450lp=lif
                    )
                    
                    writer.writerow(measurement.to_row())
                    f.flush() # Ensure it's written immediately for real-time monitoring
                    
                except json.JSONDecodeError:
                    print("[warn] Received malformed JSON, skipping.", file=sys.stderr)
                except Exception as e:
                    print(f"[err] Processing error: {e}", file=sys.stderr)
                    
    except KeyboardInterrupt:
        print("\nBridge terminated by user.", file=sys.stderr)

if __name__ == "__main__":
    main()
