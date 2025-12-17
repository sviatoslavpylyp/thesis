import json
import time
from typing import Optional

import joblib
import pandas as pd

from config import (
    MODEL_PATH,
    EVE_PATH,
    POLL_INTERVAL_SECONDS,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    DEFAULT_THRESHOLD,
)
from train_model import train_model


# =============================================================================
# Model Loading
# =============================================================================

def load_or_train_model():
    """
    Load the model if it exists, otherwise train and persist it.
    """
    if MODEL_PATH.exists():
        print(f"Loading existing model from: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print("Model not found. Training new model...")
    model = train_model()

    if not MODEL_PATH.exists():
        raise RuntimeError("Model training completed but model file was not created.")

    return model


# =============================================================================
# Parsing
# =============================================================================

def parse_eve_lines(lines: list[str]) -> pd.DataFrame:
    """
    Parse Suricata eve.json lines into a DataFrame.
    """
    records = []

    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        flow = event.get("flow", {})
        alert = event.get("alert", {})

        records.append(
            {
                "proto": event.get("proto", "UNKNOWN"),
                "service": event.get("app_proto", "-"),
                "state": alert.get("category", "GENERIC"),
                "spkts": flow.get("pkts_toserver", 0),
                "dpkts": flow.get("pkts_toclient", 0),
                "sbytes": flow.get("bytes_toserver", 0),
                "dbytes": flow.get("bytes_toclient", 0),
                "sttl": 64,
                "dttl": 64,
            }
        )

    return pd.DataFrame.from_records(records)


# =============================================================================
# Classification
# =============================================================================

def classify_alerts(
    model,
    df: pd.DataFrame,
    threshold: float,
) -> Optional[pd.DataFrame]:
    """
    Run inference on parsed alerts.
    """
    if df.empty:
        return None

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    probabilities = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["prob_attack"] = probabilities
    df["prediction"] = (probabilities > threshold).astype(int)

    return df


# =============================================================================
# Modes
# =============================================================================

def run_single_read(model, threshold: float) -> None:
    print("Running in SINGLE-READ mode")

    with EVE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    df = parse_eve_lines(lines)
    results = classify_alerts(model, df, threshold)

    if results is not None:
        print(results.head(15).to_string(index=False))


def run_continuous(model, threshold: float) -> None:
    print("Running in CONTINUOUS mode")
    print(f"Polling every {POLL_INTERVAL_SECONDS} seconds\n")

    file_offset = 0

    while True:
        try:
            with EVE_PATH.open("r", encoding="utf-8") as f:
                f.seek(file_offset)
                new_lines = f.readlines()
                file_offset = f.tell()
        except FileNotFoundError:
            print("eve.json not found, retrying...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if not new_lines:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        df = parse_eve_lines(new_lines)
        results = classify_alerts(model, df, threshold)

        if results is not None:
            attacks = (results["prediction"] == 1).sum()
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"New alerts: {len(results)}, "
                f"Attacks detected: {attacks}"
            )

        time.sleep(POLL_INTERVAL_SECONDS)


# =============================================================================
# Main
# =============================================================================

def main(
    mode: str = "single",
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    print("=== Suricata LightGBM AI Filter ===\n")

    model = load_or_train_model()

    if mode == "single":
        run_single_read(model, threshold)
    elif mode == "continuous":
        run_continuous(model, threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Default behavior (safe)
    main(mode="single")
    # main(mode="continuous")
