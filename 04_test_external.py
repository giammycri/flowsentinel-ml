"""
04_test_external.py — Test the fitted binary models on an external flow CSV.

Loads a CICFlowMeter V4 CSV (e.g. flows extracted from your own GNS3 pcap),
renames its columns to the CICIDS2017 naming the models were trained on,
selects the 40 features in the exact training order, and runs every saved
binary model against it.

Unlike 03_evaluate.py (which reconstructs the CICIDS2017 test split from
data_cache/), this script scores a completely independent capture.

Ground truth is supplied by --label, not by the CSV's own Label column:
CICFlowMeter writes "NeedManualLabel" and cannot know what is an attack.

Usage:
    python3 -u 04_test_external.py
    python3 -u 04_test_external.py --csv flussotest/testslowloris.csv --label ATTACK
"""

import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from config import CONFIG, COLUMN_MAP

IS_TTY = sys.stdout.isatty()

MODEL_NAMES = ["RandomForest", "XGBoost", "DecisionTree"]

# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def separator(title: str = "") -> None:
    line = "=" * 64
    log(f"\n{line}")
    if title:
        log(f"  {title}")
        log(line)


def model_path(seed: int, model_name: str) -> str:
    return os.path.join(CONFIG["models_dir"], f"binary_seed{seed}_{model_name}.pkl")


# ---------------------------------------------------------------------------

def load_external(path: str, feature_names: list):
    """Read the CSV, rename columns, and return the feature matrix + row report."""
    separator("STEP 1 — Loading external flows")
    log(f"  File : {path}")

    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)
    log(f"  Shape: {df.shape}")

    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        log(f"\n  ERROR: {len(missing)} required feature(s) not found in the CSV:")
        for m in missing:
            log(f"    - {m}")
        log("\n  The CSV was likely produced by a different flow exporter.")
        log("  Extend COLUMN_MAP in this file to cover them.")
        sys.exit(1)

    log(f"  Renamed {len(COLUMN_MAP)} column(s) to CICIDS2017 naming")

    # Keep the 40 training features, in the exact training order
    X = df[feature_names].copy()
    log(f"  Selected {len(feature_names)} features in training order")

    # Same cleaning the training data went through: inf → NaN, drop those rows.
    # Rows are reported, not silently dropped: they change the denominator.
    X = X.replace([np.inf, -np.inf], np.nan)
    bad = X.isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        log(f"\n  WARNING: dropping {n_bad:,} row(s) with NaN/inf "
            f"({100.0 * n_bad / n_raw:.2f}% of the capture)")
        X = X[~bad]

    X = X.astype(np.float32)
    log(f"\n  Usable flows: {len(X):,} / {n_raw:,}")

    return X.values, n_raw, n_bad


def evaluate(X: np.ndarray, y_true: np.ndarray, class_labels: list):
    """Run every binary model over X and collect per-model results."""
    separator("STEP 2 — Scoring models")

    rows = []
    for model_name in MODEL_NAMES:
        for seed in CONFIG["seeds"]:
            path = model_path(seed, model_name)
            if not os.path.exists(path):
                log(f"  SKIP  {model_name:<14s} seed {seed}  (not found)")
                continue

            model = joblib.load(path)

            t0 = time.perf_counter()
            y_pred = model.predict(X)
            inference_time = time.perf_counter() - t0

            n_attack = int((y_pred == 1).sum())
            n_benign = int((y_pred == 0).sum())
            detection_rate = 100.0 * (y_pred == y_true).mean()

            rows.append({
                "model":          model_name,
                "seed":           seed,
                "pred_attack":    n_attack,
                "pred_benign":    n_benign,
                "detection_rate": detection_rate,
                "inference_time": inference_time,
            })

            log(f"  {model_name:<14s} seed {seed}  "
                f"ATTACK {n_attack:>6,} | BENIGN {n_benign:>6,}  "
                f"→ detection {detection_rate:6.2f}%  ({inference_time:.2f}s)")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str, n_flows: int) -> pd.DataFrame:
    separator("STEP 3 — Summary across seeds")

    summary = (df.groupby("model")["detection_rate"]
                 .agg(["mean", "std", "min", "max"])
                 .reindex(MODEL_NAMES)
                 .dropna(how="all"))

    log(f"\n  Ground truth: all {n_flows:,} flows labelled {label}")
    log(f"  Metric      : detection rate = % of flows predicted {label}\n")
    log(f"  {'Model':<16s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    log(f"  {'-' * 52}")
    for model_name, r in summary.iterrows():
        log(f"  {model_name:<16s} {r['mean']:7.2f}% {r['std']:7.2f}% "
            f"{r['min']:7.2f}% {r['max']:7.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test the fitted binary models on an external flow CSV.")
    parser.add_argument("--csv", default="flussotest/fritto_misto.csv",
                        help="path to the CICFlowMeter CSV")
    parser.add_argument("--label", default="ATTACK", choices=["ATTACK", "BENIGN"],
                        help="ground-truth label for every flow in the CSV")
    parser.add_argument("--out", default=None,
                        help="where to write per-model results "
                             "(default: output/external_<csv name>.csv)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    separator("EXTERNAL FLOW TEST — binary models")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(args.csv):
        log(f"\n  ERROR: {args.csv} not found.")
        sys.exit(1)

    cache_dir = CONFIG["data_cache_dir"]
    with open(os.path.join(cache_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(cache_dir, "class_labels_binary.json")) as f:
        class_labels = json.load(f)

    X, n_raw, n_bad = load_external(args.csv, feature_names)

    if len(X) == 0:
        log("\n  ERROR: no usable rows left after cleaning.")
        sys.exit(1)

    # Single-class ground truth: BENIGN = 0, ATTACK = 1 (same as 01_preprocess.py)
    y_value = 1 if args.label == "ATTACK" else 0
    y_true  = np.full(len(X), y_value, dtype=np.int32)

    results = evaluate(X, y_true, class_labels)
    if results.empty:
        log("\n  ERROR: no models found in models/ — run 02_tune.py first.")
        sys.exit(1)

    summarize(results, args.label, len(X))

    out_path = args.out or os.path.join(
        CONFIG["output_dir"],
        f"external_{os.path.splitext(os.path.basename(args.csv))[0]}.csv")
    results.to_csv(out_path, index=False)

    separator("NOTE ON INTERPRETATION")
    log(f"  Every flow in this capture is labelled {args.label}, so only the")
    log(f"  {args.label} class is exercised. This measures detection rate")
    log("  (recall) and nothing else — precision, F1 and ROC-AUC are undefined")
    log("  without both classes present, and are deliberately not reported.")
    log("  Capture benign traffic and merge it in to measure false positives.")

    separator("TEST COMPLETE")
    log(f"  Results    : {out_path}")
    log(f"  Total time : {time.perf_counter() - t_start:.1f}s")
    log(f"  Finished   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
