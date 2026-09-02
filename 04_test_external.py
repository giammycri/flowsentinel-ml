"""
04_test_external.py — Test the fitted models on an external flow CSV.

Loads a CICFlowMeter V4 CSV (e.g. flows extracted from your own GNS3 pcap or
from a real network capture), renames its columns to the CICIDS2017 naming
the models were trained on, selects the features in the exact training
order, and runs every saved model (binary or multiclass, your choice) against
it — reporting the same metrics as 03_evaluate.py (accuracy, precision,
recall, F1, ROC-AUC, confusion matrix).

Unlike 03_evaluate.py (which reuses the saved CICIDS2017 test split from
data_cache/), this script scores a completely independent capture.

Ground truth
------------
If the CSV has a real per-row 'Label' column (not CICFlowMeter's default
"NeedManualLabel" placeholder), it is used directly — harmonized the same
way 01_preprocess.py harmonizes the training labels (Heartbleed rows
dropped, "X - Attempted" merged into BENIGN). This gives real precision/
recall/F1/ROC-AUC, not just a one-sided detection rate.

If the CSV has no usable Label column (e.g. a pure attack or pure benign
capture with "NeedManualLabel" everywhere), fall back to --label to assign
a single ground-truth class to every row.

Usage:
    python3 -u 04_test_external.py
    python3 -u 04_test_external.py --task binary
    python3 -u 04_test_external.py --task multiclass --csv flussotest/all_traffic_labeled.csv
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from config import CONFIG, COLUMN_MAP, DROP_LABEL_PREFIX

IS_TTY = sys.stdout.isatty()

MODEL_NAMES = ["RandomForest", "XGBoost", "DecisionTree"]
TASKS       = ["binary", "multiclass"]
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "inference_time"]

# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def separator(title: str = "") -> None:
    line = "=" * 64
    log(f"\n{line}")
    if title:
        log(f"  {title}")
        log(line)


def model_path(task: str, model_name: str) -> str:
    return os.path.join(CONFIG["models_dir"], f"{task}_{model_name}.pkl")


def prompt_task(task_arg: str) -> str:
    """--task wins outright; otherwise ask on the terminal, defaulting to binary
    when not interactive (e.g. output redirected to a log file)."""
    if task_arg:
        return task_arg
    if not IS_TTY:
        log("  No --task given and not running interactively — defaulting to 'binary'.")
        return "binary"
    while True:
        choice = input("  Classificazione binaria o multiclasse? [binary/multiclass]: ").strip().lower()
        if choice in ("binary", "b"):
            return "binary"
        if choice in ("multiclass", "m"):
            return "multiclass"
        print("  Rispondi con 'binary' o 'multiclass'.")


# ---------------------------------------------------------------------------
# Loading + ground truth
# ---------------------------------------------------------------------------

def harmonize_label_column(label_series: pd.Series):
    """Same relabelling 01_preprocess.py applies to the training labels:
    drop Heartbleed rows, merge 'X - Attempted' into BENIGN.

    Returns (harmonized_labels, keep_mask).
    """
    lbl = label_series.astype(str).str.strip()
    is_hb = lbl.str.startswith(DROP_LABEL_PREFIX)
    is_att = lbl.str.endswith(" - Attempted")
    lbl = lbl.mask(is_att, "BENIGN")
    return lbl, ~is_hb


def label_column_is_usable(label_series) -> bool:
    if label_series is None:
        return False
    uniq = set(label_series.dropna().astype(str).str.strip().unique())
    uniq.discard("NeedManualLabel")
    uniq.discard("")
    return len(uniq) > 0


def load_external(path: str, feature_names: list):
    """Read the CSV, rename columns, select features, clean NaN/inf rows.

    Returns (X, label_series_or_None, n_raw, n_bad).
    """
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

    cols = list(feature_names)
    has_label_col = "Label" in df.columns
    if has_label_col:
        cols = cols + ["Label"]

    df = df[cols].copy()
    log(f"  Selected {len(feature_names)} features in training order"
        + ("  (+ 'Label' column found in CSV)" if has_label_col else ""))

    feat = df[feature_names].replace([np.inf, -np.inf], np.nan)
    bad = feat.isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        log(f"\n  WARNING: dropping {n_bad:,} row(s) with NaN/inf "
            f"({100.0 * n_bad / n_raw:.2f}% of the capture)")
        df = df[~bad]
        feat = feat[~bad]

    X = feat.astype(np.float32).values
    label_series = df["Label"] if has_label_col else None
    log(f"\n  Usable flows: {len(X):,} / {n_raw:,}")

    return X, label_series, n_raw, n_bad


def build_ground_truth(task: str, label_series, class_labels: list, fallback_label: str):
    """Returns (y_true, keep_mask, source_description)."""
    if label_column_is_usable(label_series):
        lbl, keep = harmonize_label_column(label_series)

        if task == "binary":
            y = (lbl != "BENIGN").astype(np.int32).values
            return y, keep.values, "per-row 'Label' column in the CSV"

        # multiclass: map to the exact classes the models were trained on
        mapping = {c: i for i, c in enumerate(class_labels)}
        unknown = ~lbl.isin(class_labels)
        n_unknown = int((unknown & keep).sum())
        if n_unknown:
            unk_values = sorted(lbl[unknown & keep].unique())
            log(f"\n  WARNING: dropping {n_unknown:,} row(s) whose label isn't one of "
                f"the {len(class_labels)} trained classes:")
            for v in unk_values:
                log(f"    - {v}")
        keep = keep & ~unknown
        y = lbl.map(mapping).fillna(-1).astype(np.int32).values
        return y, keep.values, "per-row 'Label' column in the CSV"

    # No usable per-row Label column: fall back to a single global class
    if fallback_label is None:
        log("\n  ERROR: the CSV has no usable 'Label' column "
            "(missing, or every row is 'NeedManualLabel'),")
        log("  and no --label fallback was given. Pass --label to assign one "
            "ground-truth class to the whole file,")
        log("  e.g. --label ATTACK, or --label BENIGN, "
            f"or (multiclass) one of: {class_labels}")
        sys.exit(1)

    if task == "binary":
        valid = {"ATTACK", "BENIGN"}
        if fallback_label.upper() not in valid:
            log(f"\n  ERROR: --label must be one of {sorted(valid)} for --task binary.")
            sys.exit(1)
        y_value = 1 if fallback_label.upper() == "ATTACK" else 0
        n = len(label_series) if label_series is not None else None
        return None, None, ("fallback", y_value)  # handled by caller (needs len(X))

    else:
        matches = [c for c in class_labels if c.lower() == fallback_label.lower()]
        if not matches:
            log(f"\n  ERROR: --label '{fallback_label}' is not one of the trained "
                f"classes: {class_labels}")
            sys.exit(1)
        y_value = class_labels.index(matches[0])
        return None, None, ("fallback", y_value)


# ---------------------------------------------------------------------------
# Evaluation (mirrors 03_evaluate.py's evaluate_model)
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y_true, task: str, n_classes: int) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - t0

    y_proba = model.predict_proba(X)

    try:
        if task == "binary":
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                    average="weighted", labels=list(range(n_classes)))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":          roc_auc,
        "inference_time":   inference_time,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))),
        "y_pred":           y_pred,
    }


def plot_confusion_matrix(cm, labels, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"    Saved: {path}")


def run(task: str, X, y_true, class_labels: list, out_dir: str, csv_stem: str) -> pd.DataFrame:
    separator(f"STEP 2 — Scoring models  [{task}]")

    n_classes = len(class_labels)
    rows = []
    last_cm = {}

    for model_name in MODEL_NAMES:
        path = model_path(task, model_name)
        if not os.path.exists(path):
            log(f"  SKIP  {model_name:<14s}  (not found)")
            continue

        model = joblib.load(path)
        metrics = evaluate_model(model, X, y_true, task, n_classes)

        rows.append({
            "task": task, "model": model_name,
            "n_samples": len(X),
            "accuracy":       metrics["accuracy"],
            "precision":      metrics["precision"],
            "recall":         metrics["recall"],
            "f1":             metrics["f1"],
            "roc_auc":        metrics["roc_auc"],
            "inference_time": metrics["inference_time"],
        })

        log(f"  {model_name:<14s}  "
            f"Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
            f"AUC={metrics['roc_auc']:.4f}  ({metrics['inference_time']:.2f}s)")

        last_cm[model_name] = metrics["confusion_matrix"]

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    separator("STEP 3 — Summary")
    summary = results.set_index("model")[METRIC_COLS].reindex(MODEL_NAMES).dropna(how="all")

    header = f"  {'Model':<16s}" + "".join(f"{c:>16s}" for c in METRIC_COLS)
    log(f"\n{header}")
    log(f"  {'-' * (16 + 16 * len(METRIC_COLS))}")
    for model_name in summary.index:
        cells = "".join(f"{summary.loc[model_name, c]:16.4f}" for c in METRIC_COLS)
        log(f"  {model_name:<16s}{cells}")

    separator("CONFUSION MATRICES")
    for model_name, cm in last_cm.items():
        log(f"\n  {model_name}:")
        cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in class_labels],
                                  columns=[f"pred:{l}" for l in class_labels])
        log(cm_df.to_string())

        cm_path = os.path.join(out_dir, f"external_{csv_stem}_{task}_{model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, class_labels,
                              f"{model_name} ({task}) — external capture", cm_path)

    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the fitted models on an external flow CSV.")
    parser.add_argument("--csv", default="flussotest/all_traffic_labeled.csv",
                        help="path to the CICFlowMeter CSV")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="binary or multiclass models (asked interactively if omitted)")
    parser.add_argument("--label", default=None,
                        help="fallback ground-truth class for the whole file, used only "
                             "if the CSV has no usable per-row 'Label' column "
                             "(ATTACK/BENIGN for --task binary, a class name for multiclass)")
    parser.add_argument("--out", default=None,
                        help="where to write per-model results "
                             "(default: output/external_<csv name>_<task>.csv)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    separator("EXTERNAL FLOW TEST")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(args.csv):
        log(f"\n  ERROR: {args.csv} not found.")
        sys.exit(1)

    task = prompt_task(args.task)
    log(f"  Task : {task}")

    cache_dir = CONFIG["data_cache_dir"]
    with open(os.path.join(cache_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    label_file = "class_labels_binary.json" if task == "binary" else "class_labels_multiclass.json"
    with open(os.path.join(cache_dir, label_file)) as f:
        class_labels = json.load(f)

    X, label_series, n_raw, n_bad = load_external(args.csv, feature_names)

    if len(X) == 0:
        log("\n  ERROR: no usable rows left after cleaning.")
        sys.exit(1)

    y_true, keep_mask, source = build_ground_truth(task, label_series, class_labels, args.label)

    if keep_mask is None:
        # fallback path: single class for the whole (already-cleaned) file
        _, y_value = source
        y_true = np.full(len(X), y_value, dtype=np.int32)
        log(f"\n  Ground truth: fallback — every flow labelled "
            f"'{class_labels[y_value]}' via --label")
    else:
        X = X[keep_mask]
        y_true = y_true[keep_mask]
        log(f"\n  Ground truth: {source}")
        log(f"  Usable flows after label harmonization: {len(X):,}")
        unique, counts = np.unique(y_true, return_counts=True)
        for u, c in zip(unique, counts):
            log(f"    [{u}] {class_labels[u]:<20s} {c:>7,}  ({100.0 * c / len(y_true):.1f}%)")

    if len(X) == 0:
        log("\n  ERROR: no usable rows left after ground-truth alignment.")
        sys.exit(1)

    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_dir = CONFIG["output_dir"]
    results = run(task, X, y_true, class_labels, out_dir, csv_stem)

    if results.empty:
        log(f"\n  ERROR: no {task} models found in models/ — run 02_tune.py first.")
        sys.exit(1)

    out_path = args.out or os.path.join(out_dir, f"external_{csv_stem}_{task}.csv")
    results.to_csv(out_path, index=False)

    separator("TEST COMPLETE")
    log(f"  Results    : {out_path}")
    log(f"  Total time : {time.perf_counter() - t_start:.1f}s")
    log(f"  Finished   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
