"""
03_evaluate.py — Load fitted models and compute all metrics, plots, statistics.

Loads models from models/ and data from data_cache/, reuses the exact same
train/test split saved by 02_tune.py (no seed is used, so the split itself
is persisted to disk rather than reconstructed), evaluates on the test set,
and saves all results + plots to output/.

Usage:
    python3 -u 03_evaluate.py
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from config import CONFIG

IS_TTY = sys.stdout.isatty()

MODEL_NAMES = ["RandomForest", "XGBoost", "DecisionTree"]
TASKS       = ["binary", "multiclass"]
METRIC_COLS = ["accuracy", "precision", "recall", "f1",
               "roc_auc", "train_time", "inference_time"]

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
    fname = f"{task}_{model_name}.pkl"
    return os.path.join(CONFIG["models_dir"], fname)


def split_indices_path(task: str) -> str:
    return os.path.join(CONFIG["data_cache_dir"], f"{task}_split_indices.npz")


def get_split(X, y, task: str):
    """Reuses the exact train/test split saved by 02_tune.py."""
    idx_path = split_indices_path(task)
    if not os.path.exists(idx_path):
        log(f"ERROR: missing split indices {idx_path} — run 02_tune.py first")
        sys.exit(1)
    npz = np.load(idx_path)
    train_idx, test_idx = npz["train_idx"], npz["test_idx"]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, task: str) -> dict:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    y_proba = model.predict_proba(X_test)

    if task == "binary":
        try:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except ValueError:
            roc_auc = float("nan")
    else:
        try:
            roc_auc = roc_auc_score(y_test, y_proba,
                                    multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = float("nan")

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc":          roc_auc,
        "inference_time":   inference_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred":           y_pred,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def plot_feature_importance(importances, feature_names, title, path, top_n=20):
    top_n = min(top_n, len(feature_names))
    idx   = np.argsort(importances)[::-1][:top_n]
    feats = [feature_names[i] for i in idx][::-1]
    vals  = importances[idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), vals, color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"    Saved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    separator("CICIDS2017 — EVALUATION SCRIPT")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Load data ---
    cache = CONFIG["data_cache_dir"]
    log(f"\n  Loading cached arrays from {cache}/ …")

    X = np.load(os.path.join(cache, "X.npy"))
    datasets = {
        "binary":      np.load(os.path.join(cache, "y_binary.npy")),
        "multiclass":  np.load(os.path.join(cache, "y_multiclass.npy")),
    }
    with open(os.path.join(cache, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(cache, "class_labels_binary.json")) as f:
        class_labels = {"binary": json.load(f)}
    with open(os.path.join(cache, "class_labels_multiclass.json")) as f:
        class_labels["multiclass"] = json.load(f)

    # Load best_params for train_time reference
    params_file = os.path.join(CONFIG["models_dir"], "best_params.json")
    best_params_all = {}
    if os.path.exists(params_file):
        with open(params_file) as f:
            best_params_all = json.load(f)

    log(f"  X: {X.shape}  features: {len(feature_names)}")

    all_results  = []
    last_artifacts = {}   # {(model_name, task): {cm, model, class_labels, feature_names}}

    total_exps = len(TASKS) * len(MODEL_NAMES)
    done = 0

    for task in TASKS:
        separator(f"EVALUATING: {task.upper()}")
        y = datasets[task]
        labels = class_labels[task]

        _, X_test, _, y_test = get_split(X, y, task)

        for model_name in MODEL_NAMES:
            done += 1
            log(f"\n  [{done}/{total_exps}] {task} | {model_name}")

            mpath = model_path(task, model_name)
            if not os.path.exists(mpath):
                log(f"    MISSING model file: {mpath}  — skipping")
                continue

            model = joblib.load(mpath)
            log(f"    Model loaded from: {mpath}")

            # Retrieve fit_time from best_params JSON (recorded during training)
            key = f"{task}|{model_name}"
            train_time = best_params_all.get(key, {}).get("fit_time", float("nan"))

            metrics = evaluate_model(model, X_test, y_test, task)

            row = {
                "task":           task,
                "model":          model_name,
                "accuracy":       metrics["accuracy"],
                "precision":      metrics["precision"],
                "recall":         metrics["recall"],
                "f1":             metrics["f1"],
                "roc_auc":        metrics["roc_auc"],
                "train_time":     train_time,
                "inference_time": metrics["inference_time"],
            }
            all_results.append(row)

            log(
                f"    Acc={metrics['accuracy']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"AUC={metrics['roc_auc']:.4f}  "
                f"inf={metrics['inference_time']:.3f}s"
            )

            last_artifacts[(model_name, task)] = {
                "cm":            metrics["confusion_matrix"],
                "model":         model,
                "class_labels":  labels,
                "feature_names": feature_names,
            }

    if not all_results:
        log("\nERROR: No results collected. Did 02_tune.py complete successfully?")
        sys.exit(1)

    # --- Save raw results ---
    separator("SAVING RESULTS")
    raw_df   = pd.DataFrame(all_results)
    raw_path = os.path.join(CONFIG["output_dir"], "all_results_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    log(f"  Raw results ({len(raw_df)} rows) → {raw_path}")

    # --- Summary table ---
    summary_df = raw_df[["task", "model"] + METRIC_COLS].round(6)
    sum_path   = os.path.join(CONFIG["output_dir"], "results_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    log(f"  Summary table             → {sum_path}")

    # --- Print summary ---
    separator("RESULTS SUMMARY")
    log(summary_df.to_string(index=False))

    # --- Plots ---
    separator("PLOTS")
    for (model_name, task), art in last_artifacts.items():
        # Confusion matrix
        cm_path = os.path.join(CONFIG["output_dir"],
                               f"{model_name}_{task}_confusion_matrix.png")
        plot_confusion_matrix(
            art["cm"], art["class_labels"],
            title=f"{model_name} ({task}) — Confusion Matrix",
            path=cm_path,
        )

        # Feature importance — RF and XGBoost only
        if model_name in ("RandomForest", "XGBoost"):
            fi_path = os.path.join(CONFIG["output_dir"],
                                   f"{model_name}_{task}_feature_importance.png")
            plot_feature_importance(
                art["model"].feature_importances_,
                art["feature_names"],
                title=f"{model_name} ({task}) — Top-20 Feature Importances",
                path=fi_path,
            )

    elapsed = time.perf_counter() - t_start
    separator("EVALUATION COMPLETE")
    log(f"  Output folder : {os.path.abspath(CONFIG['output_dir'])}/")
    log(f"  Total time    : {elapsed:.1f}s")
    log(f"  Finished      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
