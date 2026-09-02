"""
02_tune.py — Model fitting with fixed hyperparameters (no search, no CV, no seed).

Loads preprocessed data from data_cache/, fits every (task, model) combination
once with the fixed hyperparameters from config.py, and saves the fitted
models to models/, overwriting any existing ones — every run retrains from
scratch on the current dataset.

Usage:
    python3 -u 02_tune.py
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import CONFIG, FIXED_HYPERPARAMS

IS_TTY = sys.stdout.isatty()

# ---------------------------------------------------------------------------
# Helpers
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


def params_path() -> str:
    return os.path.join(CONFIG["models_dir"], "best_params.json")


def load_used_params() -> dict:
    p = params_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_used_params(all_params: dict) -> None:
    with open(params_path(), "w") as f:
        json.dump(all_params, f, indent=2)

# ---------------------------------------------------------------------------
# Model definitions — fixed hyperparameters, no random_state
# ---------------------------------------------------------------------------

def get_models(task: str) -> dict:
    params = FIXED_HYPERPARAMS[task]
    return {
        "RandomForest": RandomForestClassifier(
            n_jobs=-1, **params["RandomForest"],
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", verbosity=0, device="cuda",
            **params["XGBoost"],
        ),
        "DecisionTree": DecisionTreeClassifier(
            **params["DecisionTree"],
        ),
    }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_and_save(
    task: str,
    model_name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_params: dict,
) -> None:
    """Fit one model with its fixed hyperparameters and save to disk."""
    out_path = model_path(task, model_name)

    is_gpu = isinstance(estimator, XGBClassifier)
    device_tag = "[GPU]" if is_gpu else "[CPU]"
    log(f"\n    ┌─ {model_name} {device_tag}")
    log(f"    │  Fitting with fixed hyperparameters …")

    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    log(f"    │  Total fit time      : {elapsed:.1f}s")

    joblib.dump(estimator, out_path)
    log(f"    └─ Saved: {out_path}")

    key = f"{task}|{model_name}"
    all_params[key] = {
        "used_params": FIXED_HYPERPARAMS[task][model_name],
        "fit_time":    round(elapsed, 2),
    }
    save_used_params(all_params)   # write after each model so nothing is lost


def split_indices_path(task: str) -> str:
    return os.path.join(CONFIG["data_cache_dir"], f"{task}_split_indices.npz")


def get_split(X, y, task: str):
    """Stratified train/test split with fallback for rare classes.

    No seed is used, so the split is saved to disk on first run and reused
    on every subsequent run (incl. by 03_evaluate.py) — otherwise a fresh
    unseeded split would silently leak test rows into the training set.
    """
    idx_path = split_indices_path(task)
    n = len(X)

    if os.path.exists(idx_path):
        log(f"    Reusing saved split: {idx_path}")
        npz = np.load(idx_path)
        train_idx, test_idx = npz["train_idx"], npz["test_idx"]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    all_idx = np.arange(n)
    try:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=CONFIG["test_size"], stratify=y)
    except ValueError as e:
        log(f"    WARNING: stratified split failed ({e}); falling back to random split")
        train_idx, test_idx = train_test_split(
            all_idx, test_size=CONFIG["test_size"], stratify=None)

    np.savez(idx_path, train_idx=train_idx, test_idx=test_idx)
    log(f"    Saved split: {idx_path}")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    separator("TRAINING SCRIPT — fixed hyperparameters, no CV, no seed")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Load preprocessed data ---
    cache = CONFIG["data_cache_dir"]
    log(f"\n  Loading cached arrays from {cache}/ …")

    X              = np.load(os.path.join(cache, "X.npy"))
    y_binary       = np.load(os.path.join(cache, "y_binary.npy"))
    y_multiclass   = np.load(os.path.join(cache, "y_multiclass.npy"))

    log(f"  X shape     : {X.shape}  dtype={X.dtype}")
    log(f"  y_binary    : {y_binary.shape}")
    log(f"  y_multiclass: {y_multiclass.shape}")

    TASKS = {"binary": y_binary, "multiclass": y_multiclass}

    all_params = load_used_params()   # load existing params for resume

    total_exps = len(TASKS) * 3   # 3 models
    done = 0

    for task, y in TASKS.items():
        separator(f"TASK: {task.upper()}")

        X_train, X_test, y_train, y_test = get_split(X, y, task)
        log(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

        models = get_models(task)

        model_bar = tqdm(
            models.items(),
            desc="    Models", unit="model",
            disable=not IS_TTY, position=0, leave=False,
            total=len(models), dynamic_ncols=True,
        )

        for model_name, estimator in model_bar:
            model_bar.set_description(f"    {model_name}")
            done += 1
            log(f"\n  [{done}/{total_exps}] task={task}  model={model_name}")

            fit_and_save(task, model_name, estimator, X_train, y_train, all_params)

        model_bar.close()

    elapsed = time.perf_counter() - t_start
    separator("TRAINING COMPLETE")
    log(f"  Models saved to : {os.path.abspath(CONFIG['models_dir'])}/")
    log(f"  Used params JSON: {os.path.abspath(params_path())}")
    log(f"  Total time      : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    log(f"  Finished        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
