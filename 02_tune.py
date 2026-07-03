"""
02_tune.py — Hyperparameter tuning and model fitting for all experiments.

Loads preprocessed data from data_cache/, runs RandomizedSearchCV for every
(task, seed, model) combination, and saves the fitted models to models/.

RESUME SUPPORT: if a model file already exists it is skipped automatically.
Re-run the script after a crash and it picks up where it left off.

Usage:
    python3 -u 02_tune.py
"""

import os
import sys
import time
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import CONFIG

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


def model_path(task: str, seed: int, model_name: str) -> str:
    fname = f"{task}_seed{seed}_{model_name}.pkl"
    return os.path.join(CONFIG["models_dir"], fname)


def params_path() -> str:
    return os.path.join(CONFIG["models_dir"], "best_params.json")


def load_best_params() -> dict:
    p = params_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_best_params(all_params: dict) -> None:
    with open(params_path(), "w") as f:
        json.dump(all_params, f, indent=2)

# ---------------------------------------------------------------------------
# Model / hyperparameter definitions
# ---------------------------------------------------------------------------

def _make_xgboost(seed: int) -> XGBClassifier:
    kwargs = dict(random_state=seed, eval_metric="logloss",
                  verbosity=0, device="cuda")
    try:
        return XGBClassifier(**kwargs, use_label_encoder=False)
    except TypeError:
        return XGBClassifier(**kwargs)


def get_model_param_spaces(seed: int) -> dict:
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators":      [100, 200, 300, 500, 700],
                "max_depth":         [None, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf":  [1, 2, 4, 8, 16],
                "max_features":      ["sqrt", "log2", 0.2, 0.4, 0.6],
            },
        ),
        "XGBoost": (
            _make_xgboost(seed),
            {
                "n_estimators":     [100, 200, 300, 400, 500],
                "max_depth":        [3, 5, 7, 9, 12],
                "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
                "subsample":        [0.5, 0.6, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.6, 0.8, 0.9, 1.0],
                "gamma":            [0, 0.05, 0.1, 0.5, 1.0],
            },
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "max_depth":         [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 20, 50],
                "min_samples_leaf":  [1, 2, 4, 8, 16],
                "criterion":         ["gini", "entropy"],
                "splitter":          ["best", "random"],
            },
        ),
    }

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_and_save(
    task: str,
    seed: int,
    model_name: str,
    estimator,
    param_dist: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_params: dict,
) -> None:
    """Tune one model, refit on full training set, save to disk."""
    out_path = model_path(task, seed, model_name)

    # Resume: skip if already done
    if os.path.exists(out_path):
        log(f"      SKIP — already exists: {out_path}")
        return

    is_gpu = isinstance(estimator, XGBClassifier)
    device_tag = "[GPU]" if is_gpu else "[CPU]"
    n_fits = CONFIG["n_iter"] * 3
    log(f"\n    ┌─ {model_name} {device_tag}")
    log(f"    │  Fitting {CONFIG['n_iter']} candidates × 3 folds = {n_fits} fits …")

    # n_jobs=1 in the search so all output goes through the main process
    # stdout — this is what makes verbose=2 appear in the log file.
    # The estimator itself still uses n_jobs=-1 (RF) or device=cuda (XGBoost)
    # so each individual fit is still parallelised internally.
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=CONFIG["n_iter"],
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
        random_state=seed,
        refit=True,
        verbose=2,          # each fold printed with timing — visible because n_jobs=1
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_params = search.best_params_
    best_score  = search.best_score_

    log(f"    │  Best CV F1-weighted : {best_score:.6f}")
    log(f"    │  Best params         : {best_params}")
    log(f"    │  Total tuning time   : {elapsed:.1f}s")

    # Save fitted model
    joblib.dump(search.best_estimator_, out_path)
    log(f"    └─ Saved: {out_path}")

    # Record params
    key = f"{task}|seed{seed}|{model_name}"
    all_params[key] = {
        "best_score":  round(best_score, 6),
        "best_params": best_params,
        "tuning_time": round(elapsed, 2),
    }
    save_best_params(all_params)   # write after each model so nothing is lost


def get_split(X, y, seed):
    """Stratified train/test split with fallback for rare classes (Heartbleed)."""
    try:
        return train_test_split(X, y, test_size=CONFIG["test_size"],
                                random_state=seed, stratify=y)
    except ValueError as e:
        log(f"    WARNING: stratified split failed ({e}); falling back to random split")
        return train_test_split(X, y, test_size=CONFIG["test_size"],
                                random_state=seed, stratify=None)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    separator("CICIDS2017 — TUNING SCRIPT")
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

    all_params = load_best_params()   # load existing params for resume

    # Count total experiments for overall progress
    total_exps = len(TASKS) * len(CONFIG["seeds"]) * 3   # 3 models
    done = 0

    for task, y in TASKS.items():
        separator(f"TASK: {task.upper()}")

        task_bar = tqdm(CONFIG["seeds"], desc=f"  Seeds [{task}]",
                        unit="seed", disable=not IS_TTY,
                        position=0, leave=True, dynamic_ncols=True)

        for seed in task_bar:
            task_bar.set_description(f"  Seed {seed} [{task}]")
            log(f"\n  {'─'*56}")
            log(f"  Seed {seed}  |  task={task}")
            log(f"  {'─'*56}")

            random.seed(seed)
            np.random.seed(seed)

            X_train, X_test, y_train, y_test = get_split(X, y, seed)
            log(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

            model_spaces = get_model_param_spaces(seed)

            model_bar = tqdm(
                model_spaces.items(),
                desc="    Models", unit="model",
                disable=not IS_TTY, position=1, leave=False,
                total=len(model_spaces), dynamic_ncols=True,
            )

            for model_name, (estimator, param_dist) in model_bar:
                model_bar.set_description(f"    {model_name}")
                done += 1
                log(f"\n  [{done}/{total_exps}] task={task}  seed={seed}  model={model_name}")

                tune_and_save(
                    task, seed, model_name,
                    estimator, param_dist,
                    X_train, y_train,
                    all_params,
                )

            model_bar.close()

    elapsed = time.perf_counter() - t_start
    separator("TUNING COMPLETE")
    log(f"  Models saved to : {os.path.abspath(CONFIG['models_dir'])}/")
    log(f"  Best params JSON: {os.path.abspath(params_path())}")
    log(f"  Total time      : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    log(f"  Finished        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
