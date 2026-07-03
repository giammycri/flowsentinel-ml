"""
CICIDS2017 DoS Attack Detection — Full ML Pipeline
===================================================
Binary and multiclass classification with RF, XGBoost, and Decision Tree.
Runs each experiment across 5 seeds; reports mean ± std, Wilcoxon tests,
confusion matrices, and feature importance plots.

Usage:
    python3 ml_pipeline.py

All outputs (PNG plots, CSV tables) are saved to CONFIG["output_dir"].
"""

# ============================================================
# CONFIG — edit these values to customise the run
# ============================================================
CONFIG = {
    "csv_path":        "Wednesday-workingHours.pcap_ISCX.csv",
    "seeds":           [42, 43, 44, 45, 46],
    "test_size":       0.30,   # fraction of data used as test set
    "n_iter":          20,     # number of hyperparameter combinations tried per model
    "output_dir":      "output",
    "corr_threshold":  0.95,   # drop one of each pair with |r| > this value
}

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

import sklearn
import xgboost
import scipy


# ============================================================
# SECTION 1 — Library versions
# ============================================================

def print_versions() -> None:
    """Print library versions for reproducibility."""
    tqdm.write("\n" + "=" * 60)
    tqdm.write("LIBRARY VERSIONS")
    tqdm.write("=" * 60)
    tqdm.write(f"  Python     : {sys.version.split()[0]}")
    tqdm.write(f"  numpy      : {np.__version__}")
    tqdm.write(f"  pandas     : {pd.__version__}")
    tqdm.write(f"  scikit-learn: {sklearn.__version__}")
    tqdm.write(f"  xgboost    : {xgboost.__version__}")
    tqdm.write(f"  scipy      : {scipy.__version__}")
    tqdm.write(f"  seaborn    : {sns.__version__}")
    tqdm.write(f"  matplotlib : {matplotlib.__version__}")
    import tqdm as _tqdm_mod
    tqdm.write(f"  tqdm       : {_tqdm_mod.__version__}")
    # GPU info
    try:
        import subprocess
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        tqdm.write(f"  GPU        : {gpu_info}")
        tqdm.write(f"  XGBoost    : GPU training ENABLED (device=cuda)")
    except Exception:
        tqdm.write(f"  GPU        : not detected — XGBoost will use CPU")


# ============================================================
# SECTION 2 — Data loading and preprocessing
# ============================================================

def load_and_preprocess(
    csv_path: str,
    task: str,
    corr_threshold: float,
):
    """
    Load and clean the CICIDS2017 CSV, apply feature selection, encode labels.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    task : str
        "binary" (BENIGN=0, all attacks=1) or "multiclass" (original labels).
    corr_threshold : float
        Drop one column from each pair whose |Pearson r| exceeds this value.

    Returns
    -------
    X : np.ndarray, shape (N, F), float32
    y : np.ndarray, shape (N,), int
    feature_names : list[str]
    class_labels : list[str]   human-readable label names in encoded order
    """
    tqdm.write(f"\n  Loading {csv_path} …")
    with tqdm(total=1, desc="  Reading CSV", unit="file", leave=False, dynamic_ncols=True) as pbar:
        df = pd.read_csv(csv_path, low_memory=False)
        pbar.update(1)
    tqdm.write(f"  Raw shape: {df.shape}")

    # --- Strip whitespace from column names (fixes " Label") ---
    df.columns = df.columns.str.strip()

    # --- Remove duplicate column names (keep first occurrence) ---
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # --- Separate label ---
    y_raw = df.pop("Label")

    # --- Replace infinities, drop NaN rows ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    y_raw = y_raw.loc[df.index]
    tqdm.write(f"  Dropped {before - len(df)} rows with NaN/inf")

    # --- Drop duplicate rows ---
    combined = df.copy()
    combined["__lbl__"] = y_raw.values
    combined.drop_duplicates(inplace=True)
    combined.reset_index(drop=True, inplace=True)
    dropped_dupes = len(df) - len(combined)
    df = combined.drop(columns=["__lbl__"])
    y_raw = combined["__lbl__"].reset_index(drop=True)
    tqdm.write(f"  Dropped {dropped_dupes} duplicate rows")

    # --- Drop constant columns ---
    non_const = df.columns[df.nunique() > 1]
    dropped_const = len(df.columns) - len(non_const)
    df = df[non_const]
    tqdm.write(f"  Dropped {dropped_const} constant columns")

    # --- Convert to float32 (halves memory) ---
    df = df.astype(np.float32)

    # --- Correlation-based feature selection ---
    tqdm.write("  Computing feature correlations …")
    with tqdm(total=1, desc="  Correlation filter", unit="step", leave=False, dynamic_ncols=True) as pbar:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
        df.drop(columns=to_drop, inplace=True)
        pbar.update(1)
    tqdm.write(f"  Dropped {len(to_drop)} highly correlated features (threshold={corr_threshold})")
    tqdm.write(f"  Remaining features: {df.shape[1]}")

    feature_names = df.columns.tolist()
    X = df.values  # float32 numpy array

    # --- Encode labels ---
    if task == "binary":
        y = (y_raw != "BENIGN").astype(int).values
        class_labels = ["BENIGN", "ATTACK"]
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.values)
        class_labels = le.classes_.tolist()

    tqdm.write(f"  Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    tqdm.write(f"  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = class_labels[u] if u < len(class_labels) else str(u)
        tqdm.write(f"    {lbl}: {c}")

    return X, y, feature_names, class_labels


# ============================================================
# SECTION 3 — Model definitions and hyperparameter spaces
# ============================================================

def _make_xgboost(seed: int):
    """Create XGBClassifier with GPU acceleration when available."""
    kwargs = dict(
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        device="cuda",   # Tesla T4 — GPU training via CUDA
        # n_jobs is omitted: GPU handles parallelism internally
    )
    try:
        return XGBClassifier(**kwargs, use_label_encoder=False)
    except TypeError:
        return XGBClassifier(**kwargs)


def get_model_param_spaces(seed: int) -> dict:
    """
    Return {model_name: (estimator, param_distributions)} for all three models.
    Each estimator is seeded for reproducibility.
    """
    spaces = {
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
                "n_estimators":    [100, 200, 300, 400, 500],
                "max_depth":       [3, 5, 7, 9, 12],
                "learning_rate":   [0.01, 0.05, 0.1, 0.2, 0.3],
                "subsample":       [0.5, 0.6, 0.8, 0.9, 1.0],
                "colsample_bytree":[0.5, 0.6, 0.8, 0.9, 1.0],
                "gamma":           [0, 0.05, 0.1, 0.5, 1.0],
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
    return spaces


# ============================================================
# SECTION 4 — Hyperparameter tuning
# ============================================================

def tune_model(
    estimator,
    param_dist: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int,
    seed: int,
    model_name: str,
):
    """
    Run RandomizedSearchCV on X_train only (test set never used).
    Uses 3-fold CV internally to score hyperparameter combinations.
    Prints best parameters and best CV score.

    Returns the best estimator refitted on full X_train.
    """
    # XGBoost runs on GPU: serialise the search to avoid multiple processes
    # competing for the same GPU device and causing CUDA out-of-memory errors.
    search_jobs = 1 if isinstance(estimator, XGBClassifier) else -1
    device_note = " [GPU]" if isinstance(estimator, XGBClassifier) else " [CPU]"
    tqdm.write(f"      Fitting {n_iter} candidates × 3 folds = {n_iter * 3} fits{device_note} …")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1_weighted",
        n_jobs=search_jobs,
        random_state=seed,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    tqdm.write(f"      Best CV F1  : {search.best_score_:.4f}")
    tqdm.write(f"      Best params : {search.best_params_}")
    return search.best_estimator_


# ============================================================
# SECTION 5 — Evaluation
# ============================================================

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> dict:
    """
    Compute all metrics on the test set.
    Returns a dict with accuracy, precision, recall, f1, roc_auc,
    inference_time, and confusion_matrix.
    """
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
            roc_auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            roc_auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc":          roc_auc,
        "inference_time":   inference_time,
        "confusion_matrix": cm,
    }


# ============================================================
# SECTION 6 — Plotting helpers
# ============================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str,
    path: str,
) -> None:
    """Save a confusion matrix heatmap as a PNG file."""
    n = len(labels)
    fig_size = max(6, n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tqdm.write(f"    Saved: {path}")


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    title: str,
    path: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top-N feature importances."""
    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in indices]
    top_vals  = importances[indices]

    # Reverse so highest bar is at the top
    top_feats = top_feats[::-1]
    top_vals  = top_vals[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_vals, color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tqdm.write(f"    Saved: {path}")


# ============================================================
# SECTION 7 — Statistical tests
# ============================================================

def run_wilcoxon(results_df: pd.DataFrame, task: str) -> None:
    """
    Perform pairwise Wilcoxon Signed-Rank Tests on F1 scores across 5 seeds
    for the given task. Prints significance at alpha=0.05.
    """
    df = results_df[results_df["task"] == task].copy()

    def get_f1(model_name):
        return df[df["model"] == model_name].sort_values("seed")["f1"].values

    models = df["model"].unique().tolist()
    model_f1 = {m: get_f1(m) for m in models}

    tqdm.write(f"\n  --- Wilcoxon Signed-Rank Test  [{task}] ---")
    pairs = [
        ("RandomForest", "XGBoost"),
        ("RandomForest", "DecisionTree"),
        ("XGBoost",      "DecisionTree"),
    ]
    for a_name, b_name in pairs:
        if a_name not in model_f1 or b_name not in model_f1:
            continue
        a, b = model_f1[a_name], model_f1[b_name]
        if np.all(a == b):
            tqdm.write(f"    {a_name} vs {b_name}: identical scores — test skipped")
            continue
        try:
            stat, p = wilcoxon(a, b, zero_method="pratt")
            sig = "*** SIGNIFICANT ***" if p < 0.05 else "not significant"
            tqdm.write(f"    {a_name} vs {b_name}: stat={stat:.4f}, p={p:.4f}  [{sig}]")
        except ValueError as e:
            tqdm.write(f"    {a_name} vs {b_name}: test failed ({e})")


# ============================================================
# SECTION 8 — Main orchestration
# ============================================================

def main() -> None:
    print_versions()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    all_results   = []   # list of dicts — one row per (seed, task, model)
    last_artifacts = {}  # {(model_name, task): {cm, model, class_labels, feature_names}}

    TASKS = ["binary", "multiclass"]

    task_bar = tqdm(TASKS, desc="Tasks", unit="task", position=0, leave=True, dynamic_ncols=True)

    for task in task_bar:
        task_bar.set_description(f"Task: {task}")
        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"  TASK: {task.upper()}")
        tqdm.write("=" * 60)

        # Preprocessing is done once per task (correlation filter is expensive)
        X, y, feature_names, class_labels = load_and_preprocess(
            CONFIG["csv_path"], task, CONFIG["corr_threshold"]
        )
        n_classes = len(class_labels)

        seed_bar = tqdm(
            CONFIG["seeds"], desc="  Seeds", unit="seed",
            position=1, leave=False, dynamic_ncols=True,
        )

        for seed in seed_bar:
            seed_bar.set_description(f"  Seed {seed}")
            tqdm.write(f"\n  [Seed {seed}]")

            # Fix global RNG state for this seed
            random.seed(seed)
            np.random.seed(seed)

            # Stratified train/test split — fall back if a class is too rare
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=CONFIG["test_size"],
                    random_state=seed,
                    stratify=y,
                )
            except ValueError as e:
                tqdm.write(f"    WARNING: stratified split failed ({e}); using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=CONFIG["test_size"],
                    random_state=seed,
                    stratify=None,
                )

            model_spaces = get_model_param_spaces(seed)

            model_bar = tqdm(
                model_spaces.items(), desc="    Models", unit="model",
                position=2, leave=False, dynamic_ncols=True,
                total=len(model_spaces),
            )

            for model_name, (estimator, param_dist) in model_bar:
                model_bar.set_description(f"    {model_name}")
                tqdm.write(f"\n    [{model_name}]")

                t_train_start = time.perf_counter()
                best_model = tune_model(
                    estimator, param_dist,
                    X_train, y_train,
                    CONFIG["n_iter"], seed, model_name,
                )
                train_time = time.perf_counter() - t_train_start

                metrics = evaluate(best_model, X_test, y_test, task)

                row = {
                    "seed":           seed,
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

                tqdm.write(
                    f"      Test  → Acc={metrics['accuracy']:.4f}  "
                    f"F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}  "
                    f"train={train_time:.1f}s  inf={metrics['inference_time']:.3f}s"
                )

                # Keep last-seed artifacts for plots
                if seed == CONFIG["seeds"][-1]:
                    last_artifacts[(model_name, task)] = {
                        "cm":            metrics["confusion_matrix"],
                        "model":         best_model,
                        "class_labels":  class_labels,
                        "feature_names": feature_names,
                    }

            model_bar.close()

    # --------------------------------------------------------
    # AGGREGATE RESULTS
    # --------------------------------------------------------
    raw_df = pd.DataFrame(all_results)
    raw_path = os.path.join(CONFIG["output_dir"], "all_results_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    tqdm.write(f"\n  Saved raw results → {raw_path}")

    metric_cols = ["accuracy", "precision", "recall", "f1",
                   "roc_auc", "train_time", "inference_time"]

    summary_rows = []
    for (task, model), grp in raw_df.groupby(["task", "model"]):
        row = {"task": task, "model": model}
        for col in metric_cols:
            m, s = grp[col].mean(), grp[col].std()
            row[f"{col}_mean"]     = round(m, 4)
            row[f"{col}_std"]      = round(s, 4)
            row[f"{col}_mean±std"] = f"{m:.4f} ± {s:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(CONFIG["output_dir"], "results_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    tqdm.write(f"  Saved summary    → {sum_path}")

    # --------------------------------------------------------
    # PRINT SUMMARY TABLE
    # --------------------------------------------------------
    display_cols = ["task", "model"] + [f"{c}_mean±std" for c in metric_cols]
    tqdm.write("\n" + "=" * 60)
    tqdm.write("RESULTS SUMMARY (mean ± std across 5 seeds)")
    tqdm.write("=" * 60)
    tqdm.write(summary_df[display_cols].to_string(index=False))

    # --------------------------------------------------------
    # WILCOXON SIGNED-RANK TESTS
    # --------------------------------------------------------
    tqdm.write("\n" + "=" * 60)
    tqdm.write("STATISTICAL TESTS")
    tqdm.write("=" * 60)
    for task in TASKS:
        run_wilcoxon(raw_df, task)

    # --------------------------------------------------------
    # PLOTS — confusion matrices and feature importances
    # --------------------------------------------------------
    tqdm.write("\n" + "=" * 60)
    tqdm.write("SAVING PLOTS  (from last seed: {})".format(CONFIG["seeds"][-1]))
    tqdm.write("=" * 60)

    for (model_name, task), art in last_artifacts.items():
        # Confusion matrix
        cm_path = os.path.join(
            CONFIG["output_dir"],
            f"{model_name}_{task}_confusion_matrix.png",
        )
        plot_confusion_matrix(
            art["cm"],
            art["class_labels"],
            title=f"{model_name} ({task}) — Confusion Matrix  [seed {CONFIG['seeds'][-1]}]",
            path=cm_path,
        )

        # Feature importance (RF and XGBoost only)
        if model_name in ("RandomForest", "XGBoost"):
            fi_path = os.path.join(
                CONFIG["output_dir"],
                f"{model_name}_{task}_feature_importance.png",
            )
            plot_feature_importance(
                art["model"].feature_importances_,
                art["feature_names"],
                title=f"{model_name} ({task}) — Top-20 Feature Importances  [seed {CONFIG['seeds'][-1]}]",
                path=fi_path,
            )

    tqdm.write("\nAll done. Outputs saved to:", os.path.abspath(CONFIG["output_dir"]))


# ============================================================
if __name__ == "__main__":
    main()
