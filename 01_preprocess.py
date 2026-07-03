"""
01_preprocess.py — Load, clean, and save the CICIDS2017 dataset.

Run once. Saves processed arrays to data_cache/ so the other scripts
can load them instantly without re-reading the 225 MB CSV every time.

Usage:
    python3 -u 01_preprocess.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import CONFIG

IS_TTY = sys.stdout.isatty()   # False when output is redirected to a log file

# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a message that always appears in both terminal and log file."""
    print(msg, flush=True)


def separator(title: str = "") -> None:
    line = "=" * 64
    log(f"\n{line}")
    if title:
        log(f"  {title}")
        log(line)


# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    separator("STEP 1 — Loading CSV")
    log(f"  File : {path}")
    t0 = time.perf_counter()
    with tqdm(total=1, desc="  Reading", unit="file",
              disable=not IS_TTY, leave=False) as pb:
        df = pd.read_csv(path, low_memory=False)
        pb.update(1)
    elapsed = time.perf_counter() - t0
    log(f"  Shape: {df.shape}  ({elapsed:.1f}s)")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    separator("STEP 2 — Cleaning")

    # Strip whitespace from column names (fixes " Label" → "Label")
    df.columns = df.columns.str.strip()
    log("  Column names stripped")

    # Remove duplicate column names (keeps first — fixes duplicate 'Fwd Header Length')
    before_cols = len(df.columns)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    dropped_cols = before_cols - len(df.columns)
    if dropped_cols:
        log(f"  Removed {dropped_cols} duplicate column(s)")

    # Separate label
    y_raw = df.pop("Label")

    # Replace inf with NaN, then drop rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    y_raw = y_raw.loc[df.index]
    log(f"  Dropped {before - len(df):,} rows with NaN/inf  (remaining: {len(df):,})")

    # Drop duplicate rows
    combined = df.copy()
    combined["__lbl__"] = y_raw.values
    combined.drop_duplicates(inplace=True)
    combined.reset_index(drop=True, inplace=True)
    dropped_dupes = len(df) - len(combined)
    df      = combined.drop(columns=["__lbl__"])
    y_raw   = combined["__lbl__"].reset_index(drop=True)
    log(f"  Dropped {dropped_dupes:,} duplicate rows  (remaining: {len(df):,})")

    # Drop constant / zero-variance columns
    non_const = df.columns[df.nunique() > 1]
    dropped_const = len(df.columns) - len(non_const)
    df = df[non_const]
    log(f"  Dropped {dropped_const} constant columns  (remaining: {len(df.columns)})")

    # Convert to float32 (halves memory vs float64)
    df = df.astype(np.float32)
    log(f"  Cast to float32")

    return df, y_raw


def correlation_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    separator("STEP 3 — Correlation filter")
    log(f"  Threshold : |r| > {threshold}")
    log("  Computing pairwise Pearson correlation matrix …")
    t0 = time.perf_counter()
    with tqdm(total=1, desc="  Correlating", unit="step",
              disable=not IS_TTY, leave=False) as pb:
        corr = df.corr().abs()
        pb.update(1)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    df.drop(columns=to_drop, inplace=True)
    elapsed = time.perf_counter() - t0
    log(f"  Dropped {len(to_drop)} correlated features  ({elapsed:.1f}s)")
    log(f"  Remaining features: {df.shape[1]}")
    if to_drop:
        log(f"  Dropped: {to_drop}")
    return df


def encode_labels(y_raw: pd.Series):
    """Returns (y_binary, y_multiclass, class_labels_binary, class_labels_multi)."""
    y_binary = (y_raw != "BENIGN").astype(np.int32).values
    class_labels_binary = ["BENIGN", "ATTACK"]

    le = LabelEncoder()
    y_multi = le.fit_transform(y_raw.values).astype(np.int32)
    class_labels_multi = le.classes_.tolist()

    return y_binary, y_multi, class_labels_binary, class_labels_multi


def print_class_distribution(y: np.ndarray, labels: list, title: str) -> None:
    log(f"\n  {title}:")
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for u, c in zip(unique, counts):
        pct = 100.0 * c / total
        lbl = labels[u] if u < len(labels) else str(u)
        log(f"    [{u}] {lbl:<30s}  {c:>7,}  ({pct:.1f}%)")


def save_cache(df: pd.DataFrame, y_binary: np.ndarray, y_multi: np.ndarray,
               feature_names: list, class_labels_binary: list,
               class_labels_multi: list, cache_dir: str) -> None:
    separator("STEP 4 — Saving to cache")

    np.save(os.path.join(cache_dir, "X.npy"),            df.values)
    np.save(os.path.join(cache_dir, "y_binary.npy"),     y_binary)
    np.save(os.path.join(cache_dir, "y_multiclass.npy"), y_multi)

    with open(os.path.join(cache_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(cache_dir, "class_labels_binary.json"), "w") as f:
        json.dump(class_labels_binary, f, indent=2)

    with open(os.path.join(cache_dir, "class_labels_multiclass.json"), "w") as f:
        json.dump(class_labels_multi, f, indent=2)

    log(f"  X.npy              → {df.values.nbytes / 1e6:.1f} MB  shape {df.values.shape}")
    log(f"  y_binary.npy       → {y_binary.shape}")
    log(f"  y_multiclass.npy   → {y_multi.shape}")
    log(f"  feature_names.json → {len(feature_names)} features")
    log(f"  class_labels saved → binary: {class_labels_binary}")
    log(f"                        multi : {class_labels_multi}")
    log(f"\n  All files saved to: {os.path.abspath(cache_dir)}")


# ---------------------------------------------------------------------------

CACHE_FILES = [
    "X.npy", "y_binary.npy", "y_multiclass.npy",
    "feature_names.json", "class_labels_binary.json", "class_labels_multiclass.json",
]


def cache_is_valid(cache_dir: str) -> bool:
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in CACHE_FILES)


def main():
    t_start = time.perf_counter()
    separator("CICIDS2017 — PREPROCESSING SCRIPT")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    cache_dir = CONFIG["data_cache_dir"]

    # Skip if all cache files already exist
    if cache_is_valid(cache_dir):
        log(f"\n  Cache already exists in {cache_dir}/ — skipping preprocessing.")
        log(f"  (Delete {cache_dir}/ and re-run to force reprocessing.)")
        separator("PREPROCESSING SKIPPED")
        log("")
        return

    df         = load_csv(CONFIG["csv_path"])
    df, y_raw  = clean(df)
    df         = correlation_filter(df, CONFIG["corr_threshold"])

    separator("STEP 3b — Label encoding")
    y_binary, y_multi, cl_binary, cl_multi = encode_labels(y_raw)
    print_class_distribution(y_binary, cl_binary,  "Binary distribution")
    print_class_distribution(y_multi,  cl_multi,   "Multiclass distribution")

    feature_names = df.columns.tolist()
    save_cache(df, y_binary, y_multi, feature_names, cl_binary, cl_multi, cache_dir)

    elapsed = time.perf_counter() - t_start
    separator("PREPROCESSING COMPLETE")
    log(f"  Total time : {elapsed:.1f}s")
    log(f"  Finished   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")


if __name__ == "__main__":
    main()
