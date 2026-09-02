"""
Shared configuration for the CICIDS2017 ML pipeline.
Edit this file to change any parameter — all scripts read from here.
"""

import os

CONFIG = {
    # ---------- Dataset ----------
    "csv_path":       "flussotest/dataset_finale.csv",

    # ---------- Experiment ----------
    "test_size":      0.30,
    "corr_threshold": 0.95,     # drop one of each pair with |r| > this

    # ---------- Directories ----------
    "data_cache_dir": "data_cache",   # preprocessed arrays saved here
    "models_dir":     "models",       # fitted .pkl models saved here
    "output_dir":     "output",       # plots and CSV results saved here
    "logs_dir":       "logs",         # log files saved here
}

# ---------------------------------------------------------------------------
# Fixed hyperparameters (no more RandomizedSearchCV / cross-validation).
# Reused from the best result of the previous tuning run (models/best_params.json,
# seed 42) for each task, so training is a single direct fit per model.
# ---------------------------------------------------------------------------

FIXED_HYPERPARAMS = {
    "binary": {
        "RandomForest": {
            "n_estimators":      100,
            "max_depth":         30,
            "min_samples_split": 10,
            "min_samples_leaf":  1,
            "max_features":      0.2,
        },
        "XGBoost": {
            "n_estimators":     300,
            "max_depth":        12,
            "learning_rate":    0.05,
            "subsample":        0.5,
            "colsample_bytree": 0.5,
            "gamma":            0.05,
        },
        "DecisionTree": {
            "max_depth":         30,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "criterion":         "entropy",
            "splitter":          "best",
        },
    },
    "multiclass": {
        "RandomForest": {
            "n_estimators":      100,
            "max_depth":         None,
            "min_samples_split": 5,
            "min_samples_leaf":  1,
            "max_features":      "log2",
        },
        "XGBoost": {
            "n_estimators":     400,
            "max_depth":        3,
            "learning_rate":    0.05,
            "subsample":        1.0,
            "colsample_bytree": 0.5,
            "gamma":            0.05,
        },
        "DecisionTree": {
            "max_depth":         30,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "criterion":         "entropy",
            "splitter":          "best",
        },
    },
}

# ---------------------------------------------------------------------------
# CICFlowMeter V4 (new naming) → CICIDS2017 (the original dataset's naming).
#
# Only the columns whose names actually differ are listed; every other feature
# already carries the same name in both conventions. Keeping the original
# naming means the feature space is identical to the first experiment, so any
# difference in results is attributable to the flow extraction fix alone.
#
# Used by 01_preprocess.py (training data) and 04_test_external.py (external
# captures), which must agree — hence a single definition here.
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "Dst Port":                   "Destination Port",
    "Total Fwd Packet":           "Total Fwd Packets",
    "Total Bwd packets":          "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min":          "Min Packet Length",
    "Packet Length Max":          "Max Packet Length",
    "CWR Flag Count":             "CWE Flag Count",
    "Fwd Segment Size Avg":       "Avg Fwd Segment Size",
    "Bwd Segment Size Avg":       "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg":         "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg":        "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg":          "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg":         "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg":        "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg":          "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes":         "Init_Win_bytes_forward",
    "Bwd Init Win Bytes":         "Init_Win_bytes_backward",
    "Fwd Act Data Pkts":          "act_data_pkt_fwd",
    "Fwd Seg Size Min":           "min_seg_size_forward",
}

# Columns present in the new extraction but not in the original CICIDS2017 CSV.
# Dropped to keep the feature space identical to the first experiment:
#   - flow identifiers, which are not features and would leak the ground truth
#     (the labelling itself is derived from IP/port/timestamp)
#   - features the original CICFlowMeter never emitted
DROP_COLUMNS = [
    # identifiers
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp",
    "Protocol",                      # absent from the original CSV
    # features new to the V4 extractor
    "Fwd RST Flags", "Bwd RST Flags",
    "Bwd Act Data Pkts", "Bwd Seg Size Min",
    "ICMP Code", "ICMP Type",
    "Fwd TCP Retrans. Count", "Bwd TCP Retrans. Count", "Total TCP Retrans. Count",
    "Total Connection Flow Time",
]

# Heartbleed is dropped entirely: 11 flows is too few to learn or evaluate on.
# Matches both "Heartbleed" and "Heartbleed - Attempted".
DROP_LABEL_PREFIX = "Heartbleed"

# Create all directories on import so every script is ready to write
for _d in [CONFIG["data_cache_dir"], CONFIG["models_dir"],
           CONFIG["output_dir"],     CONFIG["logs_dir"]]:
    os.makedirs(_d, exist_ok=True)
