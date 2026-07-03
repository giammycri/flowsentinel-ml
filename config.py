"""
Shared configuration for the CICIDS2017 ML pipeline.
Edit this file to change any parameter — all scripts read from here.
"""

import os

CONFIG = {
    # ---------- Dataset ----------
    "csv_path":       "Wednesday-workingHours.pcap_ISCX.csv",

    # ---------- Experiment ----------
    "seeds":          [42, 43, 44, 45, 46],
    "test_size":      0.30,
    "n_iter":         20,       # RandomizedSearchCV candidates per model
    "corr_threshold": 0.95,     # drop one of each pair with |r| > this

    # ---------- Directories ----------
    "data_cache_dir": "data_cache",   # preprocessed arrays saved here
    "models_dir":     "models",       # fitted .pkl models saved here
    "output_dir":     "output",       # plots and CSV results saved here
    "logs_dir":       "logs",         # log files saved here
}

# Create all directories on import so every script is ready to write
for _d in [CONFIG["data_cache_dir"], CONFIG["models_dir"],
           CONFIG["output_dir"],     CONFIG["logs_dir"]]:
    os.makedirs(_d, exist_ok=True)
