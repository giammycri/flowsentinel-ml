#!/bin/bash
# run_all.sh — Runs the full pipeline in sequence.
# -u flag makes Python output unbuffered so every print appears immediately.
#
# Run in foreground (you see everything live):
#   bash run_all.sh
#
# Run detached (survives closing VSCode / terminal) — use launch.sh instead.

set -e   # stop immediately if any script exits with an error

echo ""
echo "================================================================"
echo "  CICIDS2017 ML Pipeline"
echo "  Started: $(date)"
echo "================================================================"
echo ""

echo ">>> [1/3] PREPROCESSING"
echo "----------------------------------------------------------------"
python3 -u 01_preprocess.py
echo ""

echo ">>> [2/3] HYPERPARAMETER TUNING"
echo "----------------------------------------------------------------"
python3 -u 02_tune.py
echo ""

echo ">>> [3/3] EVALUATION & PLOTS"
echo "----------------------------------------------------------------"
python3 -u 03_evaluate.py
echo ""

echo "================================================================"
echo "  ALL DONE"
echo "  Finished: $(date)"
echo "================================================================"
echo ""
