"""
00_label.py — Label the re-extracted CICIDS2017 flows using Engelen et al.'s script.

CICids2017new.csv was produced with the fixed CICFlowMeter from Engelen et al.
(WTMC2021), which does not label flows: every row carries "NeedManualLabel".
This stage runs the authors' own labelling_CSV_flows.py to fill in the Label
column, so the labelling methodology is theirs, unmodified.

The upstream script is used as a library and never edited: this wrapper only
rebinds its module-level constants, which its functions read at call time.

Two constants must deviate from the upstream defaults:

  TIME_DIFFERENCE      Upstream defaults to 5h. Our capture host wrote
                       timestamps 3h ahead of New Brunswick: the CSV spans
                       11:42:42-20:10:14 while the CIC capture spans
                       08:42-17:10 local. Verified against all five Wednesday
                       attack windows (e.g. GoldenEye lands on 11:10-11:23
                       exactly). A wrong value silently mislabels everything.

  DATE_FORMAT_DATASET  Upstream expects '05/07/2017 09:47:00 AM'; this CSV is
                       ISO with microseconds ('2017-07-05 11:42:42.084372').

Usage:
    python3 -u 00_label.py
"""

import os
import sys
import csv
import time
import shutil
import importlib.util
from datetime import timedelta

# ---------------------------------------------------------------------------

SRC_CSV      = "CICids2017new.csv"
UPSTREAM     = "third_party/WTMC2021-Code/labelling_CSV_flows.py"

INPUT_DIR    = "unlabelled/"
OUTPUT_DIR   = "labelled/"

# dataset_labeling(day) builds its paths as
# INPUT_DIR + "<Day>-WorkingHours.pcap_Flow.csv" -> OUTPUT_DIR + "<Day>-WorkingHours.pcap_REVI.csv"
WEDNESDAY    = 3
IN_NAME      = "Wednesday-WorkingHours.pcap_Flow.csv"
OUT_NAME     = "Wednesday-WorkingHours.pcap_REVI.csv"

TIME_DIFFERENCE_HOURS = 3
DATE_FORMAT_DATASET   = "%Y-%m-%d %H:%M:%S.%f"
PAYLOAD_FILTER_ACTIVE = True     # "X - Attempted" for TCP flows with no fwd payload


def log(msg: str) -> None:
    print(msg, flush=True)


def separator(title: str = "") -> None:
    line = "=" * 64
    log(f"\n{line}")
    if title:
        log(f"  {title}")
        log(line)


def load_upstream():
    """Import labelling_CSV_flows.py from third_party/ without touching sys.path."""
    if not os.path.exists(UPSTREAM):
        log(f"  ERROR: {UPSTREAM} not found.")
        log("  Clone it first:")
        log("    git clone --depth 1 https://github.com/GintsEngelen/WTMC2021-Code.git \\")
        log("        third_party/WTMC2021-Code")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("labelling_CSV_flows", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t_start = time.perf_counter()
    separator("STAGE 0 — LABELLING (Engelen et al., WTMC2021)")

    if not os.path.exists(SRC_CSV):
        log(f"  ERROR: {SRC_CSV} not found in {os.getcwd()}")
        sys.exit(1)

    mod = load_upstream()

    # The upstream functions read these as globals on every call, so rebinding
    # them here is enough — no edit to their file.
    mod.TIME_DIFFERENCE      = timedelta(hours=TIME_DIFFERENCE_HOURS)
    mod.DATE_FORMAT_DATASET  = DATE_FORMAT_DATASET
    mod.PAYLOAD_FILTER_ACTIVE = PAYLOAD_FILTER_ACTIVE
    mod.INPUT_DIR            = INPUT_DIR
    mod.OUTPUT_DIR           = OUTPUT_DIR

    log(f"  TIME_DIFFERENCE       : {mod.TIME_DIFFERENCE}  (upstream default: 5:00:00)")
    log(f"  DATE_FORMAT_DATASET   : {mod.DATE_FORMAT_DATASET}")
    log(f"  PAYLOAD_FILTER_ACTIVE : {mod.PAYLOAD_FILTER_ACTIVE}")

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # dataset_labeling() derives the input filename from the weekday, so expose
    # the source CSV under the name it expects (symlink: the file is 309 MB).
    staged = os.path.join(INPUT_DIR, IN_NAME)
    if os.path.islink(staged) or os.path.exists(staged):
        os.remove(staged)
    os.symlink(os.path.abspath(SRC_CSV), staged)
    log(f"\n  Input  : {staged} -> {SRC_CSV}")

    out_path = os.path.join(OUTPUT_DIR, OUT_NAME)
    log(f"  Output : {out_path}")

    separator("Labelling Wednesday flows")
    log("  (pure-python CSV pass over ~497k rows — expect a few minutes)")
    t0 = time.perf_counter()
    mod.dataset_labeling(WEDNESDAY)
    log(f"\n  Labelled in {time.perf_counter() - t0:.1f}s")

    separator("STAGE 0 COMPLETE")
    log(f"  Total time : {time.perf_counter() - t_start:.1f}s")
    log(f"  Next       : python3 -u 01_preprocess.py")
    log("")


if __name__ == "__main__":
    main()
