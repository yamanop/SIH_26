"""
Runs the entire Day 1 pipeline end to end:
  Stage 0a: synthetic Elliptic++ stand-in data
  Stage 0b: synthetic IP/network-layer generator
  Stage 1:  ingestion, validation, dedup, timestamp alignment
  Stage 2:  feature engineering
  Stage 3:  graph construction

If data/raw/txs_features.csv already exists (e.g. you've swapped in the
real Elliptic++ download), the synthetic generator step is skipped
automatically so you don't overwrite real data.

Run: python src/run_day1.py
"""
import subprocess
import sys

import config as cfg

STEPS = [
    ("synthetic_elliptic_data.py", "Stage 0a: synthetic Elliptic++ stand-in data"),
    ("ip_generator.py", "Stage 0b: synthetic network-layer generator"),
    ("ingest.py", "Stage 1: ingestion & parsing"),
    ("features.py", "Stage 2: feature engineering"),
    ("graph_build.py", "Stage 3: graph construction"),
]


def main():
    if cfg.RAW_FILES["txs_features"].exists():
        print("[run_day1] data/raw/txs_features.csv already present -- "
              "skipping synthetic data generation (assuming real data is in place)")
        steps = STEPS[1:]  # still regenerate the network layer + run the rest
    else:
        steps = STEPS

    for script, label in steps:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[run_day1] FAILED at {script} -- stopping.")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("[run_day1] Day 1 complete. Outputs:")
    print(f"  - unified table : {cfg.PROCESSED_FILES['unified']}")
    print(f"  - feature table  : {cfg.PROCESSED_FILES['features']}")
    print(f"  - graph          : {cfg.PROCESSED_FILES['graph']}")
    print("Do not start Day 2 (modeling) until you've eyeballed all three.")


if __name__ == "__main__":
    main()
