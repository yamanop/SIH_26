"""
Runs Day 2 end to end: model ensemble training/scoring, then SHAP
explanations for the top-ranked wallets. Assumes run_day1.py has already
been run (needs feature_table.csv and the graph pickle).

Run: python src/run_day2.py
"""
import subprocess
import sys

STEPS = [
    ("train_models.py", "Stage 4: model ensemble (Isolation Forest + LightGBM + cluster risk)"),
    ("explain.py", "Explainability: SHAP-backed reasons for top-ranked wallets"),
]


def main():
    for script, label in STEPS:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[run_day2] FAILED at {script} -- stopping.")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("[run_day2] Day 2 complete. Outputs:")
    print("  - risk_scores.csv   (every wallet, ranked by final ensemble score)")
    print("  - lightgbm_model.txt")
    print("  - explanations.csv  (top-K wallets with SHAP-backed plain-English reasons)")


if __name__ == "__main__":
    main()
