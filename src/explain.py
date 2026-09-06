"""
Explainability layer -- SHAP values per flagged wallet, turned into a
human-readable reason (not just a score). No dashboard; this prints to
console and writes a CSV, per the "skip Streamlit" instruction -- you can
still hand a judge/investigator explanations.csv directly.

Run: python src/explain.py   (after train_models.py)
Output: data/processed/explanations.csv
"""
import lightgbm as lgb
import pandas as pd
import shap

import config as cfg
from train_models import FEATURE_COLUMNS

# Plain-language labels for each feature, used to build the human-readable
# reason string (SHAP gives you numbers; investigators need sentences)
FEATURE_DESCRIPTIONS = {
    "fan_in": "receives funds from an unusually high number of distinct wallets",
    "fan_out": "sends funds out to an unusually high number of distinct wallets",
    "total_btc_volume": "has moved an unusually large total BTC volume",
    "num_tx_total": "has an unusually high total transaction count",
    "avg_tx_value": "has an unusually high or low average transaction value",
    "active_lifespan_days": "has an unusual account lifespan (e.g. a short-lived 'burner' pattern)",
    "peel_chain_depth": "participates in a deep peel-chain (a classic laundering hand-off pattern)",
    "ip_reuse_score": "shares its IP address with several other wallets (possible single-operator control)",
    "burst_ratio": "shows a dormant-then-burst transaction timing pattern",
    "geo_country_diversity": "has transacted from an unusual number of different countries",
    "geo_asn_diversity": "has transacted through an unusual number of different ISPs/networks",
    "fee_btc": "pays an unusual transaction fee for its volume",
}


def build_reason(row: pd.Series, shap_row: pd.Series, top_n: int = 3) -> str:
    """Picks the top-N features pushing this wallet's score up (positive
    SHAP contribution) and turns them into one sentence."""
    positive_contribs = shap_row[shap_row > 0].sort_values(ascending=False)
    top_features = positive_contribs.head(top_n).index.tolist()

    if not top_features:
        return "No individual feature strongly drove this wallet's score up; flagged mainly by the ensemble's combined signal."

    clauses = [FEATURE_DESCRIPTIONS.get(f, f) for f in top_features]
    if len(clauses) == 1:
        joined = clauses[0]
    elif len(clauses) == 2:
        joined = f"{clauses[0]} and {clauses[1]}"
    else:
        joined = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"

    return f"Flagged because this wallet {joined}."


def main():
    df = pd.read_csv(cfg.PROCESSED_FILES["features"])
    risk = pd.read_csv(cfg.MODEL_FILES["risk_scores"])

    booster = lgb.Booster(model_file=str(cfg.MODEL_FILES["lightgbm_model"]))
    explainer = shap.TreeExplainer(booster)

    top_wallets = risk.sort_values("final_score", ascending=False).head(cfg.TOP_K)
    top_df = df[df["wallet"].isin(top_wallets["wallet"])].set_index("wallet")
    top_df = top_df.loc[top_wallets["wallet"]]  # preserve rank order

    X = top_df[FEATURE_COLUMNS]
    shap_values = explainer.shap_values(X)
    # binary classifier: shap_values may come back as a list [class0, class1]
    # or a single 2D array depending on lightgbm/shap version -- normalize
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS, index=top_df.index)

    reasons = []
    for wallet in top_df.index:
        reason = build_reason(top_df.loc[wallet], shap_df.loc[wallet])
        reasons.append(reason)

    explanations = top_wallets.copy()
    explanations["reason"] = reasons
    explanations.to_csv(cfg.MODEL_FILES["explanations"], index=False)

    print(f"[explain] wrote explanations -> {cfg.MODEL_FILES['explanations']}")
    print(f"\n--- Top {cfg.TOP_K} flagged wallets with SHAP-backed reasons ---\n")
    for _, row in explanations.iterrows():
        label_str = {1: "KNOWN ILLICIT", 2: "known licit", 3: "unlabeled"}.get(row["wallet_label"], "unlabeled")
        print(f"{row['wallet']}  (score={row['final_score']:.3f}, ground truth={label_str})")
        print(f"  -> {row['reason']}\n")


if __name__ == "__main__":
    main()
