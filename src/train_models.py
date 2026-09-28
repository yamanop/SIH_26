"""
Stage 4 — Model Ensemble ("AI Detective Team"), minus the GCN stretch goal
and minus Node2Vec+HDBSCAN (replaced with a lighter graph-connected-
-components clustering, per the 2-day plan's cut list).

Three detectives feed one weighted final score:
  1. Isolation Forest   -- unsupervised anomaly score (no labels needed)
  2. LightGBM           -- supervised, class-weighted, trained on labeled
                            wallets only, scored on every wallet
  3. Cluster risk        -- wallets connected via a shared IP inherit the
                            illicit rate of their connected component
                            ("guilt by association", the lightweight stand-in
                            for Node2Vec+HDBSCAN)
  + a network red-flags term built directly from existing features
    (ip_reuse_score, burst_ratio, geo_asn_diversity)

Honesty note on evaluation: LightGBM is trained on a 70% split of the
*labeled* wallets only. Precision@K/F1 are computed on the held-out 30% --
never on rows the model was trained on. The final risk_scores.csv still
uses this train-only model to score every wallet (including the test
split and the unlabeled wallets) -- that's normal for a real deployment,
but it means the numbers on the labeled test wallets in that file are a
fair preview of real-world performance, while training-split wallets are
not (the model has seen their labels).

Run: python src/train_models.py   (after run_day1.py)
Output: data/processed/risk_scores.csv, data/processed/lightgbm_model.txt
"""
import pickle
import json

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

import config as cfg

FEATURE_COLUMNS = [
    "fan_in", "fan_out", "total_btc_volume", "num_tx_total", "avg_tx_value",
    "active_lifespan_days", "peel_chain_depth", "ip_reuse_score",
    "burst_ratio", "geo_country_diversity", "geo_asn_diversity", "fee_btc",
]


def min_max(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def run_isolation_forest(df: pd.DataFrame) -> pd.Series:
    X = df[FEATURE_COLUMNS].values
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=cfg.RANDOM_SEED)
    model.fit(X)
    # decision_function: higher = more normal. Flip + normalize so higher = more anomalous.
    raw = -model.decision_function(X)
    return min_max(pd.Series(raw, index=df.index))


def run_lightgbm(df: pd.DataFrame):
    """Trains on labeled wallets only (label 1=illicit, 2=licit; label 3 or
    0/unlabeled is excluded from training). Returns (probabilities for every
    wallet, held-out test metrics dict)."""
    labeled = df[df["wallet_label"].isin([1, 2])].copy()
    labeled["target"] = (labeled["wallet_label"] == 1).astype(int)

    X = labeled[FEATURE_COLUMNS]
    y = labeled["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SPLIT_FRACTION, stratify=y, random_state=cfg.RANDOM_SEED
    )

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.RANDOM_SEED,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    # --- held-out evaluation (never seen during training) ---------------
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    k = min(cfg.TOP_K, len(X_test))
    top_k_idx = np.argsort(-test_probs)[:k]
    precision_at_k = y_test.values[top_k_idx].mean() if k > 0 else float("nan")

    metrics = {
        "precision_at_k": precision_at_k,
        "k": k,
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "report": classification_report(y_test, test_preds, zero_division=0),
        "n_test": len(y_test),
        "n_test_illicit": int(y_test.sum()),
    }

    # --- score every wallet (train-only model, so test/unlabeled rows are
    # a fair preview; training rows are not, per the module docstring) ----
    all_probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    model.booster_.save_model(str(cfg.MODEL_FILES["lightgbm_model"]))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    metrics["feature_importances"] = importances

    return pd.Series(all_probs, index=df.index), metrics


def cluster_risk_from_graph(df: pd.DataFrame) -> pd.Series:
    """Connected components over the wallet<->IP subgraph: wallets that
    share an IP end up in the same component. Each wallet inherits the
    illicit rate among *other* known-labeled wallets in its component --
    the lightweight stand-in for Node2Vec+HDBSCAN clustering.

    Leave-one-out is deliberate: a wallet's own label must never count
    towards its own cluster-risk score, or a labeled wallet with no shared
    IP at all (a singleton component) would trivially get risk=1.0 just by
    being illicit itself -- that's direct label leakage, not guilt by
    association. Only genuinely *other* members' labels count."""
    with open(cfg.PROCESSED_FILES["graph"], "rb") as f:
        G = pickle.load(f)

    wallet_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "wallet"]
    ip_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "ip"]

    H = nx.Graph()
    H.add_nodes_from(wallet_nodes)
    for ip in ip_nodes:
        wallets_on_ip = [n for n in G.predecessors(ip) if G.nodes[n]["type"] == "wallet"]
        for i in range(len(wallets_on_ip)):
            for j in range(i + 1, len(wallets_on_ip)):
                H.add_edge(wallets_on_ip[i], wallets_on_ip[j])

    address_to_label = df.set_index("wallet")["wallet_label"].to_dict()

    risk_by_wallet = {}
    for component in nx.connected_components(H):
        addresses = [n.replace("wallet::", "") for n in component]
        labels = {a: address_to_label.get(a) for a in addresses}
        n_illicit_total = sum(1 for l in labels.values() if l == 1)
        n_labeled_total = sum(1 for l in labels.values() if l in (1, 2))

        for a in addresses:
            own_label = labels[a]
            if own_label in (1, 2):
                # leave-one-out: strip this wallet's own vote
                n_illicit = n_illicit_total - (1 if own_label == 1 else 0)
                n_labeled = n_labeled_total - 1
            else:
                n_illicit, n_labeled = n_illicit_total, n_labeled_total
            risk_by_wallet[a] = (n_illicit / n_labeled) if n_labeled > 0 else 0.0

    return df["wallet"].map(risk_by_wallet).fillna(0.0)


def network_redflags(df: pd.DataFrame) -> pd.Series:
    """Combines existing network-layer features into one red-flag term:
    normalized IP-reuse + burst ratio (already 0-1) + normalized ASN
    diversity, averaged."""
    ip_reuse_norm = min_max(df["ip_reuse_score"])
    burst = df["burst_ratio"].clip(0, 1)
    asn_norm = min_max(df["geo_asn_diversity"])
    return ((ip_reuse_norm + burst + asn_norm) / 3.0)


def main():
    df = pd.read_csv(cfg.PROCESSED_FILES["features"])
    print(f"[train_models] loaded {len(df)} wallets, "
          f"{df['wallet_label'].isin([1, 2]).sum()} labeled (illicit/licit)")

    print("\n--- Detective 1: Isolation Forest ---")
    iso_score = run_isolation_forest(df)
    print(f"anomaly score range: {iso_score.min():.3f} - {iso_score.max():.3f}")

    print("\n--- Detective 2: LightGBM (held-out evaluation) ---")
    lgbm_prob, metrics = run_lightgbm(df)
    print(f"held-out test set: {metrics['n_test']} wallets "
          f"({metrics['n_test_illicit']} illicit)")
    print(f"Precision@{metrics['k']}: {metrics['precision_at_k']:.3f}")
    print(f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  "
          f"Recall: {metrics['recall']:.3f}")
    print(metrics["report"])
    print("Feature importances (LightGBM, gain-based):")
    print(metrics["feature_importances"].to_string())
    if metrics["precision_at_k"] < 0.2:
        print(f"\n[WARNING] Precision@{metrics['k']} on the held-out set is low. On this "
              f"synthetic stand-in dataset that usually means the injected illicit-wallet "
              f"signal (burst timing / IP reuse in ip_generator.py) isn't strong enough "
              f"relative to the random noise for LightGBM to reliably separate classes on "
              f"data it hasn't seen. This is a property of the synthetic data, not "
              f"necessarily the pipeline logic -- re-check against the real Elliptic++ "
              f"data before drawing conclusions about the actual approach.")

    print("--- Detective 3: cluster risk (shared-IP connected components) ---")
    cluster_risk = cluster_risk_from_graph(df)
    print(f"cluster risk range: {cluster_risk.min():.3f} - {cluster_risk.max():.3f}, "
          f"{(cluster_risk > 0).sum()} wallets in a component with at least one known label")

    print("\n--- Network red-flags term ---")
    redflags = network_redflags(df)

    print("\n--- Fusing into final ensemble score ---")
    w = cfg.ENSEMBLE_WEIGHTS
    final_score = (
        w["isolation_forest"] * iso_score +
        w["lightgbm"] * lgbm_prob +
        w["cluster_risk"] * cluster_risk +
        w["network_redflags"] * redflags
    )

    out = df[["wallet", "wallet_label"]].copy()
    out["iso_score"] = iso_score
    out["lgbm_prob"] = lgbm_prob
    out["cluster_risk"] = cluster_risk
    out["network_redflags"] = redflags
    out["final_score"] = final_score
    out = out.sort_values("final_score", ascending=False)

    out.to_csv(cfg.MODEL_FILES["risk_scores"], index=False)
    print(f"[train_models] wrote risk scores -> {cfg.MODEL_FILES['risk_scores']}")

    # dashboard-friendly metrics dump (the DataFrame/report text inside
    # `metrics` isn't JSON-safe as-is, so pull out the plain values)
    metrics_out = {
        "n_test": metrics["n_test"],
        "n_test_illicit": metrics["n_test_illicit"],
        "k": metrics["k"],
        "precision_at_k": metrics["precision_at_k"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "feature_importances": metrics["feature_importances"].to_dict(),
        "ensemble_weights": cfg.ENSEMBLE_WEIGHTS,
    }
    with open(cfg.MODEL_FILES["metrics"], "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[train_models] wrote metrics -> {cfg.MODEL_FILES['metrics']}")

    # --- sanity checkpoint --------------------------------------------
    print(f"\n--- sanity check: top {cfg.TOP_K} highest-risk wallets ---")
    print(out.head(cfg.TOP_K).to_string(index=False))

    print("\n--- sanity check: does the ensemble actually separate labels? ---")
    print(out.groupby("wallet_label")["final_score"].agg(["mean", "count"]))

    top_k_labels = out.head(cfg.TOP_K)["wallet_label"]
    hit_rate = (top_k_labels == 1).mean()
    print(f"\nOf the top {cfg.TOP_K} flagged wallets, {hit_rate:.0%} are known-illicit "
          f"(this is Precision@{cfg.TOP_K} on the *final ensemble score*, across all "
          f"wallets -- compare against the LightGBM-only Precision@K above)")


if __name__ == "__main__":
    main()
