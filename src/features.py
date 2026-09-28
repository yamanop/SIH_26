"""
Stage 2 — Feature Engineering.

Builds the ~15-20 highest-signal features called for in the 2-day sprint
plan (the full PRD's 60-90 feature version does deeper variance/correlation
filtering on the anonymous Elliptic columns -- skipped here to save time,
per the plan's explicit cuts). Three families:

  1. Wallet-level : fan-in/out, total volume, active lifespan, avg tx value
  2. Transaction-level : peel-chain depth (simplified)
  3. Network-level : IP reuse count, burst ratio, geo/ASN diversity

Run: python src/features.py   (after ingest.py)
Output: data/processed/feature_table.csv -- one row per wallet
"""
import numpy as np
import pandas as pd

import config as cfg


def wallet_level_features(unified: pd.DataFrame) -> pd.DataFrame:
    sent = unified.groupby("src_wallet").agg(
        total_btc_sent=("amount_btc", "sum"),
        num_tx_sent=("txid", "count"),
        fan_out=("dst_wallet", "nunique"),
        first_seen_sent=("timestamp", "min"),
        last_seen_sent=("timestamp", "max"),
    ).reset_index().rename(columns={"src_wallet": "wallet"})

    received = unified.groupby("dst_wallet").agg(
        total_btc_received=("amount_btc", "sum"),
        num_tx_received=("txid", "count"),
        fan_in=("src_wallet", "nunique"),
        first_seen_received=("timestamp", "min"),
        last_seen_received=("timestamp", "max"),
    ).reset_index().rename(columns={"dst_wallet": "wallet"})

    wallets = pd.merge(sent, received, on="wallet", how="outer").fillna({
        "total_btc_sent": 0, "num_tx_sent": 0, "fan_out": 0,
        "total_btc_received": 0, "num_tx_received": 0, "fan_in": 0,
    })

    wallets["total_btc_volume"] = wallets["total_btc_sent"] + wallets["total_btc_received"]
    wallets["num_tx_total"] = wallets["num_tx_sent"] + wallets["num_tx_received"]
    wallets["avg_tx_value"] = (wallets["total_btc_volume"] /
                                wallets["num_tx_total"].replace(0, np.nan)).fillna(0)

    all_seen = pd.concat([
        wallets[["wallet", "first_seen_sent", "last_seen_sent"]].rename(
            columns={"first_seen_sent": "first_seen", "last_seen_sent": "last_seen"}),
        wallets[["wallet", "first_seen_received", "last_seen_received"]].rename(
            columns={"first_seen_received": "first_seen", "last_seen_received": "last_seen"}),
    ])
    lifespan = all_seen.groupby("wallet").agg(
        active_first_seen=("first_seen", "min"),
        active_last_seen=("last_seen", "max"),
    ).reset_index()
    lifespan["active_first_seen"] = pd.to_datetime(lifespan["active_first_seen"])
    lifespan["active_last_seen"] = pd.to_datetime(lifespan["active_last_seen"])
    lifespan["active_lifespan_days"] = (
        lifespan["active_last_seen"] - lifespan["active_first_seen"]
    ).dt.total_seconds() / 86400.0

    wallets = wallets.merge(lifespan[["wallet", "active_lifespan_days"]], on="wallet", how="left")

    return wallets[[
        "wallet", "fan_in", "fan_out", "total_btc_volume", "num_tx_total",
        "avg_tx_value", "active_lifespan_days",
    ]]


def peel_chain_depth(unified: pd.DataFrame) -> pd.DataFrame:
    """Simplified peel-chain detector: walks src_wallet -> dst_wallet edges
    and measures, for each wallet, the longest chain of decreasing-amount
    hand-offs starting there within a short time window. A full version
    would use the tx-tx edgelist directly; this wallet-graph approximation
    is enough for a 2-day demo."""
    edges = unified[["src_wallet", "dst_wallet", "amount_btc", "timestamp"]].copy()
    edges["timestamp"] = pd.to_datetime(edges["timestamp"])
    edges = edges.sort_values("timestamp")

    # adjacency: wallet -> list of (dst, amount, time), sorted by time
    adjacency = {}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(row.src_wallet, []).append(
            (row.dst_wallet, row.amount_btc, row.timestamp)
        )

    depth_cache = {}

    def chain_depth(wallet, incoming_amount, incoming_time, visited):
        if wallet in visited or wallet not in adjacency:
            return 0
        visited = visited | {wallet}
        best = 0
        for dst, amount, ts in adjacency[wallet]:
            # "peeling": amount roughly <= incoming, and happens after
            if amount <= incoming_amount * 1.05 and ts >= incoming_time:
                best = max(best, 1 + chain_depth(dst, amount, ts, visited))
        return best

    rows = []
    for wallet, outs in adjacency.items():
        if wallet in depth_cache:
            continue
        max_amount = max(a for _, a, _ in outs)
        earliest = min(t for _, _, t in outs)
        d = chain_depth(wallet, max_amount, earliest, frozenset())
        depth_cache[wallet] = d
        rows.append({"wallet": wallet, "peel_chain_depth": d})

    return pd.DataFrame(rows)


def network_level_features(unified: pd.DataFrame) -> pd.DataFrame:
    # --- IP reuse: how many distinct wallets share each IP a wallet used
    wallet_ip = unified[["src_wallet", "src_ip"]].drop_duplicates().rename(
        columns={"src_wallet": "wallet", "src_ip": "ip"}
    )
    ip_user_counts = wallet_ip.groupby("ip")["wallet"].nunique().rename("ip_shared_with_n_wallets")
    wallet_ip = wallet_ip.merge(ip_user_counts, on="ip", how="left")
    ip_reuse = wallet_ip.groupby("wallet")["ip_shared_with_n_wallets"].max().reset_index().rename(
        columns={"ip_shared_with_n_wallets": "ip_reuse_score"}
    )
    # ip_reuse_score = 1 means this wallet's IP(s) are unique to it;
    # higher means it shares an IP with more other wallets (possible
    # same-operator signal)

    # --- burst ratio: max transactions in any single rolling 24h window,
    # divided by total transactions (close to 1 => bursty, close to
    # 1/num_days => spread out evenly)
    tx_times = unified[["src_wallet", "timestamp"]].copy()
    tx_times["timestamp"] = pd.to_datetime(tx_times["timestamp"])

    burst_rows = []
    for wallet, grp in tx_times.groupby("src_wallet"):
        times = grp["timestamp"].sort_values().tolist()
        if len(times) <= 1:
            burst_rows.append({"wallet": wallet, "burst_ratio": 0.0})
            continue
        # count max transactions falling inside any 24h window
        max_in_window = 0
        for t in times:
            window_count = sum(1 for other in times if 0 <= (other - t).total_seconds() <= 86400)
            max_in_window = max(max_in_window, window_count)
        burst_rows.append({"wallet": wallet, "burst_ratio": round(max_in_window / len(times), 4)})
    burst_df = pd.DataFrame(burst_rows)

    # --- geo/ASN diversity: how many distinct countries/ASNs this wallet's
    # transactions were observed from
    geo = unified.groupby("src_wallet").agg(
        geo_country_diversity=("country", "nunique"),
        geo_asn_diversity=("asn", "nunique"),
    ).reset_index().rename(columns={"src_wallet": "wallet"})

    out = ip_reuse.merge(burst_df, on="wallet", how="outer")
    out = out.merge(geo, on="wallet", how="outer")
    return out.fillna(0)


def main():
    unified = pd.read_csv(cfg.PROCESSED_FILES["unified"])

    wallet_feats = wallet_level_features(unified)
    print(f"[features] wallet-level features: {wallet_feats.shape}")

    peel_feats = peel_chain_depth(unified)
    print(f"[features] peel-chain depth computed for {len(peel_feats)} wallets")

    net_feats = network_level_features(unified)
    print(f"[features] network-level features: {net_feats.shape}")

    # --- bring in the 17 named Elliptic-style tx features, aggregated to
    # wallet level (mean across each wallet's transactions as sender)
    named = [c for c in cfg.NAMED_FEATURES if c in unified.columns]
    # unified doesn't carry every named feature directly (some, like
    # in_degree, live on the tx table) -- only aggregate what's present here
    # to keep this runnable against the synthetic stand-in; when you swap
    # in the real txs_features.csv, re-join before this step to bring in
    # the rest.
    if named:
        tx_level = unified.groupby("src_wallet")[named].mean().reset_index().rename(
            columns={"src_wallet": "wallet"}
        )
    else:
        tx_level = pd.DataFrame({"wallet": unified["src_wallet"].unique()})

    # --- ground-truth label per wallet, for training/validation later:
    # majority class among this wallet's own transactions as sender
    try:
        wallet_classes = pd.read_csv(
            cfg.RAW_FILES["wallets_features"], usecols=["address", "class"]
        ).rename(columns={"address": "wallet", "class": "wallet_label"})
        wallet_classes = wallet_classes.drop_duplicates("wallet")
    except (FileNotFoundError, ValueError):
        wallet_classes = pd.DataFrame(columns=["wallet", "wallet_label"])

    tx_derived_labels = unified.groupby("src_wallet")["tx_class"].agg(
        lambda s: s.value_counts().idxmax()
    ).reset_index().rename(columns={"src_wallet": "wallet", "tx_class": "wallet_label"})

    labels = pd.concat([wallet_classes, tx_derived_labels]).drop_duplicates("wallet", keep="first")

    feature_table = wallet_feats.merge(peel_feats, on="wallet", how="left")
    feature_table = feature_table.merge(net_feats, on="wallet", how="left")
    feature_table = feature_table.merge(tx_level, on="wallet", how="left")
    feature_table = feature_table.merge(labels, on="wallet", how="left")
    feature_table["wallet_label"] = feature_table["wallet_label"].fillna(3)
    feature_table = feature_table.fillna(0)

    feature_table.to_csv(cfg.PROCESSED_FILES["features"], index=False)
    print(f"[features] wrote feature table -> {cfg.PROCESSED_FILES['features']} "
          f"({feature_table.shape[0]} wallets, {feature_table.shape[1]} columns)")

    # --- sanity checkpoint
    print("\n--- sanity check: feature summary ---")
    print(feature_table.describe().T[["mean", "min", "max"]])
    print("\n--- sanity check: label distribution ---")
    print(feature_table["wallet_label"].value_counts())
    print("\n--- sanity check: do known-illicit wallets look different? ---")
    print(feature_table.groupby("wallet_label")[
        ["fan_in", "fan_out", "ip_reuse_score", "burst_ratio", "geo_asn_diversity"]
    ].mean())


if __name__ == "__main__":
    main()
