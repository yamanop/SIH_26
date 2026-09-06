"""
Stage 0a — generates a small, realistic STAND-IN for the Elliptic++ dataset.

Why this exists: the real Elliptic++ dataset is ~5GB and only distributed via
Google Drive (see config.py for the link), so it can't be fetched inside an
automated pipeline run. This script produces CSVs with the *identical column
layout* as the real dataset, at a scale small enough to test Stages 1-3
today. Swap this out for the real download when you have it — nothing else
in the pipeline changes.

Run: python src/synthetic_elliptic_data.py
"""
import numpy as np
import pandas as pd

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_SEED)


def make_txs_features(n_tx: int) -> pd.DataFrame:
    """203-feature-style transaction table: txId, 49 time steps, 17 named
    features (kept explainable) + a block of anonymous numeric noise
    features (standing in for the real dataset's 166 unexplained columns)."""
    txids = [f"tx_{i:07d}" for i in range(n_tx)]
    df = pd.DataFrame({"txId": txids})

    df["local_time_step"] = rng.integers(1, 50, size=n_tx)  # 49 time steps
    df["total_btc_transacted"] = rng.exponential(scale=2.0, size=n_tx).round(6)
    df["avg_btc_per_tx"] = (df["total_btc_transacted"] / rng.integers(1, 5, n_tx)).round(6)
    df["num_input_addresses"] = rng.integers(1, 6, size=n_tx)
    df["num_output_addresses"] = rng.integers(1, 6, size=n_tx)
    df["fee_btc"] = rng.exponential(scale=0.0004, size=n_tx).round(8)
    df["tx_size_bytes"] = rng.integers(200, 3000, size=n_tx)
    df["num_inputs"] = df["num_input_addresses"]
    df["num_outputs"] = df["num_output_addresses"]
    df["in_degree"] = rng.integers(0, 4, size=n_tx)
    df["out_degree"] = rng.integers(0, 4, size=n_tx)
    df["time_since_first_seen"] = rng.integers(0, 100000, size=n_tx)
    df["time_since_last_seen"] = rng.integers(0, 100000, size=n_tx)
    df["loop_flag"] = rng.integers(0, 2, size=n_tx)
    df["self_change_ratio"] = rng.uniform(0, 1, size=n_tx).round(4)
    df["max_input_value"] = df["total_btc_transacted"] * rng.uniform(0.3, 1.0, size=n_tx)
    df["max_output_value"] = df["total_btc_transacted"] * rng.uniform(0.3, 1.0, size=n_tx)

    # anonymous noise columns standing in for the real 166 unexplained
    # Elliptic features -- Stage 2 will mostly drop these
    n_anon = 40  # kept small here; real dataset has 166, we don't need that
                 # many to demo the variance/correlation filtering logic
    anon = rng.normal(size=(n_tx, n_anon))
    anon_cols = {f"anon_feat_{i}": anon[:, i] for i in range(n_anon)}
    df = pd.concat([df, pd.DataFrame(anon_cols)], axis=1)

    return df


def make_txs_classes(txids: list, illicit_fraction: float) -> pd.DataFrame:
    n = len(txids)
    n_illicit = int(n * illicit_fraction)
    n_unknown = int(n * 0.30)
    n_licit = n - n_illicit - n_unknown

    classes = np.array([1] * n_illicit + [2] * n_licit + [3] * n_unknown)
    rng.shuffle(classes)
    return pd.DataFrame({"txId": txids, "class": classes})


def make_txs_edgelist(txids: list, n_edges: int) -> pd.DataFrame:
    src = rng.choice(txids, size=n_edges)
    dst = rng.choice(txids, size=n_edges)
    keep = src != dst
    return pd.DataFrame({"txId1": src[keep], "txId2": dst[keep]}).drop_duplicates()


def make_wallets_and_edges(txids: list, n_wallets: int, illicit_fraction: float):
    wallets = [f"addr_{i:07d}" for i in range(n_wallets)]

    n_illicit = int(n_wallets * illicit_fraction)
    n_unknown = int(n_wallets * 0.40)
    n_licit = n_wallets - n_illicit - n_unknown
    w_classes = np.array([1] * n_illicit + [2] * n_licit + [3] * n_unknown)
    rng.shuffle(w_classes)

    wallets_df = pd.DataFrame({"address": wallets, "class": w_classes})
    # a handful of numeric wallet features (stand-in for the real 56)
    wallets_df["lifetime_days"] = rng.integers(1, 900, size=n_wallets)
    wallets_df["num_txs_as_sender"] = rng.integers(0, 50, size=n_wallets)
    wallets_df["num_txs_as_receiver"] = rng.integers(0, 50, size=n_wallets)

    # --- Give a subset of illicit wallets a "fan-in/fan-out burst" profile
    # so Stage 2's fan-in/fan-out and burst features actually have signal to
    # find later -- purely synthetic realism, not real behavior.
    illicit_idx = wallets_df.index[wallets_df["class"] == 1]
    burst_wallets = rng.choice(illicit_idx, size=max(1, len(illicit_idx) // 3), replace=False)
    wallets_df.loc[burst_wallets, "num_txs_as_sender"] += rng.integers(80, 300, size=len(burst_wallets))
    wallets_df.loc[burst_wallets, "num_txs_as_receiver"] += rng.integers(80, 300, size=len(burst_wallets))

    # AddrTx: wallet -> tx (this wallet was an input to this tx)
    n_addr_tx = len(txids) * 2
    addr_tx = pd.DataFrame({
        "input_address": rng.choice(wallets, size=n_addr_tx),
        "txId": rng.choice(txids, size=n_addr_tx),
    }).drop_duplicates()

    # TxAddr: tx -> wallet (this tx paid out to this wallet)
    n_tx_addr = len(txids) * 2
    tx_addr = pd.DataFrame({
        "txId": rng.choice(txids, size=n_tx_addr),
        "output_address": rng.choice(wallets, size=n_tx_addr),
    }).drop_duplicates()

    return wallets_df, addr_tx, tx_addr


def main():
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)

    txs_features = make_txs_features(cfg.N_TRANSACTIONS)
    txids = txs_features["txId"].tolist()

    txs_classes = make_txs_classes(txids, cfg.ILLICIT_FRACTION)
    txs_edgelist = make_txs_edgelist(txids, n_edges=int(cfg.N_TRANSACTIONS * 1.2))
    wallets_df, addr_tx, tx_addr = make_wallets_and_edges(
        txids, cfg.N_WALLETS, cfg.ILLICIT_FRACTION
    )

    txs_features.to_csv(cfg.RAW_FILES["txs_features"], index=False)
    txs_classes.to_csv(cfg.RAW_FILES["txs_classes"], index=False)
    txs_edgelist.to_csv(cfg.RAW_FILES["txs_edgelist"], index=False)
    wallets_df.to_csv(cfg.RAW_FILES["wallets_features"], index=False)
    addr_tx.to_csv(cfg.RAW_FILES["addr_tx_edgelist"], index=False)
    tx_addr.to_csv(cfg.RAW_FILES["tx_addr_edgelist"], index=False)

    print(f"[synthetic_elliptic_data] wrote {len(txs_features)} transactions, "
          f"{len(wallets_df)} wallets to {cfg.DATA_RAW}")
    print(txs_classes["class"].value_counts().rename("tx class counts"))
    print(wallets_df["class"].value_counts().rename("wallet class counts"))


if __name__ == "__main__":
    main()
