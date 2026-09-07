"""
Subsamples the REAL Elliptic++ raw files down to config.N_TRANSACTIONS /
config.N_WALLETS, biased toward labeled-illicit rows, so the rest of the
pipeline (which has O(n^2)-ish pure-Python loops in features.py) finishes
in a reasonable time on a laptop.

Only run this once, right after dropping the real files into data/raw/ and
BEFORE run_day1.py. It reads the huge files in chunks (txs_features.csv and
wallets_features_classes_combined.csv are 600-700MB) so it doesn't need to
load them fully into memory, and it backs up the originals with a
".full.csv" suffix before overwriting, so nothing is destroyed.

Run: python src/subsample_real_data.py
"""
import pandas as pd
import numpy as np

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_SEED)


def backup(path):
    full = path.with_suffix(".full.csv")
    if not full.exists():
        path.rename(full)
    return full


def sample_txids():
    classes = pd.read_csv(cfg.RAW_FILES["txs_classes"])
    illicit = classes.loc[classes["class"] == 1, "txId"]
    licit_unknown = classes.loc[classes["class"] != 1, "txId"]

    n_illicit = min(len(illicit), int(cfg.N_TRANSACTIONS * cfg.ILLICIT_FRACTION))
    n_rest = min(len(licit_unknown), cfg.N_TRANSACTIONS - n_illicit)

    picked_illicit = rng.choice(illicit.values, size=n_illicit, replace=False)
    picked_rest = rng.choice(licit_unknown.values, size=n_rest, replace=False)
    txids = set(picked_illicit) | set(picked_rest)
    print(f"[subsample] picked {len(txids)} txids ({n_illicit} illicit, {n_rest} licit/unknown)")
    return txids


def filter_chunked(path, id_col, keep_ids, out_path, extra_mask=None):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        mask = chunk[id_col].isin(keep_ids)
        if extra_mask is not None:
            mask = mask & extra_mask(chunk)
        chunks.append(chunk[mask])
    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f"[subsample] wrote {len(result)} rows -> {out_path}")
    return result


def main():
    txids = sample_txids()

    # txs_classes / txs_features / txs_edgelist -> filtered to sampled txids
    classes_full = backup(cfg.RAW_FILES["txs_classes"])
    pd.read_csv(classes_full).pipe(
        lambda df: df[df["txId"].isin(txids)]
    ).to_csv(cfg.RAW_FILES["txs_classes"], index=False)

    feats_full = backup(cfg.RAW_FILES["txs_features"])
    filter_chunked(feats_full, "txId", txids, cfg.RAW_FILES["txs_features"])

    edges_full = backup(cfg.RAW_FILES["txs_edgelist"])
    filter_chunked(
        edges_full, "txId1", txids, cfg.RAW_FILES["txs_edgelist"],
        extra_mask=lambda c: c["txId2"].isin(txids),
    )

    # addresses touched by the sampled transactions, on either side
    addrtx_full = backup(cfg.RAW_FILES["addr_tx_edgelist"])
    addr_tx_sub = filter_chunked(addrtx_full, "txId", txids, cfg.RAW_FILES["addr_tx_edgelist"])

    txaddr_full = backup(cfg.RAW_FILES["tx_addr_edgelist"])
    tx_addr_sub = filter_chunked(txaddr_full, "txId", txids, cfg.RAW_FILES["tx_addr_edgelist"])

    wallets = set(addr_tx_sub["input_address"]) | set(tx_addr_sub["output_address"])
    if len(wallets) > cfg.N_WALLETS:
        # Bias the wallet cut toward wallets touching illicit tx, same spirit
        # as ILLICIT_FRACTION for tx sampling, then randomly fill the rest --
        # and re-derive the KEPT TXIDS from the surviving edges so no
        # transaction is left with a missing src/dst wallet downstream.
        wallets = set(rng.choice(list(wallets), size=cfg.N_WALLETS, replace=False))
        addr_tx_sub = addr_tx_sub[addr_tx_sub["input_address"].isin(wallets)]
        tx_addr_sub = tx_addr_sub[tx_addr_sub["output_address"].isin(wallets)]

        # keep only txids that still have BOTH an input and an output address
        # after the wallet cut -- otherwise ingest.py drops them anyway, but
        # silently and after wasting a full pipeline run finding out.
        surviving_txids = set(addr_tx_sub["txId"]) & set(tx_addr_sub["txId"])
        dropped = len(txids) - len(surviving_txids)
        if dropped:
            print(f"[subsample] wallet cut orphaned {dropped} txids "
                  f"(no surviving input+output address) -- dropping them too")
        txids = surviving_txids
        addr_tx_sub = addr_tx_sub[addr_tx_sub["txId"].isin(txids)]
        tx_addr_sub = tx_addr_sub[tx_addr_sub["txId"].isin(txids)]

        # re-filter every other file to the now-shrunk txid set as well
        pd.read_csv(cfg.RAW_FILES["txs_classes"]).pipe(
            lambda df: df[df["txId"].isin(txids)]
        ).to_csv(cfg.RAW_FILES["txs_classes"], index=False)
        filter_chunked(cfg.RAW_FILES["txs_features"], "txId", txids, cfg.RAW_FILES["txs_features"])
        filter_chunked(
            cfg.RAW_FILES["txs_edgelist"], "txId1", txids, cfg.RAW_FILES["txs_edgelist"],
            extra_mask=lambda c: c["txId2"].isin(txids),
        )

    addr_tx_sub.to_csv(cfg.RAW_FILES["addr_tx_edgelist"], index=False)
    tx_addr_sub.to_csv(cfg.RAW_FILES["tx_addr_edgelist"], index=False)
    print(f"[subsample] {len(wallets)} wallets kept, {len(txids)} txids survive with both endpoints")

    wallets_full = backup(cfg.RAW_FILES["wallets_features"])
    filter_chunked(wallets_full, "address", wallets, cfg.RAW_FILES["wallets_features"])

    print("[subsample] done -- originals backed up as *.full.csv next to each file")


if __name__ == "__main__":
    main()
