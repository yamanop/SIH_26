"""
Stage 1 — Ingestion & Parsing.

Reads the raw blockchain-layer files (Elliptic-style CSVs) and the
network-layer file (synthetic today, real IP data later), normalizes them
onto one fixed schema (see config.UNIFIED_COLUMNS), validates types with
Pydantic, deduplicates by TXID, and aligns blockchain vs. network-layer
timestamps into one row per transaction.

Run: python src/ingest.py   (after synthetic_elliptic_data.py + ip_generator.py)
Output: data/processed/unified_transactions.csv
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator

import config as cfg


class UnifiedTransaction(BaseModel):
    """Validates one row of the unified schema. Pydantic raises on bad
    types instead of silently letting a string amount through, etc."""
    txid: str
    timestamp: datetime
    src_wallet: str
    dst_wallet: str
    amount_btc: float
    fee_btc: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    country: str
    asn: str
    tx_class: int

    @field_validator("tx_class")
    @classmethod
    def class_in_range(cls, v):
        if v not in (1, 2, 3):
            raise ValueError(f"tx_class must be 1 (illicit), 2 (licit), or 3 (unknown); got {v}")
        return v

    @field_validator("amount_btc", "fee_btc")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("amount/fee cannot be negative")
        return v


def load_and_join() -> pd.DataFrame:
    txs_features = pd.read_csv(cfg.RAW_FILES["txs_features"])
    txs_classes = pd.read_csv(cfg.RAW_FILES["txs_classes"])
    addr_tx = pd.read_csv(cfg.RAW_FILES["addr_tx_edgelist"])   # input_address, txId
    tx_addr = pd.read_csv(cfg.RAW_FILES["tx_addr_edgelist"])   # txId, output_address
    network = pd.read_csv(cfg.RAW_FILES["network_layer"])       # txId, timestamp, ips, geo...

    # one representative src/dst wallet per tx (first input, first output)
    first_input = addr_tx.drop_duplicates("txId").rename(
        columns={"input_address": "src_wallet"}
    )[["txId", "src_wallet"]]
    first_output = tx_addr.drop_duplicates("txId").rename(
        columns={"output_address": "dst_wallet"}
    )[["txId", "dst_wallet"]]

    df = txs_features.merge(txs_classes, on="txId", how="left")
    df = df.merge(first_input, on="txId", how="left")
    df = df.merge(first_output, on="txId", how="left")
    df = df.merge(network, on="txId", how="left")

    # --- normalize onto the fixed unified schema -----------------------
    unified = pd.DataFrame({
        "txid": df["txId"],
        "timestamp": df["timestamp"],
        "src_wallet": df["src_wallet"],
        "dst_wallet": df["dst_wallet"],
        "amount_btc": df["total_btc_transacted"],   # Elliptic naming -> ours
        "fee_btc": df["fee_btc"],
        "src_ip": df["src_ip"],
        "dst_ip": df["dst_ip"],
        "src_port": df["src_port"],
        "dst_port": df["dst_port"],
        "country": df["country"],
        "asn": df["asn"],
        "tx_class": df["class"],
    })

    return unified


def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Runs every row through the Pydantic model; rows that fail validation
    are dropped and counted rather than silently corrupting the pipeline."""
    good_rows = []
    n_bad = 0
    for row in df.to_dict(orient="records"):
        try:
            validated = UnifiedTransaction(**row)
            good_rows.append(validated.model_dump())
        except Exception:
            n_bad += 1
    if n_bad:
        print(f"[ingest] dropped {n_bad} rows that failed type/range validation")
    return pd.DataFrame(good_rows)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="txid", keep="first")
    after = len(df)
    if before != after:
        print(f"[ingest] removed {before - after} duplicate TXIDs")
    return df


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """In the real pipeline this would reconcile blockchain-confirmation
    time vs. network-observed time if they came from separate sources with
    separate clocks. Here they're already generated in lockstep (one row
    per tx in ip_generator.py), so this step is a no-op pass-through --
    kept as an explicit stage so the real version (with two independent
    timestamp columns to reconcile) slots in without restructuring the
    pipeline."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_and_join()
    print(f"[ingest] joined raw sources: {len(df)} rows")

    df = df.dropna(subset=["src_wallet", "dst_wallet", "timestamp"])
    print(f"[ingest] after dropping rows with missing wallet/timestamp: {len(df)} rows")

    df = validate_types(df)
    df = deduplicate(df)
    df = align_timestamps(df)

    df.to_csv(cfg.PROCESSED_FILES["unified"], index=False)
    print(f"[ingest] wrote unified table -> {cfg.PROCESSED_FILES['unified']} ({len(df)} rows)")

    # --- sanity checkpoint (per the sprint plan: eyeball before moving on)
    print("\n--- sanity check: 5 sample rows ---")
    print(df.sample(min(5, len(df)), random_state=1).to_string(index=False))
    print("\n--- sanity check: class balance ---")
    print(df["tx_class"].value_counts())
    print("\n--- sanity check: any nulls left? ---")
    null_counts = df.isnull().sum()
    remaining = null_counts[null_counts > 0]
    print(remaining if not remaining.empty else "none")


if __name__ == "__main__":
    main()
