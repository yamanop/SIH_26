"""
ChainSentry — shared configuration.

IMPORTANT — swapping in the real Elliptic++ dataset:
The real dataset is distributed via Google Drive (not a direct-download URL),
linked from https://github.com/git-disl/EllipticPlusPlus . Download it there,
then drop the files into data/raw/ using the names below. The real repo's
exact filenames have drifted across releases, so if yours differ, just
rename them (or edit RAW_FILES below) to match:

    data/raw/txs_features.csv                 # transaction features (203K rows, 183 features)
    data/raw/txs_classes.csv                  # txId -> class (1=illicit, 2=licit, 3=unknown)
    data/raw/txs_edgelist.csv                 # txId1 -> txId2 (money flow between txs)
    data/raw/wallets_features_classes.csv     # address -> features (56) + class
    data/raw/AddrTx_edgelist.csv              # address -> txId (address sent into this tx)
    data/raw/TxAddr_edgelist.csv              # txId -> address (tx paid out to this address)

Until you have the real files, run `python src/synthetic_elliptic_data.py`
first — it generates small stand-in CSVs with the *same* column layout so
the rest of the pipeline (ingest -> features -> graph) is fully testable
today. Swapping in the real files later requires no code changes, only
replacing the CSVs in data/raw/.
"""

from pathlib import Path

# ---- paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_FILES = {
    "txs_features": DATA_RAW / "txs_features.csv",
    "txs_classes": DATA_RAW / "txs_classes.csv",
    "txs_edgelist": DATA_RAW / "txs_edgelist.csv",
    "wallets_features": DATA_RAW / "wallets_features_classes_combined.csv",  # real Elliptic++ wallet file (fixed from wallets_features_classes.csv, which was the synthetic-only stand-in)
    "addr_tx_edgelist": DATA_RAW / "AddrTx_edgelist.csv",
    "tx_addr_edgelist": DATA_RAW / "TxAddr_edgelist.csv",
    "network_layer": DATA_RAW / "network_layer_synthetic.csv",
}

PROCESSED_FILES = {
    "unified": DATA_PROCESSED / "unified_transactions.csv",
    "features": DATA_PROCESSED / "feature_table.csv",
    "graph": DATA_PROCESSED / "chainsentry_graph.gpickle",
}

# ---- subsample size for the 2-day build (Day 1 scope) -----------------
# Keep this small so every stage runs in seconds on a laptop. Bias toward
# including known-illicit transactions so the demo has real positives.
N_TRANSACTIONS = 20000
N_WALLETS = 30000
ILLICIT_FRACTION = 0.08          # inflated vs. real ~2% so Day-1 dev/testing
                                  # actually has enough positives to look at;
                                  # tune back down toward real-world skew
                                  # once you're validating final metrics.
RANDOM_SEED = 42

# ---- the 17 "named" Elliptic-style features we keep + explain --------
# (the other ~166 raw features in the real dataset are anonymous; Stage 2
# drops most of those via variance/correlation filtering instead of
# hand-listing them here)
NAMED_FEATURES = [
    "total_btc_transacted",
    "avg_btc_per_tx",
    "num_input_addresses",
    "num_output_addresses",
    "fee_btc",
    "tx_size_bytes",
    "num_inputs",
    "num_outputs",
    "in_degree",           # count of tx-tx edges pointing in
    "out_degree",          # count of tx-tx edges pointing out
    "local_time_step",     # which of the 49 Elliptic time steps
    "time_since_first_seen",
    "time_since_last_seen",
    "loop_flag",           # 1 if this tx participates in a value loop
    "self_change_ratio",   # fraction of output value returned to sender
    "max_input_value",
    "max_output_value",
]

# ---- unified schema all Stage-1 output rows conform to ----------------
UNIFIED_COLUMNS = [
    "txid", "timestamp", "src_wallet", "dst_wallet", "amount_btc", "fee_btc",
    "src_ip", "dst_ip", "src_port", "dst_port", "country", "asn",
    "tx_class",  # 1=illicit, 2=licit, 3=unknown
]

# ---- Day 2: model output paths -----------------------------------------
MODEL_FILES = {
    "risk_scores": DATA_PROCESSED / "risk_scores.csv",
    "lightgbm_model": DATA_PROCESSED / "lightgbm_model.txt",
    "explanations": DATA_PROCESSED / "explanations.csv",
    "metrics": DATA_PROCESSED / "metrics.json",
}

# ---- Day 2: ensemble fusion weights (from the PRD; heuristic, not learned) --
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.30,
    "lightgbm": 0.40,
    "cluster_risk": 0.20,
    "network_redflags": 0.10,
}

TOP_K = 20  # how many top-ranked wallets the demo/dashboard highlights
TEST_SPLIT_FRACTION = 0.30  # held-out fraction of labeled wallets for honest eval
