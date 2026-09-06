# ChainSentry — Day 1 + Day 2 Pipeline (no dashboard)

Implements the 2-day build plan minus the Streamlit dashboard:

- **Day 1**: subsample data → synthetic IP generator → ingestion → feature
  engineering → graph construction.
- **Day 2**: model ensemble (Isolation Forest + LightGBM + graph-based
  cluster risk) → SHAP-backed plain-English explanations for the
  top-ranked wallets.

Runs end to end on synthetic stand-in data out of the box; swap in the
real Elliptic++ dataset when you have it, with no code changes.

## Quick start

```bash
pip install -r requirements.txt
cd src
python run_day1.py
python run_day2.py
```

Day 1 runs five steps and prints a sanity checkpoint after each one (a few
seconds on a laptop, at the default subsample size — 3,000 transactions /
5,000 wallets, see `N_TRANSACTIONS` / `N_WALLETS` in `src/config.py`). Day 2
trains the ensemble, prints held-out evaluation metrics, and writes ranked
risk scores + explanations.

## What each stage does

| Script | Stage | Output |
|---|---|---|
| `synthetic_elliptic_data.py` | 0a | `data/raw/txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`, `wallets_features_classes.csv`, `AddrTx_edgelist.csv`, `TxAddr_edgelist.csv` |
| `ip_generator.py` | 0b | `data/raw/network_layer_synthetic.csv` |
| `ingest.py` | 1 | `data/processed/unified_transactions.csv` — one clean row per transaction |
| `features.py` | 2 | `data/processed/feature_table.csv` — one row per wallet, ~14 features |
| `graph_build.py` | 3 | `data/processed/chainsentry_graph.gpickle` — heterogeneous NetworkX graph |

`run_day1.py` runs all five in order and stops immediately if any stage
fails, so a broken upstream file can't silently corrupt a downstream one.

## Swapping in the real Elliptic++ dataset

The real dataset (203K transactions, 822K wallets) is only distributed via
Google Drive, linked from
[git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus)
— it can't be auto-downloaded from a script. To use it:

1. Download it from the Google Drive link in that repo.
2. Rename/place the files into `data/raw/` matching the names in
   `src/config.py` → `RAW_FILES` (the real repo's filenames have drifted
   across releases — if yours don't match, either rename the files or
   edit the dict).
3. Subsample if needed — at full scale (822K wallets), the peel-chain-depth
   and burst-ratio loops in `features.py` are written for clarity, not
   raw speed, and will be slow. For a 2-day timeline, keep sampling down
   to a few thousand wallets (bias toward including labeled-illicit ones)
   rather than trying to run the full dataset before your deadline.
4. Run `python run_day1.py` again — it detects real data automatically
   (skips regenerating synthetic transactions, but still regenerates the
   network layer, since that's simulated either way) and re-runs Stages
   1–3 against it.
5. Get a real MaxMind GeoLite2 `.mmdb` file (free account) and swap it in
   for `fake_geoip_lookup()` in `ip_generator.py` if you want real
   IP→country/ASN lookups instead of the hashed stand-in.

## What's deliberately cut for the 2-day timeline

- Full 60–90 feature set → cut to ~14 highest-signal features (see
  `features.py` docstring).
- Variance/correlation filtering of the anonymous Elliptic columns →
  skipped; only the named features are aggregated.
- Neo4j → NetworkX only.
- Real network-layer data → synthetic generator (clearly labeled
  everywhere in code/output as simulated, consistent with the PRD's
  feasibility section).

## Before starting Day 2 (modeling)

Don't move on until:
- `unified_transactions.csv` sample rows look correct (run `ingest.py`
  directly to see the 5-row sample + null check).
- `feature_table.csv` shows illicit wallets (`wallet_label == 1`) with
  visibly different stats than licit ones for at least fan-in/out,
  ip_reuse_score, or burst_ratio (`features.py`'s last printout) — if they
  look identical, the synthetic generator's illicit-wallet signal
  injection (in `ip_generator.py`) may need strengthening before models
  have anything to learn from.
- `graph_build.py`'s shared-IP sanity check returns a non-empty neighbor
  list and a real shortest path — confirms the graph can actually answer
  the "who else used this IP" query the whole cross-layer-fusion pitch
  depends on.

## Day 2: how the ensemble is scored and evaluated

`train_models.py` trains LightGBM on a 70% split of *labeled* wallets only
and evaluates Precision@K / F1 on the held-out 30% it never saw during
training — that's the only honest way to know if the model generalizes.
The final `risk_scores.csv` still scores every wallet (including
unlabeled ones) using that same train-only model, which is normal for
real deployment.

The cluster-risk term (Detective 3, standing in for Node2Vec+HDBSCAN) uses
leave-one-out: a wallet's own label never counts toward its own cluster
score, only *other* wallets in its shared-IP component — otherwise an
isolated labeled wallet would trivially get a perfect risk score just from
its own ground truth, which is leakage, not a real signal.

**What to actually check when you run it:**
1. `train_models.py`'s printed `classification_report` and `Precision@K`
   are the honest, held-out numbers — this is the metric to quote to
   judges, not the "top 20 across all wallets" number further down (that
   one mixes in wallets the model was trained on, so it's structurally
   optimistic).
2. If `Precision@K` on the held-out set comes back near 0, the script
   prints a warning — on the synthetic stand-in data this usually means
   the injected illicit-wallet pattern is too weak relative to random
   noise for the model to reliably generalize. Check the printed feature
   importances: if `ip_reuse_score` and `burst_ratio` (the features
   `ip_generator.py` deliberately engineered signal into) rank near the
   bottom, that's the tell — strengthen the injected pattern in
   `ip_generator.py`, or treat it as expected until you swap in the real
   Elliptic++ data, where the illicit signal is genuine rather than
   synthetic.
3. `out.groupby("wallet_label")["final_score"]` (printed near the end)
   should show a materially higher mean for label `1` (illicit) than for
   `2` (licit) — a small but real gap is normal and expected; if it's
   completely flat, something upstream broke.
4. `explanations.csv` — read a few reasons for wallets you already know
   are `KNOWN ILLICIT` vs `known licit` in the printed ground-truth
   column. The reasons should feel like genuinely different stories, not
   the same three features repeated for every wallet regardless of label
   — if every wallet gets an identical-sounding reason, the SHAP
   explanations aren't adding real differentiation yet.
