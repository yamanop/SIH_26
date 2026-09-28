"""
ChainSentry Streamlit dashboard.

Run: streamlit run src/dashboard.py    (from the SIH_26 project, after
                                         run_day1.py + run_day2.py + explain.py)

Reads only the artifacts the pipeline already produces:
  data/processed/risk_scores.csv
  data/processed/feature_table.csv
  data/processed/explanations.csv
  data/processed/chainsentry_graph.gpickle
  data/processed/lightgbm_model.txt
  data/processed/metrics.json
Nothing here mutates the pipeline outputs -- it's read-only.
"""
import json

import lightgbm as lgb
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

import config as cfg
from train_models import FEATURE_COLUMNS

st.set_page_config(page_title="ChainSentry", layout="wide", page_icon="\U0001F50D")

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
LABEL_TEXT = {1: "Known illicit", 2: "Known licit", 3: "Unlabeled"}
LABEL_COLOR = {1: "#e74c3c", 2: "#2ecc71", 3: "#95a5a6"}


# ---------------------------------------------------------------- loaders --
@st.cache_data
def load_risk_scores():
    return pd.read_csv(cfg.MODEL_FILES["risk_scores"])


@st.cache_data
def load_feature_table():
    return pd.read_csv(cfg.PROCESSED_FILES["features"])


@st.cache_data
def load_explanations():
    path = cfg.MODEL_FILES["explanations"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_metrics():
    path = cfg.MODEL_FILES["metrics"]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_graph():
    path = cfg.PROCESSED_FILES["graph"]
    if not path.exists():
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_booster():
    return lgb.Booster(model_file=str(cfg.MODEL_FILES["lightgbm_model"]))


@st.cache_resource
def load_explainer(_booster):
    return shap.TreeExplainer(_booster)


def build_reason(shap_row: pd.Series, top_n: int = 3) -> str:
    positive = shap_row[shap_row > 0].sort_values(ascending=False)
    top = positive.head(top_n).index.tolist()
    if not top:
        return "No individual feature strongly drove this wallet's score up; flagged mainly by the ensemble's combined signal."
    clauses = [FEATURE_DESCRIPTIONS.get(f, f) for f in top]
    joined = clauses[0] if len(clauses) == 1 else (
        f"{clauses[0]} and {clauses[1]}" if len(clauses) == 2
        else f"{', '.join(clauses[:-1])}, and {clauses[-1]}"
    )
    return f"Flagged because this wallet {joined}."


# ------------------------------------------------------------- data guard --
missing = [name for name, path in cfg.MODEL_FILES.items()
           if name != "metrics" and not path.exists()]
if missing:
    st.error(
        f"Missing pipeline outputs: {missing}. Run `python run_day1.py` then "
        f"`python run_day2.py` from `src/` before starting the dashboard."
    )
    st.stop()

risk_df = load_risk_scores()
feat_df = load_feature_table()
expl_df = load_explanations()
metrics = load_metrics()
graph = load_graph()

# -------------------------------------------------------------- sidebar --
st.sidebar.title(" ChainSentry")
page = st.sidebar.radio("View", ["Overview", "Wallet Explorer", "Network Graph", "Model Insights"])
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(risk_df)} wallets scored")
if metrics:
    st.sidebar.caption(f"Precision@{metrics['k']}: {metrics['precision_at_k']:.2f}")

# ============================================================== Overview =
if page == "Overview":
    st.title("Risk Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wallets scored", f"{len(risk_df):,}")
    c2.metric("Known illicit (ground truth)", int((risk_df["wallet_label"] == 1).sum()))
    flagged = risk_df.sort_values("final_score", ascending=False).head(cfg.TOP_K)
    c3.metric(f"Top-{cfg.TOP_K} flagged, true illicit", int((flagged["wallet_label"] == 1).sum()))
    if metrics:
        c4.metric(f"Precision@{metrics['k']}", f"{metrics['precision_at_k']:.2f}")
    else:
        c4.metric("Precision@K", "n/a")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Final risk score distribution")
        fig = go.Figure()
        for label, name in LABEL_TEXT.items():
            subset = risk_df[risk_df["wallet_label"] == label]
            if len(subset):
                fig.add_trace(go.Histogram(x=subset["final_score"], name=name,
                                            marker_color=LABEL_COLOR[label], opacity=0.65))
        fig.update_layout(barmode="overlay", height=380, xaxis_title="final_score", yaxis_title="wallets")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Ground truth mix")
        counts = risk_df["wallet_label"].map(LABEL_TEXT).value_counts()
        fig2 = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                                 marker_colors=[LABEL_COLOR[k] for k, v in LABEL_TEXT.items() if v in counts.index]))
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"Top {cfg.TOP_K} riskiest wallets")
    display = flagged[["wallet", "wallet_label", "final_score", "iso_score", "lgbm_prob",
                        "cluster_risk", "network_redflags"]].copy()
    display["wallet_label"] = display["wallet_label"].map(LABEL_TEXT)
    st.dataframe(display.style.background_gradient(subset=["final_score"], cmap="Reds"),
                 use_container_width=True, height=420)

# =========================================================== Wallet view =
elif page == "Wallet Explorer":
    st.title("Wallet Explorer")

    search = st.text_input("Search wallet address")
    pool = risk_df
    if search:
        pool = pool[pool["wallet"].str.contains(search, case=False, na=False)]
    pool = pool.sort_values("final_score", ascending=False)

    if pool.empty:
        st.warning("No wallet matches that search.")
        st.stop()

    wallet = st.selectbox("Select wallet", pool["wallet"].tolist())
    row = risk_df[risk_df["wallet"] == wallet].iloc[0]

    label = int(row["wallet_label"])
    label_text = LABEL_TEXT.get(label, f"Unknown ({label})")
    label_color = LABEL_COLOR.get(label, "#7f8c8d")
    st.markdown(
        f"### `{wallet}`  &nbsp; "
        f"<span style='background-color:{label_color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:14px'>{label_text}</span>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final score", f"{row['final_score']:.3f}")
    c2.metric("IsolationForest", f"{row['iso_score']:.3f}")
    c3.metric("LightGBM prob", f"{row['lgbm_prob']:.3f}")
    c4.metric("Cluster risk", f"{row['cluster_risk']:.3f}")
    c5.metric("Network red-flags", f"{row['network_redflags']:.3f}")

    st.subheader("Why this score")
    existing_reason = expl_df.loc[expl_df["wallet"] == wallet, "reason"] if not expl_df.empty else pd.Series()
    if len(existing_reason):
        st.info(existing_reason.iloc[0])
    elif wallet in feat_df["wallet"].values:
        with st.spinner("Computing SHAP explanation..."):
            booster = load_booster()
            explainer = load_explainer(booster)
            x_row = feat_df.loc[feat_df["wallet"] == wallet, FEATURE_COLUMNS]
            shap_values = explainer.shap_values(x_row)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_row = pd.Series(shap_values[0], index=FEATURE_COLUMNS)
            st.info(build_reason(shap_row))
            st.caption("Computed live (this wallet wasn't in the top-K explanations.csv).")
    else:
        st.warning("This wallet has no feature row to explain.")

    st.subheader("Feature values")
    if wallet in feat_df["wallet"].values:
        feat_row = feat_df.loc[feat_df["wallet"] == wallet, FEATURE_COLUMNS].iloc[0]
        fig = go.Figure(go.Bar(x=feat_row.values, y=feat_row.index, orientation="h",
                                marker_color="#3498db"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================ Graph view =
elif page == "Network Graph":
    st.title("Network Graph Explorer")
    if graph is None:
        st.error("chainsentry_graph.gpickle not found.")
        st.stop()

    wallet = st.selectbox("Center on wallet", risk_df.sort_values("final_score", ascending=False)["wallet"].tolist())
    radius = st.slider("Neighborhood radius (hops)", 1, 3, 2)

    wallet_node = f"wallet::{wallet}"
    if wallet_node not in graph:
        st.warning("This wallet has no graph node (it may have been dropped as an orphan during ingest).")
        st.stop()

    ego = nx.ego_graph(graph.to_undirected(), wallet_node, radius=radius)
    st.caption(f"{ego.number_of_nodes()} nodes, {ego.number_of_edges()} edges within {radius} hop(s)")

    pos = nx.spring_layout(ego, seed=cfg.RANDOM_SEED)
    node_colors = {"wallet": "#3498db", "transaction": "#7f8c8d", "ip": "#e67e22"}
    risk_lookup = risk_df.set_index("wallet")["final_score"].to_dict()

    edge_x, edge_y = [], []
    for a, b in ego.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    node_x, node_y, colors, sizes, texts = [], [], [], [], []
    for n, data in ego.nodes(data=True):
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        ntype = data.get("type", "?")
        if ntype == "wallet":
            addr = data.get("address", "")
            score = risk_lookup.get(addr)
            colors.append("#e74c3c" if (score is not None and score > 0.7) else node_colors["wallet"])
            sizes.append(22 if n == wallet_node else 14)
            texts.append(f"wallet {addr}" + (f"<br>score={score:.3f}" if score is not None else ""))
        elif ntype == "ip":
            colors.append(node_colors["ip"])
            sizes.append(10)
            texts.append(f"ip {data.get('ip','')}<br>{data.get('country','')}")
        else:
            colors.append(node_colors["transaction"])
            sizes.append(8)
            texts.append(f"tx {data.get('txid','')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(width=1, color="#ccc"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", hovertext=texts, hoverinfo="text",
                              marker=dict(size=sizes, color=colors, line=dict(width=1, color="white"))))
    fig.update_layout(showlegend=False, height=600, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("\U0001F535 wallet (large = selected, red = high risk) \u2022 \U0001F7E0 IP \u2022 \u26AB transaction")

# ========================================================= Model insights =
elif page == "Model Insights":
    st.title("Model Insights")
    if metrics is None:
        st.warning("metrics.json not found -- rerun `python run_day2.py` with the updated train_models.py to generate it.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Precision@{metrics['k']}", f"{metrics['precision_at_k']:.3f}")
    c2.metric("F1", f"{metrics['f1']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")
    st.caption(f"Held-out test set: {metrics['n_test']} wallets ({metrics['n_test_illicit']} illicit)")

    st.subheader("LightGBM feature importance (gain)")
    imp = pd.Series(metrics["feature_importances"]).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#9b59b6"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ensemble weights")
    st.json(metrics["ensemble_weights"])
