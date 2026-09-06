"""
Stage 3 — Graph Construction (NetworkX, per the 2-day plan's cut of Neo4j).

Builds a heterogeneous graph with three node types (Wallet, Transaction, IP)
and edges:
    Wallet  --SENT-->        Transaction
    Transaction --SENT-->    Wallet
    Wallet  --USED_IP-->     IP

Also ships two helper queries the dashboard/analysis stage will want:
  - shared_ip_neighbors(wallet): other wallets that used the same IP(s)
  - shortest_link_path(wallet_a, wallet_b): is there a path between two
    wallets through the tx/IP graph, and how short is it

Run: python src/graph_build.py   (after ingest.py)
Output: data/processed/chainsentry_graph.gpickle
"""
import pickle

import networkx as nx
import pandas as pd

import config as cfg


def build_graph(unified: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for row in unified.itertuples(index=False):
        wallet_src = f"wallet::{row.src_wallet}"
        wallet_dst = f"wallet::{row.dst_wallet}"
        tx_node = f"tx::{row.txid}"
        ip_src = f"ip::{row.src_ip}"
        ip_dst = f"ip::{row.dst_ip}"

        G.add_node(wallet_src, type="wallet", address=row.src_wallet)
        G.add_node(wallet_dst, type="wallet", address=row.dst_wallet)
        G.add_node(tx_node, type="transaction", txid=row.txid,
                   amount_btc=row.amount_btc, timestamp=row.timestamp,
                   tx_class=row.tx_class)
        G.add_node(ip_src, type="ip", ip=row.src_ip, country=row.country, asn=row.asn)
        G.add_node(ip_dst, type="ip", ip=row.dst_ip, country=row.country, asn=row.asn)

        G.add_edge(wallet_src, tx_node, relation="SENT", timestamp=row.timestamp)
        G.add_edge(tx_node, wallet_dst, relation="SENT", timestamp=row.timestamp)
        G.add_edge(wallet_src, ip_src, relation="USED_IP")
        G.add_edge(wallet_dst, ip_dst, relation="USED_IP")

    return G


def shared_ip_neighbors(G: nx.MultiDiGraph, wallet_address: str) -> set:
    """All other wallets that used any of the same IP(s) as this wallet."""
    wallet_node = f"wallet::{wallet_address}"
    if wallet_node not in G:
        return set()

    ips = {n for n in G.successors(wallet_node) if G.nodes[n]["type"] == "ip"}
    neighbors = set()
    for ip_node in ips:
        for candidate in G.predecessors(ip_node):
            if G.nodes[candidate]["type"] == "wallet" and candidate != wallet_node:
                neighbors.add(G.nodes[candidate]["address"])
    return neighbors


def shortest_link_path(G: nx.MultiDiGraph, wallet_a: str, wallet_b: str):
    """Shortest path between two wallets through the tx/IP graph (undirected
    view, since 'connected at all' matters more than direction for a first
    triage pass)."""
    a_node, b_node = f"wallet::{wallet_a}", f"wallet::{wallet_b}"
    if a_node not in G or b_node not in G:
        return None
    try:
        return nx.shortest_path(G.to_undirected(), a_node, b_node)
    except nx.NetworkXNoPath:
        return None


def main():
    unified = pd.read_csv(cfg.PROCESSED_FILES["unified"])
    G = build_graph(unified)

    with open(cfg.PROCESSED_FILES["graph"], "wb") as f:
        pickle.dump(G, f)

    node_type_counts = pd.Series([d["type"] for _, d in G.nodes(data=True)]).value_counts()
    print(f"[graph_build] wrote graph -> {cfg.PROCESSED_FILES['graph']}")
    print(f"[graph_build] nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print("[graph_build] node types:")
    print(node_type_counts)

    # --- sanity checkpoint: pick a wallet with high IP reuse and confirm
    # the graph actually surfaces its shared-IP neighbors
    features = pd.read_csv(cfg.PROCESSED_FILES["features"])
    top_reuse = features.sort_values("ip_reuse_score", ascending=False).iloc[0]
    wallet = top_reuse["wallet"]
    neighbors = shared_ip_neighbors(G, wallet)
    print(f"\n--- sanity check: wallet with highest IP-reuse score ---")
    print(f"wallet={wallet}  ip_reuse_score={top_reuse['ip_reuse_score']}")
    print(f"shares an IP with {len(neighbors)} other wallet(s): "
          f"{list(neighbors)[:10]}{'...' if len(neighbors) > 10 else ''}")

    if len(neighbors) > 0:
        other = next(iter(neighbors))
        path = shortest_link_path(G, wallet, other)
        print(f"\n--- sanity check: shortest path {wallet} -> {other} ---")
        print(path)


if __name__ == "__main__":
    main()
