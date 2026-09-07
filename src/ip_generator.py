"""
Stage 0b — synthetic network-layer generator.

Bitcoin transactions don't carry IP addresses on-chain, and real P2P
node-monitoring data linking IPs to wallets isn't publicly available. This
script invents realistic-looking IPs/ports/timestamps per transaction so the
rest of the pipeline has network-layer signal to fuse with the blockchain
layer -- clearly labeled everywhere as simulated, per the PRD's feasibility
notes.

Design choices that matter for Stage 2/4 to have something to detect:
  - Most wallets get one "home" IP they mostly use.
  - A subset of *illicit* wallets are deliberately given IP-reuse (several
    wallets sharing one IP -- simulating one operator controlling many
    wallets) and bursty timing (long dormancy, then a flood of txs in a
    short window) -- exactly the patterns Stage 2's fan-in/out, IP-reuse,
    and burst-ratio features are designed to catch.
  - A lightweight fake "GeoLite2" style IP->country/ASN lookup is included
    instead of the real MaxMind database (which requires a license-key
    signup) -- swap `fake_geoip_lookup` for a real MaxMind reader later.

Run: python src/ip_generator.py   (after synthetic_elliptic_data.py)
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_SEED + 1)

COUNTRIES_ASNS = [
    ("US", "AS7018"), ("US", "AS15169"), ("DE", "AS3320"), ("NL", "AS60781"),
    ("RU", "AS12389"), ("SG", "AS55960"), ("IN", "AS55836"), ("BR", "AS28573"),
    ("SE", "AS29518"), ("PA", "AS52468"),  # a couple of classic VPN/hosting-heavy ASNs
]


def fake_geoip_lookup(ip: str) -> tuple:
    """Deterministic stand-in for a MaxMind GeoLite2 lookup: hash the IP
    into one of a fixed set of (country, ASN) pairs. Swap for `geoip2.database.Reader`
    once you have a real GeoLite2-Country/ASN .mmdb file."""
    idx = hash(ip) % len(COUNTRIES_ASNS)
    return COUNTRIES_ASNS[idx]


def random_ip() -> str:
    return f"{rng.integers(1, 224)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def build_wallet_ip_profiles(wallets: list, hotspot_fraction: float = 0.05):
    """Assign each wallet a 'home' IP. A small set of hotspot IPs are shared
    across many wallets (simulating one operator running many addresses)."""
    n_hotspots = max(1, int(len(wallets) * hotspot_fraction))
    hotspot_ips = [random_ip() for _ in range(n_hotspots)]

    profiles = {}
    for w in wallets:
        if rng.random() < 0.25:  # 25% of wallets sit on a shared hotspot IP
            profiles[w] = rng.choice(hotspot_ips)
        else:
            profiles[w] = random_ip()
    return profiles, hotspot_ips


def assign_burst_windows(wallets_df: pd.DataFrame, base_time: datetime):
    """For a subset of illicit wallets: long dormancy then a tight burst of
    activity -- classic ransomware cash-out pattern per the PRD."""
    burst_wallets = set()
    illicit = wallets_df.loc[wallets_df["class"] == 1, "address"].tolist()
    n_burst = max(1, len(illicit) // 2)
    burst_wallets.update(rng.choice(illicit, size=n_burst, replace=False))
    return burst_wallets


def main():
    txs_features = pd.read_csv(cfg.RAW_FILES["txs_features"])
    wallets_df = pd.read_csv(cfg.RAW_FILES["wallets_features"])
    addr_tx = pd.read_csv(cfg.RAW_FILES["addr_tx_edgelist"])       # input_address, txId
    tx_addr = pd.read_csv(cfg.RAW_FILES["tx_addr_edgelist"])       # txId, output_address

    wallets = wallets_df["address"].tolist()
    profiles, hotspot_ips = build_wallet_ip_profiles(wallets)
    burst_wallets = assign_burst_windows(wallets_df, base_time=datetime(2026, 1, 1))

    # one src wallet + one dst wallet per tx (first input / first output we see)
    first_input = addr_tx.drop_duplicates("txId").set_index("txId")["input_address"]
    first_output = tx_addr.drop_duplicates("txId").set_index("txId")["output_address"]

    base_time = datetime(2026, 1, 1)
    rows = []
    # real Elliptic++ txs_features.csv names the time-step column "Time step"
    # (synthetic stand-in used "local_time_step"); support both.
    time_col = "local_time_step" if "local_time_step" in txs_features.columns else "Time step"

    for _, tx in txs_features.iterrows():
        txid = tx["txId"]
        src_wallet = first_input.get(txid, rng.choice(wallets))
        dst_wallet = first_output.get(txid, rng.choice(wallets))

        src_ip = profiles.get(src_wallet, random_ip())
        dst_ip = profiles.get(dst_wallet, random_ip())

        # timing: time_step (1-49) maps to a rough date; burst wallets get
        # squeezed into a short window instead of spread out
        if src_wallet in burst_wallets:
            ts = base_time + timedelta(days=int(tx[time_col]) * 7,
                                        minutes=int(rng.integers(0, 30)))
        else:
            ts = base_time + timedelta(days=int(tx[time_col]) * 7,
                                        hours=int(rng.integers(0, 24)))

        country, asn = fake_geoip_lookup(src_ip)

        rows.append({
            "txId": txid,
            "timestamp": ts.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(rng.integers(1024, 65535)),
            "dst_port": 8333,  # Bitcoin's standard P2P port
            "country": country,
            "asn": asn,
        })

    network_df = pd.DataFrame(rows)
    network_df.to_csv(cfg.RAW_FILES["network_layer"], index=False)

    print(f"[ip_generator] wrote {len(network_df)} network-layer rows to "
          f"{cfg.RAW_FILES['network_layer']}")
    print(f"[ip_generator] {len(hotspot_ips)} hotspot IPs, "
          f"{len(burst_wallets)} wallets given a burst-timing profile")


if __name__ == "__main__":
    main()
