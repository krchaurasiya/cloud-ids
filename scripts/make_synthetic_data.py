"""Generate FAKE CSVs with the CIC-IDS2017 column layout.

For smoke-testing the pipeline only (CI, or before the real download finishes).
Never report results from this data.

  python -m scripts.make_synthetic_data --rows 6000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import DATA_RAW

# Header exactly as in MachineLearningCSV (note the leading spaces)
CIC_COLUMNS = [
    " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Max",
    " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std",
    "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean",
    " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean",
    " Flow IAT Std", " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean",
    " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean",
    " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
    " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length",
    "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length", " Max Packet Length",
    " Packet Length Mean", " Packet Length Std", " Packet Length Variance", "FIN Flag Count",
    " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count",
    " URG Flag Count", " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio",
    " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size",
    " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes",
    " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward",
    " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward",
    "Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std",
    " Idle Max", " Idle Min", " Label",
]
ATTACKS = ["DoS Hulk", "DDoS", "PortScan", "FTP-Patator", "SSH-Patator"]


def make_file(rows: int, attack_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    n_feat = len(CIC_COLUMNS) - 1
    is_attack = rng.random(rows) < attack_ratio
    X = rng.lognormal(mean=3, sigma=1.5, size=(rows, n_feat))
    # make attacks separable-but-noisy on a few features
    X[is_attack, 1] *= rng.uniform(0.05, 0.4, is_attack.sum())     # Flow Duration
    X[is_attack, 2] *= rng.uniform(2, 6, is_attack.sum())          # Total Fwd Packets
    X[is_attack, 15] *= rng.uniform(3, 10, is_attack.sum())        # Flow Packets/s
    X[:, 0] = np.where(is_attack, rng.choice([80, 21, 22], rows), rng.integers(1, 65535, rows))
    X[:, 30:34] = 0                                                # constant flag columns
    X[:, 55] = X[:, 34]                                            # duplicated header col
    df = pd.DataFrame(X, columns=CIC_COLUMNS[:-1])
    df[" Label"] = np.where(is_attack, rng.choice(ATTACKS, rows), "BENIGN")
    df.iloc[:5, 14] = np.inf                                       # CIC has inf values
    return pd.concat([df, df.head(20)], ignore_index=True)         # and duplicates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for i, ratio in enumerate([0.0, 0.3, 0.5]):
        path = DATA_RAW / f"SYNTHETIC-{i}.csv"
        make_file(args.rows, ratio, rng).to_csv(path, index=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
