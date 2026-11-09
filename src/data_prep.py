"""Phase 1 - Dataset preparation for CIC-IDS2017 (MachineLearningCSV files).

Steps
  1. Load each CSV (optionally a random sample of rows per file).
  2. Strip column-name spaces, coerce features to numeric, drop the duplicated column.
  3. Replace +/-inf with NaN and drop non-finite rows.
  4. Binary target: BENIGN -> 0, anything else -> 1 (ATTACK).
  5. Leakage control: drop exact duplicate rows, then drop feature vectors that
     appear with BOTH labels (they cannot be learned and would leak across splits).
  6. Stratified train / validation / test split (70 / 15 / 15).
  7. Drop features that are constant in the TRAINING split only.
  8. Save splits (parquet), the feature list, a reference sample for drift
     detection, and a data report.

Usage (PowerShell, from the repo root):
  python -m src.data_prep --rows-per-file 50000
  python -m src.data_prep --files "Monday-WorkingHours.pcap_ISCX.csv" "Wednesday-workingHours.pcap_ISCX.csv"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (ATTACK_TYPE_COL, DATA_PROCESSED, DATA_RAW, DROP_COLS,
                        LABEL_COL, MODELS_DIR, RANDOM_STATE, SOURCE_COL, TARGET_COL)


def load_csvs(files: list[str] | None, rows_per_file: int | None) -> pd.DataFrame:
    paths = sorted(DATA_RAW.glob("*.csv"))
    if files:
        paths = [p for p in paths if p.name in set(files)]
    if not paths:
        raise SystemExit(f"No CSV files found in {DATA_RAW}. See README step 2.")

    frames = []
    for path in paths:
        # latin-1 because some CIC label strings contain a non-UTF-8 dash
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        if rows_per_file and len(df) > rows_per_file:
            df = df.sample(n=rows_per_file, random_state=RANDOM_STATE)
        df[SOURCE_COL] = path.name
        print(f"  loaded {path.name:55s} rows={len(df):,}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, list[str]]:
    report["rows_loaded"] = int(len(df))
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL].str.len() > 0]
    feature_cols = [c for c in df.columns if c not in (LABEL_COL, SOURCE_COL)]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feature_cols)
    report["rows_dropped_non_finite"] = int(before - len(df))
    df[feature_cols] = df[feature_cols].astype("float64")

    df[TARGET_COL] = (df[LABEL_COL] != "BENIGN").astype("int8")
    df[ATTACK_TYPE_COL] = df[LABEL_COL]
    df = df.drop(columns=[LABEL_COL])

    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + [TARGET_COL])
    report["rows_dropped_exact_duplicates"] = int(before - len(df))

    before = len(df)
    conflict = df.duplicated(subset=feature_cols, keep=False)
    df = df[~conflict]
    report["rows_dropped_conflicting_labels"] = int(before - len(df))
    report["rows_after_cleaning"] = int(len(df))
    return df.reset_index(drop=True), feature_cols


def class_counts(frame: pd.DataFrame) -> dict:
    return {"BENIGN": int((frame[TARGET_COL] == 0).sum()),
            "ATTACK": int((frame[TARGET_COL] == 1).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", help="Subset of CSV file names in data/raw")
    ap.add_argument("--rows-per-file", type=int, default=50000,
                    help="Random sample per file (0 = use all rows)")
    ap.add_argument("--reference-size", type=int, default=5000)
    args = ap.parse_args()

    report: dict = {}
    print("Loading CSVs ...")
    df = load_csvs(args.files, args.rows_per_file or None)
    df, feature_cols = clean(df, report)

    # Split BEFORE fitting anything (feature selection below uses train only)
    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET_COL],
                                   random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET_COL],
                                 random_state=RANDOM_STATE)

    constant = [c for c in feature_cols if train[c].nunique() <= 1]
    features = [c for c in feature_cols if c not in constant]
    report["constant_features_dropped"] = constant
    report["n_features"] = len(features)

    keep = features + [TARGET_COL, ATTACK_TYPE_COL, SOURCE_COL]
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[keep].to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)
        report[f"{name}_rows"] = int(len(part))
        report[f"{name}_classes"] = class_counts(part)

    report["attack_types_in_test"] = test[ATTACK_TYPE_COL].value_counts().to_dict()
    report["files"] = sorted(df[SOURCE_COL].unique().tolist())

    (DATA_PROCESSED / "features.json").write_text(json.dumps(features, indent=2))

    # Reference window for drift detection: sample of TRAINING features only.
    ref = train[features].sample(n=min(args.reference_size, len(train)),
                                 random_state=RANDOM_STATE)
    ref.to_parquet(DATA_PROCESSED / "reference.parquet", index=False)
    # A smaller copy is baked into the Docker image for the live /drift endpoint
    ref.head(2000).to_csv(MODELS_DIR / "reference_sample.csv.gz", index=False)

    (DATA_PROCESSED / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("constant_features_dropped",)}, indent=2))


if __name__ == "__main__":
    main()
