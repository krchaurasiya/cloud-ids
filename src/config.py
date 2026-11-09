"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

LABEL_COL = "Label"            # CIC-IDS2017 label column (after stripping spaces)
TARGET_COL = "y"               # 0 = BENIGN, 1 = ATTACK
ATTACK_TYPE_COL = "attack_type"
SOURCE_COL = "source_file"

# Exact duplicate of "Fwd Header Length" in the CIC CSVs
DROP_COLS = ["Fwd Header Length.1"]

RANDOM_STATE = 42

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
