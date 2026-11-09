"""Phase 2 - Train and evaluate model versions; save versioned artefacts.

Versions
  v1  Random Forest (baseline / first stable model)
  v2  HistGradientBoosting (candidate model)
  v3  Gaussian Naive Bayes - a simple alternative that is expected to perform
      poorly on these heavy-tailed flow features. It is used to show the offline
      release gate rejecting a weak candidate. Say so in the report.

Every version is saved to models/<version>/ as
  model.joblib   - the fitted estimator
  metadata.json  - features, threshold, validation + test metrics,
                   local latency, library versions, training time

Usage:
  python -m src.train                 # trains v1 v2 v3
  python -m src.train --versions v1 v2
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB

from src.config import (ATTACK_TYPE_COL, DATA_PROCESSED, MODELS_DIR, RANDOM_STATE,
                        RESULTS_DIR, TARGET_COL)

MODEL_SPECS = {
    "v1": ("RandomForest baseline", lambda: RandomForestClassifier(
        n_estimators=100, max_depth=25, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE)),
    "v2": ("HistGradientBoosting candidate", lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1,
        class_weight="balanced", random_state=RANDOM_STATE)),
    "v3": ("GaussianNB (expected weak; gate-rejection test)", lambda: GaussianNB()),
}
THRESHOLD = 0.5


def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def per_attack_recall(test: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict:
    pred = proba >= threshold
    out = {}
    for attack, idx in test.groupby(ATTACK_TYPE_COL).groups.items():
        mask = test.index.isin(idx)
        if attack == "BENIGN":
            out[attack] = {"n": int(mask.sum()),
                           "false_alarm_rate": float(pred[mask].mean())}
        else:
            out[attack] = {"n": int(mask.sum()), "recall": float(pred[mask].mean())}
    return out


def local_latency(model, X: pd.DataFrame, n: int = 200) -> dict:
    """Single-record predict_proba latency (ms), as the API will call it."""
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)          # same setting the API uses
    rows = X.sample(n=min(n, len(X)), random_state=RANDOM_STATE).to_numpy()
    times = []
    for r in rows:
        t0 = time.perf_counter()
        model.predict_proba(r.reshape(1, -1))
        times.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    model.predict_proba(X.head(1000).to_numpy())
    batch_ms = (time.perf_counter() - t0) * 1000
    return {"single_p50_ms": float(np.percentile(times, 50)),
            "single_p95_ms": float(np.percentile(times, 95)),
            "batch_1000_ms": float(batch_ms)}


def save_confusion_png(cm: dict, title: str, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > arr.max() / 2 else "black")
    ax.set_xticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_yticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=list(MODEL_SPECS))
    args = ap.parse_args()

    features = json.loads((DATA_PROCESSED / "features.json").read_text())
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet").reset_index(drop=True)
    X_tr, y_tr = train[features], train[TARGET_COL].to_numpy()
    X_va, y_va = val[features], val[TARGET_COL].to_numpy()
    X_te, y_te = test[features], test[TARGET_COL].to_numpy()

    summary = {}
    for version in args.versions:
        name, factory = MODEL_SPECS[version]
        print(f"\n== {version}: {name}")
        model = factory()
        t0 = time.perf_counter()
        # fit on numpy so the model has no pandas feature-name dependency at serve time
        model.fit(X_tr.to_numpy(), y_tr)
        train_s = time.perf_counter() - t0

        p_va = model.predict_proba(X_va.to_numpy())[:, 1]
        p_te = model.predict_proba(X_te.to_numpy())[:, 1]
        meta = {
            "version": version,
            "name": name,
            "algorithm": type(model).__name__,
            "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in model.get_params().items()},
            "features": features,
            "threshold": THRESHOLD,
            "positive_class": "ATTACK",
            "metrics": {"val": metrics(y_va, p_va, THRESHOLD),
                        "test": metrics(y_te, p_te, THRESHOLD)},
            "per_attack_test": per_attack_recall(test, p_te, THRESHOLD),
            "local_latency": local_latency(model, X_te),
            "train_rows": int(len(train)),
            "train_seconds": round(train_s, 2),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        }
        out = MODELS_DIR / version
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib", compress=3)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        size_mb = (out / "model.joblib").stat().st_size / 1e6
        save_confusion_png(meta["metrics"]["test"]["confusion_matrix"],
                           f"{version} {name} (test)", RESULTS_DIR / f"confusion_{version}.png")

        m = meta["metrics"]["test"]
        summary[version] = {"precision": m["precision"], "recall": m["recall"],
                            "f1": m["f1"], "fpr": m["false_positive_rate"],
                            "single_p95_ms": meta["local_latency"]["single_p95_ms"],
                            "model_mb": round(size_mb, 1), "train_s": meta["train_seconds"]}
        print(json.dumps(summary[version], indent=2))

    comp_path = RESULTS_DIR / "model_comparison.csv"
    comp = pd.DataFrame(summary).T
    if comp_path.exists():                      # keep rows for versions not retrained
        old = pd.read_csv(comp_path, index_col=0)
        comp = pd.concat([old.drop(index=comp.index, errors="ignore"), comp]).sort_index()
    comp.round(4).to_csv(comp_path)
    print(f"\nSaved comparison to {RESULTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
