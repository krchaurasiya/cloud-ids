"""Phase 6 - Offline drift experiment (reference window vs current windows).

Windows compared against the reference (a sample of TRAINING features):
  A. no_shift        random held-out test rows             -> expect little/no drift
  B. feature_shift   same rows, chosen feature x factor     -> controlled covariate drift
  C. attack_heavy    test rows re-sampled to 60% ATTACK      -> prior / mix drift

For each window: two-sample Kolmogorov-Smirnov test per feature, statistic, p-value,
alpha = 0.05 with Bonferroni correction across features, PSI for the focus feature,
AND model quality on the same labelled rows (drift != accuracy loss).

  python -m monitoring.drift --feature "Flow Duration" --factor 3 --version v1
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TARGET_COL


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], alpha: float) -> pd.DataFrame:
    alpha_b = alpha / len(features)
    rows = []
    for f in features:
        stat, p = ks_2samp(ref[f].to_numpy(), cur[f].to_numpy())
        rows.append({"feature": f, "ks_statistic": stat, "p_value": p,
                     "reject_h0_bonferroni": p < alpha_b})
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def quality(model, X: pd.DataFrame, y: np.ndarray, thr: float) -> dict:
    pred = (model.predict_proba(X.to_numpy())[:, 1] >= thr).astype(int)
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4)}


def plot_feature(ref, windows, feature, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    allv = np.concatenate([ref[feature]] + [w[feature] for w in windows.values()])
    bins = np.histogram_bin_edges(np.log1p(np.clip(allv, 0, None)), bins=50)
    ax.hist(np.log1p(np.clip(ref[feature], 0, None)), bins=bins, alpha=0.5,
            density=True, label="reference (train)")
    for name, w in windows.items():
        ax.hist(np.log1p(np.clip(w[feature], 0, None)), bins=bins, alpha=0.4,
                density=True, histtype="stepfilled", label=name)
    ax.set_xlabel(f"log1p({feature})")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="Flow Duration")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    features = json.loads((DATA_PROCESSED / "features.json").read_text())
    if args.feature not in features:
        raise SystemExit(f"{args.feature!r} not in features.json")
    ref = pd.read_parquet(DATA_PROCESSED / "reference.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    model = joblib.load(MODELS_DIR / args.version / "model.joblib")
    thr = json.loads((MODELS_DIR / args.version / "metadata.json").read_text())["threshold"]

    n = min(args.window, len(test))
    a = test.sample(n=n, random_state=RANDOM_STATE)
    b = a.copy()
    b[args.feature] = b[args.feature] * args.factor
    n_att = int(n * 0.6)
    att, ben = test[test[TARGET_COL] == 1], test[test[TARGET_COL] == 0]
    c = pd.concat([att.sample(n_att, replace=len(att) < n_att, random_state=1),
                   ben.sample(n - n_att, replace=len(ben) < n - n_att, random_state=1)])

    windows = {"A_no_shift": a, f"B_{args.feature}_x{args.factor:g}": b, "C_attack_heavy_60pct": c}
    report = {"reference_n": len(ref), "window_n": n, "alpha": args.alpha,
              "alpha_bonferroni": args.alpha / len(features), "n_features": len(features),
              "focus_feature": args.feature, "model_version": args.version, "windows": {}}
    for name, w in windows.items():
        tab = ks_table(ref, w, features, args.alpha)
        tab.to_csv(RESULTS_DIR / f"drift_{name}.csv", index=False)
        focus = tab[tab.feature == args.feature].iloc[0]
        report["windows"][name] = {
            "features_drifted": int(tab.reject_h0_bonferroni.sum()),
            "focus_feature": {"ks_statistic": round(float(focus.ks_statistic), 4),
                              "p_value": float(focus.p_value),
                              "reject_h0": bool(focus.reject_h0_bonferroni),
                              "psi": round(psi(ref[args.feature].to_numpy(),
                                               w[args.feature].to_numpy()), 4)},
            "top5": tab.head(5)[["feature", "ks_statistic", "p_value"]]
                       .round({"ks_statistic": 4}).to_dict(orient="records"),
            "attack_share": round(float(w[TARGET_COL].mean()), 3),
            "model_quality": quality(model, w[features], w[TARGET_COL].to_numpy(), thr),
        }
    plot_feature(ref, {k: v for k, v in windows.items() if not k.startswith("C")},
                 args.feature, RESULTS_DIR / "drift_focus_feature.png")
    (RESULTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"Reference n={len(ref)}  window n={n}  alpha={args.alpha} "
          f"(Bonferroni {report['alpha_bonferroni']:.2e})\n")
    print(f"{'window':32s} {'#drifted':>8s} {'KS':>7s} {'p-value':>10s} {'PSI':>7s} {'F1':>6s} {'recall':>7s}")
    for name, r in report["windows"].items():
        f = r["focus_feature"]
        print(f"{name:32s} {r['features_drifted']:8d} {f['ks_statistic']:7.4f} "
              f"{f['p_value']:10.2e} {f['psi']:7.3f} {r['model_quality']['f1']:6.3f} "
              f"{r['model_quality']['recall']:7.3f}")
    print("\nSaved results/drift_report.json, drift_*.csv, drift_focus_feature.png")


if __name__ == "__main__":
    main()
