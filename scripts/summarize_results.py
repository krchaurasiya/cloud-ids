"""Collect every result file into results/summary.md (tables to paste into the report).

  python -m scripts.summarize_results
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, d=4):
    return round(x, d) if isinstance(x, float) else x


def main() -> None:
    parts = ["# Results summary\n"]

    rows = []
    for meta_path in sorted(MODELS.glob("*/metadata.json")):
        m = json.loads(meta_path.read_text())
        t = m["metrics"]["test"]
        rows.append([m["version"], m["name"], f(t["precision"]), f(t["recall"]), f(t["f1"]),
                     f(t["false_positive_rate"]), f(t["roc_auc"]),
                     f(m["local_latency"]["single_p50_ms"], 2),
                     f(m["local_latency"]["single_p95_ms"], 2)])
    parts += ["## Model quality (held-out test split)\n",
              table(["Version", "Model", "Precision", "Recall", "F1", "FPR", "ROC-AUC",
                     "Local p50 ms", "Local p95 ms"], rows), ""]

    rows = []
    for p in sorted(RES.glob("*.json")):
        if p.name.startswith(("release_", "drift_", "payloads")):
            continue
        r = json.loads(p.read_text())
        if "p95_ms" in r:
            rows.append([p.stem, r["url"], r["n_requests"], r["concurrency"], r["p50_ms"],
                         r["p95_ms"], r["throughput_rps"], r["error_rate"]])
    if rows:
        parts += ["## Load tests\n", table(["Run", "URL", "Requests", "Concurrency", "p50 ms",
                                            "p95 ms", "RPS", "Error rate"], rows), ""]

    drift = RES / "drift_report.json"
    if drift.exists():
        d = json.loads(drift.read_text())
        rows = [[name, w["features_drifted"], w["focus_feature"]["ks_statistic"],
                 f"{w['focus_feature']['p_value']:.2e}", w["focus_feature"]["reject_h0"],
                 w["focus_feature"]["psi"], w["model_quality"]["f1"], w["model_quality"]["recall"]]
                for name, w in d["windows"].items()]
        parts += [f"## Drift (KS, alpha={d['alpha']}, Bonferroni {d['alpha_bonferroni']:.2e}, "
                  f"reference n={d['reference_n']}, window n={d['window_n']}, "
                  f"focus feature: {d['focus_feature']})\n",
                  table(["Window", "# features drifted", "KS (focus)", "p-value", "Reject H0",
                         "PSI", "F1", "Recall"], rows), ""]

    rows = []
    for p in sorted(RES.glob("release_*.json")):
        r = json.loads(p.read_text())
        rb = r.get("rollback") or {}
        rows.append([r.get("label"), r.get("candidate"), r.get("decision"),
                     "; ".join(r.get("reasons") or []), rb.get("detected_after_promotion_s"),
                     rb.get("total_detect_to_recovered_s"), rb.get("failed_requests_total"),
                     (rb.get("active_traffic_after") or [{}])[0].get("revision")])
    if rows:
        parts += ["## Release decisions\n",
                  table(["Run", "Candidate", "Decision", "Reasons", "Detected after (s)",
                         "Detect->recovered (s)", "Failed requests", "Active after"], rows), ""]

    out = RES / "summary.md"
    out.write_text("\n".join(parts))
    print(out.read_text())


if __name__ == "__main__":
    main()
