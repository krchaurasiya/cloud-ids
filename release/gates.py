"""Release decision rules (pure functions, unit-tested in tests/test_gates.py)."""
from __future__ import annotations


def offline_gate(candidate: dict, stable: dict, policy: dict) -> tuple[bool, list[str]]:
    """Compare candidate vs stable model metadata on the validation split."""
    g = policy["offline_gate"]
    c, s = candidate["metrics"]["val"], stable["metrics"]["val"]
    reasons = []
    if c["recall"] < g["min_recall"]:
        reasons.append(f"recall {c['recall']:.4f} < min {g['min_recall']}")
    if s["recall"] - c["recall"] > g["max_recall_drop_vs_stable"]:
        reasons.append(f"recall drop vs stable {s['recall'] - c['recall']:.4f} "
                       f"> {g['max_recall_drop_vs_stable']}")
    if c["f1"] < g["min_f1"]:
        reasons.append(f"F1 {c['f1']:.4f} < min {g['min_f1']}")
    if c["false_positive_rate"] > g["max_false_positive_rate"]:
        reasons.append(f"FPR {c['false_positive_rate']:.4f} > max {g['max_false_positive_rate']}")
    lat = candidate.get("local_latency", {}).get("single_p95_ms")
    if lat is not None and lat > g["max_local_single_p95_ms"]:
        reasons.append(f"local p95 {lat:.1f} ms > {g['max_local_single_p95_ms']} ms")
    return (not reasons), reasons


def health_check(load: dict, max_p95_ms: float, max_error_rate: float) -> tuple[bool, list[str]]:
    """Judge one load-test result (from monitoring.loadtest.run_load)."""
    reasons = []
    if load["error_rate"] > max_error_rate:
        reasons.append(f"error rate {load['error_rate']:.3f} > {max_error_rate}")
    if load["p95_ms"] is None:
        reasons.append("no successful responses")
    elif load["p95_ms"] > max_p95_ms:
        reasons.append(f"p95 {load['p95_ms']:.0f} ms > {max_p95_ms} ms")
    return (not reasons), reasons
