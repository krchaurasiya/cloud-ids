import yaml

from release.controller import ROOT
from release.gates import health_check, offline_gate

POLICY = yaml.safe_load((ROOT / "release" / "policy.yaml").read_text())


def meta(recall, f1, fpr, p95=1.0):
    return {"metrics": {"val": {"recall": recall, "f1": f1, "false_positive_rate": fpr}},
            "local_latency": {"single_p95_ms": p95}}


def test_good_candidate_passes():
    ok, reasons = offline_gate(meta(0.99, 0.98, 0.005), meta(0.985, 0.97, 0.01), POLICY)
    assert ok and reasons == []


def test_weak_candidate_rejected():
    ok, reasons = offline_gate(meta(0.80, 0.70, 0.20), meta(0.99, 0.98, 0.005), POLICY)
    assert not ok and len(reasons) >= 3


def test_recall_drop_rejected():
    ok, reasons = offline_gate(meta(0.96, 0.97, 0.005), meta(0.99, 0.98, 0.005), POLICY)
    assert not ok and "recall drop" in reasons[0]


def test_slow_candidate_rejected():
    ok, reasons = offline_gate(meta(0.99, 0.98, 0.005, p95=500), meta(0.99, 0.98, 0.005), POLICY)
    assert not ok and "local p95" in reasons[0]


def test_health_check():
    assert health_check({"error_rate": 0.0, "p95_ms": 100}, 800, 0.01)[0]
    assert not health_check({"error_rate": 0.2, "p95_ms": 100}, 800, 0.01)[0]
    assert not health_check({"error_rate": 0.0, "p95_ms": 1600}, 800, 0.01)[0]
    assert not health_check({"error_rate": 1.0, "p95_ms": None}, 800, 0.01)[0]
