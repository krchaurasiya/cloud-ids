"""Phase 7 - Risk-aware release controller (promotion + rollback on Cloud Run).

Revision-based flow for `release`:
  1. OFFLINE GATE   candidate vs stable metrics (validation split) -> reject = nothing deployed
  2. DEPLOY         candidate as a new revision with 0% traffic, tag "candidate"
  3. ONLINE GATE    load test on the candidate's private tag URL   -> reject = tag removed,
                    users never saw it
  4. PROMOTE        100% traffic to the candidate revision
  5. WATCH          repeated small load tests on the main URL for `duration_s`
  6. ROLLBACK       if `consecutive_breaches_to_rollback` batches breach the limits:
                    100% traffic back to the stable revision, then measure recovery
                    (time until N responses in a row come from the stable revision)
  Every run is saved to results/release_<time>.json and appended to release/history.jsonl.

In-app flow (`inapp-demo`): one revision loads both models; when the active model fails,
the app answers with the stable model and, after repeated failures, switches itself.

Commands (PowerShell, repo root):
  python -m release.controller deploy-stable --version v1
  python -m release.controller offline-gate --candidate v3
  python -m release.controller release --candidate v2
  python -m release.controller release --candidate v2 --fault-latency-ms 1500 --label latency_fault
  python -m release.controller release --candidate v2 --fault-error-rate 0.4 --fault-start-after-s 40 --label delayed_errors
  python -m release.controller inapp-demo --candidate v2 --fault-error-rate 0.5
  python -m release.controller rollback
  python -m release.controller status
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from monitoring.loadtest import identity_token, load_payloads, run_load
from release import gcloud_ops as ops
from release.gates import health_check, offline_gate

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "release" / "state.json"
HISTORY = ROOT / "release" / "history.jsonl"
RESULTS = ROOT / "results"


# ------------------------------------------------------------------ helpers
def load_policy() -> dict:
    policy = yaml.safe_load((ROOT / "release" / "policy.yaml").read_text())
    if "YOUR_PROJECT_ID" in json.dumps(policy["deployment"]):
        raise SystemExit("Edit release/policy.yaml: set project_id and image first.")
    return policy


def meta(version: str) -> dict:
    return json.loads((ROOT / "models" / version / "metadata.json").read_text())


def read_state() -> dict:
    if not STATE_FILE.exists():
        raise SystemExit("No stable release recorded. Run: python -m release.controller deploy-stable --version v1")
    return json.loads(STATE_FILE.read_text())


def write_state(version: str, revision: str) -> None:
    STATE_FILE.write_text(json.dumps({"stable": {"version": version, "revision": revision},
                                      "updated": now()}, indent=2))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def suffix(version: str) -> str:
    return f"{version}-{datetime.now():%m%d-%H%M%S}"


def save_record(record: dict) -> Path:
    path = RESULTS / f"release_{record.get('label', 'run')}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    with HISTORY.open("a") as fh:
        fh.write(json.dumps({k: record.get(k) for k in
                             ("time", "label", "candidate", "stable", "decision", "reasons",
                              "rollback")}) + "\n")
    print(f"\nRecord saved: {path}")
    return path


def fault_env(args) -> dict[str, str]:
    env = {}
    if args.fault_error_rate:
        env["FAULT_ERROR_RATE"] = str(args.fault_error_rate)
    if args.fault_latency_ms:
        env["FAULT_EXTRA_LATENCY_MS"] = str(args.fault_latency_ms)
    if args.fault_start_after_s:
        env["FAULT_START_AFTER_S"] = str(args.fault_start_after_s)
    return env


def wait_for_recovery(url: str, token: str, stable_revision: str, need_ok: int,
                      payloads: list[dict], timeout_s: float = 180) -> dict:
    """Poll the main URL until `need_ok` consecutive 200s come from the stable revision."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ok_streak, sent, failed = 0, 0, 0
    t0 = time.perf_counter()
    with httpx.Client(headers=headers, timeout=15) as client:
        i = 0
        while time.perf_counter() - t0 < timeout_s:
            i += 1
            try:
                r = client.post(url.rstrip("/") + "/predict",
                                json={"features": payloads[i % len(payloads)]["features"]})
                sent += 1
                if r.status_code != 200:
                    failed += 1
                good = r.status_code == 200 and r.headers.get("X-Revision") == stable_revision
            except httpx.HTTPError:
                sent += 1
                failed += 1
                good = False
            ok_streak = ok_streak + 1 if good else 0
            if ok_streak >= need_ok:
                return {"recovered": True, "seconds": round(time.perf_counter() - t0, 2),
                        "requests_sent": sent, "failed_requests": failed}
            time.sleep(0.2)
    return {"recovered": False, "seconds": round(time.perf_counter() - t0, 2),
            "requests_sent": sent, "failed_requests": failed}


# ------------------------------------------------------------------ commands
def cmd_deploy_stable(args) -> None:
    d = load_policy()["deployment"]
    rev = ops.deploy_revision(d, suffix(args.version),
                              {"MODEL_VERSION": args.version, "STABLE_VERSION": args.version},
                              no_traffic=False)
    ops.route_all_traffic(d, rev)          # pin traffic to this revision (not "latest")
    write_state(args.version, rev)
    print(f"\nStable release: {args.version} on revision {rev}")
    print(f"URL: {ops.service_url(ops.describe(d))}")


def cmd_offline_gate(args) -> None:
    policy = load_policy()
    stable_v = args.stable or read_state()["stable"]["version"]
    passed, reasons = offline_gate(meta(args.candidate), meta(stable_v), policy)
    print(json.dumps({"candidate": args.candidate, "stable": stable_v,
                      "passed": passed, "reasons": reasons}, indent=2))


def cmd_release(args) -> None:
    policy = load_policy()
    d, on, post = policy["deployment"], policy["online_gate"], policy["post_deploy"]
    state = read_state()
    stable_v, stable_rev = state["stable"]["version"], state["stable"]["revision"]
    payloads = load_payloads()
    record = {"time": now(), "label": args.label, "candidate": args.candidate,
              "stable": {"version": stable_v, "revision": stable_rev},
              "faults_injected": fault_env(args), "policy": policy, "steps": []}

    # 1. offline gate
    print("\n[1] Offline gate")
    passed, reasons = offline_gate(meta(args.candidate), meta(stable_v), policy)
    record["offline_gate"] = {"passed": passed, "reasons": reasons}
    print("   ", "PASS" if passed else f"FAIL: {reasons}")
    if not passed and not args.skip_offline_gate:
        record.update(decision="REJECTED_OFFLINE", reasons=reasons)
        save_record(record)
        return

    # 2. deploy candidate with no traffic
    print("\n[2] Deploy candidate revision (0% traffic, tag 'candidate')")
    env = {"MODEL_VERSION": args.candidate, "STABLE_VERSION": stable_v, **fault_env(args)}
    cand_rev = ops.deploy_revision(d, suffix(args.candidate), env, no_traffic=True, tag="candidate")
    record["candidate_revision"] = cand_rev
    desc = ops.describe(d)
    cand_url, main_url = ops.tag_url(desc, "candidate"), ops.service_url(desc)
    token = None if d.get("allow_unauthenticated") else identity_token()

    # 3. online gate on the tag URL
    print(f"\n[3] Online gate on {cand_url}")
    load = run_load(cand_url, payloads, on["n_requests"], on["concurrency"], token)
    ok, reasons = health_check(load, on["max_p95_ms"], on["max_error_rate"])
    record["online_gate"] = {"passed": ok, "reasons": reasons, "load": load}
    print(f"    p50={load['p50_ms']} p95={load['p95_ms']} err={load['error_rate']}  ->",
          "PASS" if ok else f"FAIL: {reasons}")
    if not ok:
        ops.remove_tag(d, "candidate")
        record.update(decision="REJECTED_PRECHECK", reasons=reasons)
        save_record(record)
        return

    # 4. promote
    print("\n[4] Promote: 100% traffic -> candidate")
    ops.route_all_traffic(d, cand_rev)
    t_promoted = time.perf_counter()
    record["promoted_at"] = now()

    # 5. watch
    print(f"\n[5] Post-deploy watch for {post['duration_s']} s on {main_url}")
    breaches, failed_total, batches = 0, 0, []
    rollback_reasons = None
    while time.perf_counter() - t_promoted < post["duration_s"]:
        b = run_load(main_url, payloads, post["batch_requests"], post["batch_concurrency"],
                     token, warmup=0)
        ok, reasons = health_check(b, post["max_p95_ms"], post["max_error_rate"])
        failed_total += b["errors"]
        breaches = 0 if ok else breaches + 1
        t_rel = round(time.perf_counter() - t_promoted, 1)
        batches.append({"t_s": t_rel, "p95_ms": b["p95_ms"], "error_rate": b["error_rate"],
                        "ok": ok, "reasons": reasons, "versions": b["model_versions"]})
        print(f"    t={t_rel:6.1f}s p95={b['p95_ms']} err={b['error_rate']:.2f} "
              f"{'ok' if ok else 'BREACH ' + '; '.join(reasons)}")
        if breaches >= post["consecutive_breaches_to_rollback"]:
            rollback_reasons = reasons
            break
        time.sleep(post["pause_s"])
    record["watch_batches"] = batches

    if rollback_reasons is None:
        ops.remove_tag(d, "candidate")
        write_state(args.candidate, cand_rev)
        record.update(decision="PROMOTED", reasons=[],
                      failed_requests_during_watch=failed_total)
        print(f"\nPROMOTED: {args.candidate} ({cand_rev}) is the new stable release")
        save_record(record)
        return

    # 6. rollback
    print(f"\n[6] ROLLBACK to {stable_rev}")
    t_detect = time.perf_counter()
    ops.route_all_traffic(d, stable_rev)
    t_switched = time.perf_counter()
    rec = wait_for_recovery(main_url, token, stable_rev, post["recovery_ok_in_a_row"], payloads)
    ops.remove_tag(d, "candidate")
    active = [t for t in ops.traffic_table(ops.describe(d)) if t["percent"]]
    record.update(decision="ROLLED_BACK", reasons=rollback_reasons, rollback={
        "detected_after_promotion_s": round(t_detect - t_promoted, 2),
        "traffic_switch_s": round(t_switched - t_detect, 2),
        "recovery_after_switch_s": rec["seconds"],
        "total_detect_to_recovered_s": round(t_switched - t_detect + rec["seconds"], 2),
        "recovered": rec["recovered"],
        "failed_requests_during_watch": failed_total,
        "failed_requests_during_recovery": rec["failed_requests"],
        "failed_requests_total": failed_total + rec["failed_requests"],
        "active_traffic_after": active,
    })
    print(json.dumps(record["rollback"], indent=2))
    save_record(record)


def cmd_inapp_demo(args) -> None:
    """Deploy a no-traffic revision with in-app fallback and a faulty candidate model."""
    policy = load_policy()
    d = policy["deployment"]
    stable_v = read_state()["stable"]["version"]
    if args.candidate == stable_v:
        raise SystemExit(f"Candidate {args.candidate} is already the stable version; "
                         "run inapp-demo before promoting it, or pick another candidate.")
    env = {"MODEL_VERSION": args.candidate, "STABLE_VERSION": stable_v,
           "INAPP_FALLBACK": "true", "FAULT_TARGET_VERSION": args.candidate,
           "FAULT_ERROR_RATE": str(args.fault_error_rate),
           "FALLBACK_SWITCH_THRESHOLD": str(args.switch_threshold)}
    rev = ops.deploy_revision(d, suffix(args.candidate + "-inapp"), env,
                              no_traffic=True, tag="inapp")
    url = ops.tag_url(ops.describe(d), "inapp")
    token = None if d.get("allow_unauthenticated") else identity_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    load = run_load(url, load_payloads(), args.n, 5, token)
    stats = httpx.get(url + "/stats", headers=headers, timeout=15).json()
    ops.remove_tag(d, "inapp")
    record = {"time": now(), "label": "inapp_fallback", "revision": rev, "env": env,
              "load": load, "instance_stats": stats,
              "decision": "INAPP_SWITCHED" if stats.get("auto_switched") else "INAPP_FALLBACK_ONLY"}
    print(json.dumps({"load_error_rate": load["error_rate"],
                      "fallback_responses": load["fallback_responses"],
                      "versions_answering": load["model_versions"],
                      "instance_stats": stats}, indent=2))
    save_record(record)


def cmd_rollback(_args) -> None:
    d = load_policy()["deployment"]
    st = read_state()["stable"]
    t0 = time.perf_counter()
    ops.route_all_traffic(d, st["revision"])
    print(f"Traffic -> {st['revision']} ({st['version']}) in {time.perf_counter() - t0:.1f}s")


def cmd_status(_args) -> None:
    d = load_policy()["deployment"]
    desc = ops.describe(d)
    print("URL:", ops.service_url(desc))
    print(json.dumps(ops.traffic_table(desc), indent=2))
    if STATE_FILE.exists():
        print("Recorded stable:", json.loads(STATE_FILE.read_text())["stable"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy-stable")
    p.add_argument("--version", default="v1")
    p = sub.add_parser("offline-gate")
    p.add_argument("--candidate", required=True)
    p.add_argument("--stable")
    p = sub.add_parser("release")
    p.add_argument("--candidate", required=True)
    p.add_argument("--label", default="release")
    p.add_argument("--skip-offline-gate", action="store_true")
    p.add_argument("--fault-error-rate", type=float, default=0)
    p.add_argument("--fault-latency-ms", type=float, default=0)
    p.add_argument("--fault-start-after-s", type=float, default=0)
    p = sub.add_parser("inapp-demo")
    p.add_argument("--candidate", default="v2")
    p.add_argument("--fault-error-rate", type=float, default=0.5)
    p.add_argument("--switch-threshold", type=int, default=5)
    p.add_argument("--n", type=int, default=200)
    sub.add_parser("rollback")
    sub.add_parser("status")
    args = ap.parse_args()
    {"deploy-stable": cmd_deploy_stable, "offline-gate": cmd_offline_gate,
     "release": cmd_release, "inapp-demo": cmd_inapp_demo,
     "rollback": cmd_rollback, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
