"""Phase 5 - Repeatable load test that reports p50 / p95 latency, throughput, errors.

Also used by the release controller for pre-promotion and post-deploy checks.

Build request payloads from the held-out test set (run once):
  python -m monitoring.loadtest make-payloads --n 1000
  python -m monitoring.loadtest make-payloads --n 1000 --shift "Flow Duration=3" --out results/payloads_shifted.json

Run a test (local or Cloud Run; --auth fetches a Google identity token via gcloud):
  python -m monitoring.loadtest run --url http://localhost:8080 --n 500 --concurrency 10
  python -m monitoring.loadtest run --url https://ids-api-xxxx.a.run.app --auth --n 1000 --concurrency 20
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOADS = ROOT / "results" / "payloads.json"


# ------------------------------------------------------------------ payloads
def make_payloads(n: int, shift: list[str] | None, out: Path, attack_share: float | None) -> None:
    import pandas as pd
    proc = ROOT / "data" / "processed"
    features = json.loads((proc / "features.json").read_text())
    test = pd.read_parquet(proc / "test.parquet")
    if attack_share is not None:
        n_att = int(n * attack_share)
        att, ben = test[test["y"] == 1], test[test["y"] == 0]
        test = pd.concat([att.sample(n_att, replace=len(att) < n_att, random_state=1),
                          ben.sample(n - n_att, replace=len(ben) < n - n_att, random_state=1)])
    sample = test.sample(n=min(n, len(test)), random_state=1)
    X = sample[features].copy()
    for item in shift or []:
        name, factor = item.rsplit("=", 1)
        X[name] = X[name] * float(factor)
    payloads = [{"features": {k: float(v) for k, v in row.items()}, "label": int(y)}
                for row, y in zip(X.to_dict(orient="records"), sample["y"])]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payloads))
    print(f"wrote {len(payloads)} payloads to {out}")


def load_payloads(path: Path = DEFAULT_PAYLOADS) -> list[dict]:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ auth
def identity_token() -> str:
    """Google identity token for calling a private Cloud Run service."""
    gcloud = shutil.which("gcloud") or "gcloud"
    out = subprocess.run([gcloud, "auth", "print-identity-token"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


# ------------------------------------------------------------------ load test
def run_load(url: str, payloads: list[dict], n_requests: int, concurrency: int,
             token: str | None = None, timeout: float = 15.0, warmup: int = 5) -> dict:
    """Send n_requests POST /predict with a fixed number of concurrent workers."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    endpoint = url.rstrip("/") + "/predict"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    with httpx.Client(headers=headers, timeout=timeout, limits=limits) as client:
        for _ in range(warmup):                       # cold start excluded from stats
            try:
                client.post(endpoint, json={"features": random.choice(payloads)["features"]})
            except httpx.HTTPError:
                pass

        def one(_):
            p = random.choice(payloads)
            t0 = time.perf_counter()
            try:
                r = client.post(endpoint, json={"features": p["features"]})
                ms = (time.perf_counter() - t0) * 1000
                ver = r.headers.get("X-Model-Version", "")
                body = r.json() if r.status_code == 200 else {}
                correct = (body.get("prediction") == ("ATTACK" if p.get("label") else "BENIGN")
                           if body and "label" in p else None)
                return ms, r.status_code, ver, body.get("fallback_used", False), correct
            except httpx.HTTPError as exc:
                return (time.perf_counter() - t0) * 1000, f"EXC:{type(exc).__name__}", "", False, None

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, range(n_requests)))
        wall = time.perf_counter() - t_start

    lat = np.array([r[0] for r in rows])
    ok = np.array([r[1] == 200 for r in rows])
    correct = [r[4] for r in rows if r[4] is not None]
    return {
        "url": url, "n_requests": n_requests, "concurrency": concurrency,
        "duration_s": round(wall, 2),
        "throughput_rps": round(n_requests / wall, 2),
        "error_rate": round(float(1 - ok.mean()), 4),
        "errors": int((~ok).sum()),
        "status_codes": dict(collections.Counter(str(r[1]) for r in rows)),
        "p50_ms": round(float(np.percentile(lat[ok], 50)), 2) if ok.any() else None,
        "p95_ms": round(float(np.percentile(lat[ok], 95)), 2) if ok.any() else None,
        "p99_ms": round(float(np.percentile(lat[ok], 99)), 2) if ok.any() else None,
        "mean_ms": round(float(lat[ok].mean()), 2) if ok.any() else None,
        "model_versions": dict(collections.Counter(r[2] for r in rows if r[2])),
        "fallback_responses": int(sum(bool(r[3]) for r in rows)),
        "accuracy_on_labelled": round(float(np.mean(correct)), 4) if correct else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("make-payloads")
    mp.add_argument("--n", type=int, default=1000)
    mp.add_argument("--shift", nargs="*", help='e.g. "Flow Duration=3"')
    mp.add_argument("--attack-share", type=float, help="force this share of ATTACK rows")
    mp.add_argument("--out", type=Path, default=DEFAULT_PAYLOADS)
    rp = sub.add_parser("run")
    rp.add_argument("--url", required=True)
    rp.add_argument("--n", type=int, default=500)
    rp.add_argument("--concurrency", type=int, default=10)
    rp.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    rp.add_argument("--auth", action="store_true", help="send a gcloud identity token")
    rp.add_argument("--label", default="load", help="name used in the output file")
    args = ap.parse_args()

    if args.cmd == "make-payloads":
        make_payloads(args.n, args.shift, args.out, args.attack_share)
        return
    token = identity_token() if args.auth else None
    res = run_load(args.url, load_payloads(args.payloads), args.n, args.concurrency, token)
    print(json.dumps(res, indent=2))
    out = ROOT / "results" / f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
