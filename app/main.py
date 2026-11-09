"""Phase 3 - FastAPI prediction service.

Endpoints
  GET  /health                 liveness + active/stable version + Cloud Run revision
  GET  /metadata               metrics and feature list of the active model
  POST /predict                one flow record  -> BENIGN / ATTACK
  POST /predict/batch          up to 100 records
  GET  /stats                  per-instance counters (requests, errors, fallbacks)
  GET  /drift                  KS test: recent inputs on THIS instance vs reference
  POST /admin/active-version   in-app model switch (needs X-Admin-Token)

Configuration (environment variables; Cloud Run revisions differ only by these)
  MODEL_VERSION          active model version            (default v1)
  STABLE_VERSION         fallback / known-good version    (default v1)
  LOAD_VERSIONS          comma list to load (default: every folder in models/)
  INAPP_FALLBACK         "true" -> if the active model fails, answer with the
                         stable model; after FALLBACK_SWITCH_THRESHOLD failures in
                         60 s, switch this instance to the stable model
  FALLBACK_SWITCH_THRESHOLD  default 5
  ADMIN_TOKEN            enables /admin/active-version when set
  Fault injection (for the rollback experiments ONLY):
  FAULT_ERROR_RATE       0..1 probability a prediction raises an error
  FAULT_EXTRA_LATENCY_MS extra delay added to each prediction
  FAULT_START_AFTER_S    fault starts this many seconds after the container starts
  FAULT_TARGET_VERSION   only that model version is faulty (empty = any)
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import random
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
REVISION = os.getenv("K_REVISION", "local")          # set automatically by Cloud Run
MAX_BATCH = 100
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "1000"))


def _env_bool(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes")


class Settings:
    def __init__(self) -> None:
        self.active = os.getenv("MODEL_VERSION", "v1")
        self.stable = os.getenv("STABLE_VERSION", "v1")
        self.inapp_fallback = _env_bool("INAPP_FALLBACK")
        self.switch_threshold = int(os.getenv("FALLBACK_SWITCH_THRESHOLD", "5"))
        self.admin_token = os.getenv("ADMIN_TOKEN", "")
        self.fault_error_rate = float(os.getenv("FAULT_ERROR_RATE", "0"))
        self.fault_latency_ms = float(os.getenv("FAULT_EXTRA_LATENCY_MS", "0"))
        self.fault_start_after_s = float(os.getenv("FAULT_START_AFTER_S", "0"))
        self.fault_target = os.getenv("FAULT_TARGET_VERSION", "")


# --------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """Cloud Run turns JSON lines on stdout into structured log entries."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"severity": record.levelname, "message": record.getMessage(),
                 "revision": REVISION}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


log = logging.getLogger("ids")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
log.handlers = [_handler]
log.setLevel(logging.INFO)
log.propagate = False


# --------------------------------------------------------------------------- state
class ModelFault(RuntimeError):
    pass


class State:
    def __init__(self) -> None:
        self.settings = Settings()
        self.models: dict[str, object] = {}
        self.meta: dict[str, dict] = {}
        self.features: list[str] = []
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.counts = collections.Counter()
        self.fallback_times: collections.deque = collections.deque()
        self.recent = collections.deque(maxlen=DRIFT_WINDOW)
        self.reference: np.ndarray | None = None
        self.auto_switched = False

    def load(self) -> None:
        wanted = os.getenv("LOAD_VERSIONS")
        versions = ([v.strip() for v in wanted.split(",") if v.strip()] if wanted else
                    sorted(p.parent.name for p in MODELS_DIR.glob("*/metadata.json")))
        for v in versions:
            meta = json.loads((MODELS_DIR / v / "metadata.json").read_text())
            model = joblib.load(MODELS_DIR / v / "model.joblib")
            if hasattr(model, "n_jobs"):
                model.set_params(n_jobs=1)     # one thread per request is faster for 1 row
            self.models[v], self.meta[v] = model, meta
        s = self.settings
        for v in (s.active, s.stable):
            if v not in self.models:
                raise RuntimeError(f"Model version {v!r} not found in {MODELS_DIR}")
        self.features = self.meta[s.active]["features"]
        for v, m in self.meta.items():
            if m["features"] != self.features:
                raise RuntimeError(f"Feature list of {v} differs from {s.active}")
        ref_path = MODELS_DIR / "reference_sample.csv.gz"
        if ref_path.exists():
            import pandas as pd
            self.reference = pd.read_csv(ref_path)[self.features].to_numpy()
        log.info("models loaded", extra={"extra_fields": {
            "loaded": list(self.models), "active": s.active, "stable": s.stable,
            "inapp_fallback": s.inapp_fallback, "fault_error_rate": s.fault_error_rate,
            "fault_latency_ms": s.fault_latency_ms, "fault_target": s.fault_target}})


STATE = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.load()
    yield


app = FastAPI(title="Adaptive Cloud IDS", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- schemas
class FlowRecord(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature name -> value, using the names from GET /metadata")


class BatchRequest(BaseModel):
    records: list[FlowRecord] = Field(..., min_length=1, max_length=MAX_BATCH)


class SwitchRequest(BaseModel):
    version: str


# --------------------------------------------------------------------------- helpers
def to_matrix(records: list[FlowRecord]) -> np.ndarray:
    expected = set(STATE.features)
    rows = []
    for i, rec in enumerate(records):
        keys = set(rec.features)
        missing, extra = sorted(expected - keys), sorted(keys - expected)
        if missing or extra:
            raise HTTPException(422, detail={
                "record": i, "missing_features": missing[:10], "n_missing": len(missing),
                "unexpected_features": extra[:10], "n_unexpected": len(extra)})
        row = [rec.features[f] for f in STATE.features]
        bad = [f for f, v in zip(STATE.features, row) if not math.isfinite(v)]
        if bad:
            raise HTTPException(422, detail={"record": i, "non_finite_features": bad[:10]})
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def fault_applies(version: str) -> bool:
    s = STATE.settings
    if s.fault_target and s.fault_target != version:
        return False
    return time.monotonic() - STATE.started >= s.fault_start_after_s


def run_model(version: str, X: np.ndarray) -> np.ndarray:
    s = STATE.settings
    if fault_applies(version):
        if s.fault_latency_ms > 0:
            time.sleep(s.fault_latency_ms / 1000)
        if s.fault_error_rate > 0 and random.random() < s.fault_error_rate:
            raise ModelFault(f"injected fault in model {version}")
    return STATE.models[version].predict_proba(X)[:, 1]


def register_fallback() -> None:
    """Count failures of the active model; switch this instance after too many."""
    s = STATE.settings
    now = time.monotonic()
    with STATE.lock:
        STATE.counts["fallbacks"] += 1
        STATE.fallback_times.append(now)
        while STATE.fallback_times and now - STATE.fallback_times[0] > 60:
            STATE.fallback_times.popleft()
        if len(STATE.fallback_times) >= s.switch_threshold and s.active != s.stable:
            log.warning("in-app auto switch to stable", extra={"extra_fields": {
                "from": s.active, "to": s.stable, "failures_60s": len(STATE.fallback_times)}})
            s.active = s.stable
            STATE.auto_switched = True


def predict_matrix(X: np.ndarray) -> tuple[np.ndarray, str, bool]:
    s = STATE.settings
    active = s.active
    try:
        proba = run_model(active, X)
        used, fallback = active, False
    except ModelFault as exc:
        if not (s.inapp_fallback and active != s.stable):
            with STATE.lock:
                STATE.counts["model_errors"] += 1
            log.error("prediction failed", extra={"extra_fields": {
                "model_version": active, "error": str(exc)}})
            raise HTTPException(500, detail=f"prediction failed ({active})") from exc
        register_fallback()
        proba = run_model(s.stable, X)
        used, fallback = s.stable, True
    with STATE.lock:
        STATE.counts[f"predictions_{used}"] += len(X)
        STATE.recent.extend(X)
    return proba, used, fallback


def result(p: float, version: str) -> dict:
    thr = STATE.meta[version]["threshold"]
    return {"prediction": "ATTACK" if p >= thr else "BENIGN",
            "attack_probability": round(float(p), 6), "threshold": thr}


# --------------------------------------------------------------------------- middleware
@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:                       # unexpected crash -> still log it
        status = 500
        response = JSONResponse({"detail": "internal error"}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    response.headers["X-Revision"] = REVISION
    with STATE.lock:
        STATE.counts["requests"] += 1
        if status >= 500:
            STATE.counts["errors_5xx"] += 1
    if request.url.path.startswith("/predict"):
        log.info("request", extra={"extra_fields": {
            "path": request.url.path, "status": status, "latency_ms": round(ms, 2),
            "model_version": response.headers.get("X-Model-Version", "")}})
    return response


# --------------------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"service": "Adaptive Cloud IDS", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    s = STATE.settings
    return {"status": "ok", "active_version": s.active, "stable_version": s.stable,
            "loaded_versions": list(STATE.models), "revision": REVISION,
            "uptime_s": round(time.monotonic() - STATE.started, 1)}


@app.get("/metadata")
def metadata():
    m = STATE.meta[STATE.settings.active]
    return {k: m[k] for k in ("version", "name", "algorithm", "threshold", "features",
                              "metrics", "local_latency", "trained_at", "sklearn_version")}


@app.post("/predict")
def predict(record: FlowRecord):
    X = to_matrix([record])
    proba, used, fallback = predict_matrix(X)
    body = {**result(proba[0], used), "model_version": used,
            "fallback_used": fallback, "revision": REVISION}
    return JSONResponse(body, headers={"X-Model-Version": used})


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    X = to_matrix(batch.records)
    proba, used, fallback = predict_matrix(X)
    body = {"model_version": used, "fallback_used": fallback, "revision": REVISION,
            "results": [result(p, used) for p in proba]}
    return JSONResponse(body, headers={"X-Model-Version": used})


@app.get("/stats")
def stats():
    s = STATE.settings
    with STATE.lock:
        counts = dict(STATE.counts)
    return {"revision": REVISION, "active_version": s.active, "stable_version": s.stable,
            "auto_switched": STATE.auto_switched, "inapp_fallback": s.inapp_fallback,
            "counts": counts, "drift_buffer": len(STATE.recent),
            "note": "Counters are per container instance."}


@app.get("/drift")
def drift(alpha: float = 0.05, min_samples: int = 100, top: int = 10):
    """Two-sample KS test per feature; Bonferroni-corrected significance."""
    from scipy.stats import ks_2samp
    if STATE.reference is None:
        raise HTTPException(404, "reference_sample.csv.gz not in image")
    with STATE.lock:
        current = np.array(STATE.recent)
    if len(current) < min_samples:
        return {"status": "insufficient_data", "current_n": len(current),
                "min_samples": min_samples}
    alpha_corr = alpha / len(STATE.features)
    rows = []
    for j, f in enumerate(STATE.features):
        stat, p = ks_2samp(STATE.reference[:, j], current[:, j])
        rows.append({"feature": f, "ks_statistic": round(float(stat), 4),
                     "p_value": float(p), "drift": bool(p < alpha_corr)})
    rows.sort(key=lambda r: r["ks_statistic"], reverse=True)
    n_drift = sum(r["drift"] for r in rows)
    return {"status": "drift_warning" if n_drift else "no_drift",
            "reference_n": len(STATE.reference), "current_n": len(current),
            "alpha": alpha, "alpha_bonferroni": alpha_corr,
            "features_drifted": n_drift, "top_features": rows[:top],
            "note": "Drift is a warning, not proof that accuracy dropped."}


@app.post("/admin/active-version")
def switch_version(req: SwitchRequest, x_admin_token: str = Header(default="")):
    s = STATE.settings
    if not s.admin_token:
        raise HTTPException(403, "admin endpoint disabled (ADMIN_TOKEN not set)")
    if x_admin_token != s.admin_token:
        raise HTTPException(401, "bad admin token")
    if req.version not in STATE.models:
        raise HTTPException(404, f"version {req.version} not loaded")
    old, s.active = s.active, req.version
    log.warning("manual in-app switch", extra={"extra_fields": {"from": old, "to": req.version}})
    return {"previous": old, "active_version": s.active, "revision": REVISION,
            "note": "Applies to this container instance only."}
