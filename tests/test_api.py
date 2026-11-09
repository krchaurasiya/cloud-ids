from tests.conftest import FEATURES, make_client, record


def test_health_and_predict(models_dir):
    with make_client(models_dir) as c:
        h = c.get("/health").json()
        assert h["active_version"] == "v1" and set(h["loaded_versions"]) == {"v1", "v2"}
        r = c.post("/predict", json=record())
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in ("BENIGN", "ATTACK")
        assert body["model_version"] == "v1" and r.headers["X-Model-Version"] == "v1"
        assert 0 <= body["attack_probability"] <= 1


def test_validation_errors(models_dir):
    with make_client(models_dir) as c:
        missing = {"features": {FEATURES[0]: 1.0}}
        assert c.post("/predict", json=missing).status_code == 422
        extra = record(**{"Not A Feature": 1.0})
        r = c.post("/predict", json=extra)
        assert r.status_code == 422 and r.json()["detail"]["n_unexpected"] == 1
        assert c.post("/predict", json={"features": "x"}).status_code == 422
        assert c.post("/predict/batch", json={"records": []}).status_code == 422


def test_batch(models_dir):
    with make_client(models_dir) as c:
        r = c.post("/predict/batch", json={"records": [record(), record(**{"Flow Duration": 3})]})
        assert r.status_code == 200 and len(r.json()["results"]) == 2


def test_fault_without_fallback_returns_500(models_dir):
    with make_client(models_dir, MODEL_VERSION="v2", FAULT_ERROR_RATE=1) as c:
        assert c.post("/predict", json=record()).status_code == 500
        assert c.get("/stats").json()["counts"]["model_errors"] == 1


def test_latency_fault(models_dir):
    with make_client(models_dir, FAULT_EXTRA_LATENCY_MS=200) as c:
        r = c.post("/predict", json=record())
        assert float(r.headers["X-Process-Time-Ms"]) >= 200


def test_delayed_fault(models_dir):
    with make_client(models_dir, FAULT_ERROR_RATE=1, FAULT_START_AFTER_S=3600) as c:
        assert c.post("/predict", json=record()).status_code == 200


def test_inapp_fallback_and_auto_switch(models_dir):
    with make_client(models_dir, MODEL_VERSION="v2", STABLE_VERSION="v1",
                     INAPP_FALLBACK="true", FAULT_TARGET_VERSION="v2",
                     FAULT_ERROR_RATE=1, FALLBACK_SWITCH_THRESHOLD=3) as c:
        first = c.post("/predict", json=record()).json()
        assert first["fallback_used"] and first["model_version"] == "v1"
        for _ in range(3):
            c.post("/predict", json=record())
        s = c.get("/stats").json()
        assert s["auto_switched"] and s["active_version"] == "v1"
        after = c.post("/predict", json=record()).json()
        assert after["model_version"] == "v1" and not after["fallback_used"]


def test_admin_switch(models_dir):
    with make_client(models_dir, ADMIN_TOKEN="secret") as c:
        assert c.post("/admin/active-version", json={"version": "v2"}).status_code == 401
        r = c.post("/admin/active-version", json={"version": "v2"},
                   headers={"X-Admin-Token": "secret"})
        assert r.status_code == 200
        assert c.post("/predict", json=record()).json()["model_version"] == "v2"
    with make_client(models_dir) as c:
        assert c.post("/admin/active-version", json={"version": "v2"}).status_code == 403


def test_drift_endpoint(models_dir):
    with make_client(models_dir) as c:
        assert c.get("/drift").json()["status"] == "insufficient_data"
        for i in range(120):
            c.post("/predict", json=record(**{"Flow Duration": 5.0 + i * 0.01}))
        d = c.get("/drift").json()
        assert d["status"] == "drift_warning"
        assert d["top_features"][0]["feature"] in FEATURES
