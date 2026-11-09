"""Tests run on tiny synthetic models so they work without the real dataset."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

FEATURES = ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Destination Port"]


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    for version, model in (("v1", RandomForestClassifier(n_estimators=10, random_state=0)),
                           ("v2", GaussianNB())):
        model.fit(X, y)
        (d / version).mkdir()
        joblib.dump(model, d / version / "model.joblib")
        (d / version / "metadata.json").write_text(json.dumps({
            "version": version, "name": version, "algorithm": type(model).__name__,
            "threshold": 0.5, "features": FEATURES, "metrics": {}, "local_latency": {},
            "trained_at": "test", "sklearn_version": "test"}))
    pd.DataFrame(X, columns=FEATURES).to_csv(d / "reference_sample.csv.gz", index=False)
    return d


def make_client(models_dir: Path, **env):
    """Fresh app import with the given environment."""
    import importlib
    from fastapi.testclient import TestClient
    base = {"MODELS_DIR": str(models_dir), "MODEL_VERSION": "v1", "STABLE_VERSION": "v1"}
    for k in list(os.environ):
        if k.startswith(("FAULT_", "INAPP_", "ADMIN_", "FALLBACK_")):
            del os.environ[k]
    os.environ.update({**base, **{k: str(v) for k, v in env.items()}})
    import app.main as main
    main = importlib.reload(main)
    return TestClient(main.app)


def record(**overrides):
    feats = {f: 0.5 for f in FEATURES}
    feats.update(overrides)
    return {"features": feats}
