"""Locust load test (alternative to monitoring/loadtest.py, with a web UI and CSV output).

PowerShell:
  $env:ID_TOKEN = (gcloud auth print-identity-token)     # only for a private service
  locust -f monitoring/locustfile.py --host https://YOUR-SERVICE-URL `
         --headless -u 20 -r 5 -t 2m --csv results/locust_cloud
Results: results/locust_cloud_stats.csv (has 50% and 95% columns).
"""
import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOADS = json.loads((Path(__file__).resolve().parents[1] / "results" / "payloads.json").read_text())
TOKEN = os.getenv("ID_TOKEN", "")


class IDSUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        if TOKEN:
            self.client.headers["Authorization"] = f"Bearer {TOKEN}"

    @task(10)
    def predict(self):
        self.client.post("/predict", json={"features": random.choice(PAYLOADS)["features"]},
                         name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
