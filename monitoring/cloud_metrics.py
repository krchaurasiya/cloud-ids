"""Pull Cloud Run metrics from Google Cloud Monitoring for the report.

Reads, per revision, over the last N minutes (1-minute buckets):
  run.googleapis.com/request_latencies  -> p50 and p95 (server-side, ms)
  run.googleapis.com/request_count      -> requests by response code class

One-time auth on your PC:  gcloud auth application-default login

  python -m monitoring.cloud_metrics --project YOUR_PROJECT_ID --service ids-api --minutes 60
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from google.cloud import monitoring_v3


def query(client, project, mtype, service, minutes, aligner, reducer=None, group_by=None):
    now = time.time()
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": int(now)}, "start_time": {"seconds": int(now - minutes * 60)}})
    agg = {"alignment_period": {"seconds": 60}, "per_series_aligner": aligner}
    if reducer:
        agg.update({"cross_series_reducer": reducer, "group_by_fields": group_by or []})
    return client.list_time_series(request={
        "name": f"projects/{project}",
        "filter": (f'metric.type = "{mtype}" AND resource.type = "cloud_run_revision" '
                   f'AND resource.labels.service_name = "{service}"'),
        "interval": interval,
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        "aggregation": monitoring_v3.Aggregation(agg),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--service", default="ids-api")
    ap.add_argument("--minutes", type=int, default=60)
    args = ap.parse_args()
    client = monitoring_v3.MetricServiceClient()
    A = monitoring_v3.Aggregation

    out: dict = defaultdict(dict)
    for label, aligner in (("p50_ms", A.Aligner.ALIGN_PERCENTILE_50),
                           ("p95_ms", A.Aligner.ALIGN_PERCENTILE_95)):
        for ts in query(client, args.project, "run.googleapis.com/request_latencies",
                        args.service, args.minutes, aligner):
            rev = ts.resource.labels["revision_name"]
            vals = [p.value.double_value for p in ts.points]
            out[rev][label + "_per_minute"] = [round(v, 1) for v in vals]
            out[rev][label + "_max_minute"] = round(max(vals), 1) if vals else None

    for ts in query(client, args.project, "run.googleapis.com/request_count", args.service,
                    args.minutes, A.Aligner.ALIGN_SUM, A.Reducer.REDUCE_SUM,
                    ["resource.labels.revision_name", "metric.labels.response_code_class"]):
        rev = ts.resource.labels["revision_name"]
        cls = ts.metric.labels.get("response_code_class", "?")
        out[rev].setdefault("requests_by_class", {})
        out[rev]["requests_by_class"][cls] = int(sum(p.value.int64_value for p in ts.points))

    for rev, d in out.items():
        req = d.get("requests_by_class", {})
        total = sum(req.values())
        d["error_rate_5xx"] = round(req.get("5xx", 0) / total, 4) if total else None
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
