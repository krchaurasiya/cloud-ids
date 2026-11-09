# Adaptive Cloud Intrusion Detection with Risk-Aware Model Deployment

CIC-IDS2017 → binary classifier (BENIGN / ATTACK) → FastAPI → Docker → Google Cloud Run,
with monitoring (p50/p95, errors), KS drift detection, release gates and rollback
(Cloud Run revision rollback **and** in-app model fallback).

All commands are for **Windows PowerShell**, run from the repo folder.

---

## Repo layout

| Path | What it does | Owner (proposal §12) |
|---|---|---|
| `src/data_prep.py` | Clean CIC CSVs, binary label, dedupe, leakage-aware split, reference sample | Member 1 (ML) |
| `src/train.py` | Train v1 (RandomForest), v2 (HistGradientBoosting), v3 (GaussianNB, weak); save versioned artefacts + metrics | Member 1 |
| `monitoring/drift.py` | Offline drift experiment: KS + Bonferroni, PSI, model quality per window | Member 1 |
| `app/main.py` | FastAPI: `/predict`, `/predict/batch`, `/health`, `/metadata`, `/stats`, `/drift`, `/admin/active-version`; input validation; fault injection; in-app fallback | Member 2 (API/cloud) |
| `Dockerfile`, `.gcloudignore` | Container image (all model versions baked in) | Member 2 |
| `monitoring/loadtest.py`, `monitoring/locustfile.py` | Load tests → p50/p95/p99, throughput, error rate | Member 3 (monitoring/release) |
| `monitoring/cloud_metrics.py` | Pulls Cloud Run latency/request metrics from Cloud Monitoring per revision | Member 3 |
| `release/policy.yaml` | Deployment settings + all gate thresholds | Member 3 |
| `release/gates.py`, `release/controller.py` | Offline gate → no-traffic deploy → online gate → promote → watch → rollback | Member 3 |
| `scripts/summarize_results.py` | Turns every result file into tables in `results/summary.md` | All |
| `tests/` | 14 pytest tests (API, validation, faults, fallback, gates) | All |

---

## Step 0: Install once

1. **Python 3.11** from python.org (tick "Add to PATH"). Use 3.11 specifically: the pinned
   scikit-learn must be the same on your PC and inside the container.
2. **Git**.
3. **Google Cloud CLI**: https://cloud.google.com/sdk/docs/install. Then run `gcloud init` and log in.
4. Optional: **Docker Desktop**, to test the container locally. Cloud Build does the real build, so you don't need it.

```powershell
cd C:\path\to\cloud-ids
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # only if Activate.ps1 is blocked
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1: Smoke test with fake data (2 minutes, optional)

This checks that your setup works before the big download. **Never report these numbers.**

```powershell
python -m scripts.make_synthetic_data
python -m src.data_prep --rows-per-file 0
python -m src.train
pytest -q
Remove-Item data\raw\SYNTHETIC-*.csv           # delete before using the real data
```

## Step 2: Get CIC-IDS2017

1. Open https://www.unb.ca/cic/datasets/ids-2017.html, then "Download this dataset" (it asks for a short form).
2. Download **`MachineLearningCSV.zip`** (about 230 MB; you don't need the PCAPs).
3. Extract it and copy the **8 CSV files** directly into `data\raw\`:
   `Monday-WorkingHours.pcap_ISCX.csv`, `Tuesday-…`, `Wednesday-…`, `Thursday-…WebAttacks…`,
   `Thursday-…Infilteration…`, `Friday-…Morning…`, `Friday-…PortScan…`, `Friday-…DDos…`

## Step 3: Prepare data (Phase 1)

```powershell
python -m src.data_prep --rows-per-file 50000      # 8 files x 50k = manageable subset
# all rows (needs ~8 GB RAM):  python -m src.data_prep --rows-per-file 0
```
Outputs in `data\processed\`: `train/val/test.parquet`, `features.json`, `reference.parquet`,
and **`data_report.json`** (rows dropped for inf/NaN, duplicates, conflicting labels, and the class balance per split).
Put that report in your Phase 1 section.

Leakage controls (state these in the report): duplicates are removed **before** the split, feature
vectors that appear with both labels are dropped, constant features are chosen from **train only**,
and the release gates use the **validation** split, so the test split is touched only for final numbers.

## Step 4: Train and evaluate (Phase 2)

```powershell
python -m src.train
```
For each version you get `models\vX\model.joblib` and `metadata.json` (precision, recall, F1,
FPR, ROC-AUC, confusion matrix, per-attack recall, local latency). You also get
`results\confusion_vX.png` and `results\model_comparison.csv`.

## Step 5: Run the API locally (Phase 3)

```powershell
python -m monitoring.loadtest make-payloads --n 1000         # request bodies from the test split
uvicorn app.main:app --port 8080
```
Open http://localhost:8080/docs and try `/predict`. In a **second** terminal (activate the venv again):
```powershell
python -m monitoring.loadtest run --url http://localhost:8080 --n 500 --concurrency 10 --label local
pytest -q
```
Optional Docker check:
```powershell
docker build -t ids-api .
docker run --rm -p 8080:8080 ids-api
```

## Step 6: Set up Google Cloud (Phase 4)

```powershell
$PROJECT = "your-project-id"
$REGION  = "asia-south1"
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com monitoring.googleapis.com logging.googleapis.com
gcloud artifacts repositories create ids-repo --repository-format=docker --location=$REGION
```
**Budget alert (do this now):** Console → Billing → Budgets & alerts → Create budget
(for example ₹500, alerts at 50/90/100%). Take a screenshot for the report.

Build the image in the cloud. **Train the models first**, because they are copied into the image:
```powershell
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT/ids-repo/ids-api:1.0" .
```
> If the build fails with a permission error, grant the Compute Engine default service account
> `Artifact Registry Writer`, `Logs Writer` and `Storage Object Viewer` in IAM, then retry.

Edit **`release/policy.yaml`**: set `project_id` and `image`
(`asia-south1-docker.pkg.dev/<project>/ids-repo/ids-api:1.0`).

Deploy v1 as the stable release:
```powershell
python -m release.controller deploy-stable --version v1
python -m release.controller status
```
Call it yourself. The service is private, so you need an identity token:
```powershell
$URL   = gcloud run services describe ids-api --region $REGION --format "value(status.url)"
$TOKEN = gcloud auth print-identity-token
Invoke-RestMethod -Uri "$URL/health" -Headers @{Authorization = "Bearer $TOKEN"}
$p    = (Get-Content results\payloads.json | ConvertFrom-Json)[0]
$body = @{features = $p.features} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Method Post -Uri "$URL/predict" -Headers @{Authorization = "Bearer $TOKEN"} -ContentType "application/json" -Body $body
```

## Step 7: Monitoring and load testing (Phase 5)

```powershell
python -m monitoring.loadtest run --url $URL --auth --n 1000 --concurrency 10 --label cloud_c10
python -m monitoring.loadtest run --url $URL --auth --n 1000 --concurrency 20 --label cloud_c20
```
Or use Locust (a second tool, same idea):
```powershell
$env:ID_TOKEN = gcloud auth print-identity-token
locust -f monitoring/locustfile.py --host $URL --headless -u 20 -r 5 -t 2m --csv results/locust_cloud
```
Server-side metrics from Cloud Monitoring (p50/p95 per revision, 2xx/5xx counts):
```powershell
gcloud auth application-default login      # once
python -m monitoring.cloud_metrics --project $PROJECT --service ids-api --minutes 60
```
**Screenshots:** Cloud Run → ids-api → *Metrics* tab (request count, latency, container instances),
Monitoring → Metrics Explorer (`Cloud Run Revision > Request Latencies`, 95th percentile),
Logs Explorer with `jsonPayload.message="request"` (one structured log line per prediction, with latency and model version).
Make sure the **project ID is visible** in at least one screenshot.

In the report, state the test configuration: request count, concurrency, region, CPU/memory,
min/max instances, concurrency per container (from `policy.yaml`), and whether the run included a cold start.

## Step 8: Drift detection (Phase 6)

Offline experiment (this goes in the report):
```powershell
python -m monitoring.drift --feature "Flow Duration" --factor 3 --version v1
```
This compares three windows against the training reference: **A** has no shift, **B** has Flow Duration ×3, and **C** has a 60%-attack mix.
For each it reports the KS statistic, p-value, Bonferroni-corrected α, PSI, **and** F1/recall on the same rows, so you
can show that drift and accuracy loss are different things. Outputs: `results\drift_report.json`,
`drift_*.csv`, `drift_focus_feature.png`.

Live drift on the deployed service:
```powershell
python -m monitoring.loadtest make-payloads --n 1000 --shift "Flow Duration=3" --out results/payloads_shifted.json
python -m monitoring.loadtest run --url $URL --auth --n 300 --concurrency 5 --payloads results/payloads_shifted.json --label shifted
Invoke-RestMethod -Uri "$URL/drift" -Headers @{Authorization = "Bearer $TOKEN"} | ConvertTo-Json -Depth 5
```
(`/drift` uses the recent inputs **of one container instance**. Say so as a limitation.)

## Step 9: Risk-aware promotion and rollback (Phase 7)

First fix the final thresholds in `release/policy.yaml` from your baseline numbers
(Steps 4 and 7), and write down why you chose them. Then run these scenarios **in this order**:

| # | Command | Expected decision |
|---|---|---|
| 1 | `python -m release.controller release --candidate v3 --label weak_model` | **REJECTED_OFFLINE** (nothing deployed) |
| 2 | `python -m release.controller release --candidate v2 --fault-latency-ms 1500 --label slow_release` | **REJECTED_PRECHECK** (deployed with 0% traffic, failed the p95 limit, tag removed; users never saw it) |
| 3 | `python -m release.controller release --candidate v2 --fault-error-rate 0.4 --fault-start-after-s 40 --label bad_release` | **ROLLED_BACK** (passes the pre-check, gets promoted, starts failing after 40 s, watch detects it, traffic goes back to v1) |
| 4 | `python -m release.controller inapp-demo --candidate v2 --fault-error-rate 0.5` | In-app fallback: 0 client errors, `fallback_used` responses answered by v1, then the instance auto-switches to v1 (run this **before** v2 is promoted) |
| 5 | `python -m release.controller release --candidate v2 --label good_release` | **PROMOTED** (v2 becomes the new stable release) |

Scenario 3 is the main rollback evidence. Its record in `results\release_bad_release_*.json` has
the detection time, traffic-switch time, recovery time, failed requests and the active revision afterwards.
Run it 3 times and report the mean and range. Take a screenshot of Cloud Run → *Revisions* before and after.

Manual rollback any time: `python -m release.controller rollback`.

> The faults are injected on purpose through environment variables (`FAULT_*`) only on the
> candidate revision. Describe them in the report as a controlled fault, as the proposal says.

## Step 10: Collect results for the report

```powershell
python -m scripts.summarize_results
```
This writes `results\summary.md` with tables for model quality, load tests, drift and release decisions.

## Step 11: Clean up (stop charges)

```powershell
gcloud run services delete ids-api --region $REGION --quiet
gcloud artifacts repositories delete ids-repo --location $REGION --quiet
gcloud storage rm -r "gs://$($PROJECT)_cloudbuild"
```
Screenshot the empty Cloud Run list for the report.

---

## How rollback works (for the report)

* **Revision rollback (main).** All model versions are in one image. Each Cloud Run
  revision chooses one with `MODEL_VERSION`. A candidate is deployed with `--no-traffic --tag candidate`,
  so it gets its own private URL for testing. Promotion and rollback are both
  `gcloud run services update-traffic --to-revisions <rev>=100`. Recovery counts as finished when
  10 responses in a row come back with `X-Revision` equal to the stable revision.
* **In-app fallback (secondary).** With `INAPP_FALLBACK=true`, if the active model fails the request,
  it is answered by the stable model (`fallback_used: true`). After `FALLBACK_SWITCH_THRESHOLD`
  failures in 60 s, the instance switches to the stable model. This works per container
  instance and protects single requests. Revision rollback protects the whole service.
* **Drift is only a warning.** It never triggers rollback by itself. Rollback is triggered by error rate and p95.

## Known limitations to state

CIC-IDS2017 is a 2017 lab dataset, and the incoming traffic is replayed from its test split. `/stats` and
`/drift` are per instance. Client-side latency includes your internet path to Mumbai, so also report
the Cloud Monitoring server-side numbers. v3 is a deliberately simple model, used to show the gate rejecting a candidate.
# cloud-ids
# cloud-ids
# cloud-ids
