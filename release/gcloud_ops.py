"""Thin wrappers around the gcloud CLI for Cloud Run revisions and traffic."""
from __future__ import annotations

import json
import shutil
import subprocess

GCLOUD = shutil.which("gcloud") or "gcloud"     # resolves gcloud.cmd on Windows


def gcloud(args: list[str]) -> str:
    cmd = [GCLOUD, *args]
    print("  $ gcloud " + " ".join(args))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gcloud failed ({res.returncode}):\n{res.stderr.strip()}")
    return res.stdout


def _base(d: dict) -> list[str]:
    return ["--region", d["region"], "--project", d["project_id"]]


def describe(d: dict) -> dict:
    return json.loads(gcloud(["run", "services", "describe", d["service"], *_base(d),
                              "--format", "json"]))


def service_exists(d: dict) -> bool:
    try:
        describe(d)
        return True
    except RuntimeError:
        return False


def deploy_revision(d: dict, suffix: str, env: dict[str, str], *,
                    no_traffic: bool, tag: str | None = None) -> str:
    """Deploy the image as a new revision; returns the revision name."""
    env_str = ",".join(f"{k}={v}" for k, v in env.items())
    args = ["run", "deploy", d["service"], "--image", d["image"], *_base(d),
            "--revision-suffix", suffix, "--set-env-vars", env_str,
            "--memory", str(d["memory"]), "--cpu", str(d["cpu"]),
            "--min-instances", str(d["min_instances"]),
            "--max-instances", str(d["max_instances"]),
            "--concurrency", str(d["concurrency"]), "--quiet"]
    args.append("--allow-unauthenticated" if d.get("allow_unauthenticated")
                else "--no-allow-unauthenticated")
    if no_traffic:
        args.append("--no-traffic")
    if tag:
        args += ["--tag", tag]
    gcloud(args)
    return f"{d['service']}-{suffix}"


def route_all_traffic(d: dict, revision: str) -> None:
    gcloud(["run", "services", "update-traffic", d["service"], *_base(d),
            "--to-revisions", f"{revision}=100", "--quiet"])


def remove_tag(d: dict, tag: str) -> None:
    try:
        gcloud(["run", "services", "update-traffic", d["service"], *_base(d),
                "--remove-tags", tag, "--quiet"])
    except RuntimeError as exc:
        print(f"  (could not remove tag {tag}: {exc})")


def service_url(desc: dict) -> str:
    return desc["status"]["url"]


def tag_url(desc: dict, tag: str) -> str:
    for t in desc["status"].get("traffic", []):
        if t.get("tag") == tag:
            return t["url"]
    raise RuntimeError(f"tag {tag!r} not found on service")


def traffic_table(desc: dict) -> list[dict]:
    return [{"revision": t.get("revisionName", "LATEST"), "percent": t.get("percent", 0),
             "tag": t.get("tag", ""), "url": t.get("url", "")}
            for t in desc["status"].get("traffic", [])]
