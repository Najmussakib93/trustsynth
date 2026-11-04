"""Metadata-only inventory. Run with python -m revision.inventory."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess

from .safe_reporting import ROOT, PUBLIC, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "sdv", "ctgan",
                "rdt", "copulas", "torch", "xgboost", "gower", "matplotlib",
                "synthcity", "anonymeter", "smartnoise-synth"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    # Data file hashes are computed locally; no content is emitted.
    artifacts = []
    for path in sorted((ROOT / "data").glob("*.csv")):
        info = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
        if path.stat().st_size < 100_000_000:
            info["sha256"] = sha256(path)
        artifacts.append(info)
    source = {p.relative_to(ROOT).as_posix(): sha256(p) for p in ROOT.glob("*.py")}
    result = {
        "python": platform.python_version(), "versions": versions,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_remote": subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip(),
        "source_sha256": source, "artifacts": artifacts,
        "saved_aggregate_schema": {},
    }
    old = json.loads((ROOT / "results" / "detailed_metrics.json").read_text(encoding="utf-8"))
    # Only keys are emitted. Never dump the historical JSON wholesale.
    result["saved_aggregate_schema"] = {
        str(k): {str(d): list(metrics) if isinstance(metrics, dict) else type(metrics).__name__
                 for d, metrics in v.items()} for k, v in old.items()
    }
    write_json(PUBLIC / "audit" / "inventory.json", result)
    print(json.dumps({"versions": versions, "saved_aggregate_schema": result["saved_aggregate_schema"]}, indent=2))


if __name__ == "__main__":
    main()
