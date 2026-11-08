#!/usr/bin/env python3
"""Recover missing PhysioNet/CinC 2019 files from the public S3 mirror.

The PhysioNet open bucket is publicly readable, so credentialed data access is
not bypassed: the bucket only exposes the open-access Challenge 2019 training
partition. This script reads the missing-file manifest produced by
``scripts/audit_dataset_fast.py``, downloads each file with retries, and skips
files that already exist with non-zero size.

Usage:
    python scripts/recover_missing_files.py --dataset training_setB
    python scripts/recover_missing_files.py --dataset training_setB --workers 16
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "download"
BASE_URL = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training"


def _fetch(url: str, target: Path, retries: int) -> tuple[str, bool, str]:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response body")
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(target)
            return target.name, True, ""
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            if attempt == retries:
                return target.name, False, str(exc)
            time.sleep(2 * attempt)
    return target.name, False, "unreachable"


def _needs_download(path: Path) -> bool:
    return not path.exists() or path.stat().st_size == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="training_setB")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Missing-file manifest (default: data/reports/download/missing_files_<dataset>.txt)",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    dataset = args.dataset
    manifest = args.manifest or (
        REPORTS_DIR / f"missing_files_{dataset.replace('training_', '')}.txt"
    )
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2

    names = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    dataset_dir = RAW_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pending = [name for name in names if _needs_download(dataset_dir / name)]
    print(f"Dataset: {dataset}")
    print(f"Manifest entries: {len(names)}; requiring download: {len(pending)}")

    started = datetime.now(timezone.utc)
    failures: list[tuple[str, str]] = []
    downloaded = 0

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _fetch,
                    f"{BASE_URL}/{dataset}/{name}",
                    dataset_dir / name,
                    args.retries,
                ): name
                for name in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                name, ok, error = future.result()
                if ok:
                    downloaded += 1
                else:
                    failures.append((name, error))
                if completed % 200 == 0 or completed == len(pending):
                    print(f"  progress: {completed}/{len(pending)}")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log_path = REPORTS_DIR / f"recovery_python_{dataset}.log"
    lines = [
        "PhysioNet Dataset Recovery (Python)",
        f"Timestamp: {started.isoformat()}",
        f"Dataset: {dataset}",
        f"Manifest: {manifest}",
        f"Manifest entries: {len(names)}",
        f"Requiring download: {len(pending)}",
        f"Downloaded: {downloaded}",
        f"Failures: {len(failures)}",
        f"Elapsed seconds: {elapsed:.1f}",
    ]
    lines.extend(f"[FAIL] {name}: {error}" for name, error in failures)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Downloaded: {downloaded}; failures: {len(failures)}")
    print(f"Log: {log_path}")
    if failures:
        for name, error in failures[:20]:
            print(f"  {name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
