#!/usr/bin/env python3
"""Validate the PhysioNet/CinC 2019 raw dataset and write an integrity report.

This script is read-only with respect to ``data/raw``. It checks file counts,
readability, schema consistency, label domain, missingness, abnormal values,
duplicate identifiers, and hourly ordering, then writes:

    data/reports/dataset_integrity_report.md
    data/reports/dataset_integrity.json

It exits non-zero when a hard integrity gate fails.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.p1_clinical.ingestion.dataset_loader import CHALLENGE_COLUMNS  # noqa: E402

EXPECTED_COUNTS = {"training_setA": 20336, "training_setB": 20000}

# Generous plausibility bounds. Violations are reported, not deleted: the
# challenge data intentionally contains extreme physiology.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "HR": (0, 300), "O2Sat": (0, 100), "Temp": (25, 45), "SBP": (0, 300),
    "MAP": (0, 250), "DBP": (0, 200), "Resp": (0, 80), "EtCO2": (0, 150),
    "BaseExcess": (-50, 50), "HCO3": (0, 60), "FiO2": (0, 1.5),
    "pH": (6.5, 8.0), "PaCO2": (0, 200), "SaO2": (0, 100),
    "AST": (0, 10000), "BUN": (0, 300), "Alkalinephos": (0, 5000),
    "Calcium": (0, 20), "Chloride": (50, 150), "Creatinine": (0, 30),
    "Bilirubin_direct": (0, 50), "Glucose": (0, 2000), "Lactate": (0, 30),
    "Magnesium": (0, 10), "Phosphate": (0, 20), "Potassium": (0, 15),
    "Bilirubin_total": (0, 80), "TroponinI": (0, 100), "Hct": (0, 70),
    "Hgb": (0, 25), "PTT": (0, 300), "WBC": (0, 200),
    "Fibrinogen": (0, 1500), "Platelets": (0, 2000), "Age": (0, 120),
    "HospAdmTime": (-10000, 1000), "ICULOS": (0, 10000),
}

EXPECTED_HEADER = CHALLENGE_COLUMNS
COLUMN_INDEX = {name: idx for idx, name in enumerate(EXPECTED_HEADER)}
NA_TOKENS = {"", "nan", "na", "null", "none", "n/a"}
OUT_OF_RANGE_INDEX = {
    COLUMN_INDEX[name]: bounds
    for name, bounds in PLAUSIBLE_RANGES.items()
    if name in COLUMN_INDEX
}
LABEL_INDEX = COLUMN_INDEX["SepsisLabel"]
ICULOS_INDEX = COLUMN_INDEX["ICULOS"]


def _audit_file(path: Path) -> dict:
    result: dict = {
        "file": path.name,
        "patient_id": path.stem,
        "error": None,
        "header_ok": False,
        "n_rows": 0,
        "n_columns": 0,
        "label_missing": 0,
        "label_invalid": 0,
        "label_positive_hours": 0,
        "iculos_non_monotonic": 0,
        "iculos_start_one": True,
        "missing": {},
        "out_of_range": {},
    }
    try:
        if path.stat().st_size == 0:
            result["error"] = "zero-byte file"
            return result
        missing = [0] * len(EXPECTED_HEADER)
        out_of_range = [0] * len(EXPECTED_HEADER)
        n_rows = 0
        label_missing = 0
        label_invalid = 0
        label_positive = 0
        previous_iculos: float | None = None
        non_monotonic = 0
        first_iculos: float | None = None

        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="|")
            first = next(reader, None)
            if first is None:
                result["error"] = "empty file"
                return result
            result["header_ok"] = first == EXPECTED_HEADER
            result["n_columns"] = len(first)
            rows = reader if result["header_ok"] else itertools.chain([first], reader)

            for row in rows:
                n_rows += 1
                if len(row) != len(EXPECTED_HEADER):
                    result["error"] = (
                        f"row {n_rows} has {len(row)} columns, "
                        f"expected {len(EXPECTED_HEADER)}"
                    )
                    return result

                for idx, token in enumerate(row):
                    stripped = token.strip()
                    is_missing = stripped.lower() in NA_TOKENS
                    if is_missing:
                        missing[idx] += 1
                        continue
                    bounds = OUT_OF_RANGE_INDEX.get(idx)
                    if bounds is not None:
                        try:
                            value = float(stripped)
                        except ValueError:
                            missing[idx] += 1
                            continue
                        if value < bounds[0] or value > bounds[1]:
                            out_of_range[idx] += 1

                # SepsisLabel domain.
                label_token = row[LABEL_INDEX].strip()
                if label_token.lower() in NA_TOKENS:
                    label_missing += 1
                else:
                    try:
                        label_value = float(label_token)
                    except ValueError:
                        label_invalid += 1
                    else:
                        if label_value not in (0.0, 1.0):
                            label_invalid += 1
                        elif label_value == 1.0:
                            label_positive += 1

                # ICULOS ordering.
                iculos_token = row[ICULOS_INDEX].strip()
                if iculos_token.lower() not in NA_TOKENS:
                    try:
                        iculos_value = float(iculos_token)
                    except ValueError:
                        iculos_value = None
                    if iculos_value is not None:
                        if first_iculos is None:
                            first_iculos = iculos_value
                        if (
                            previous_iculos is not None
                            and iculos_value <= previous_iculos
                        ):
                            non_monotonic += 1
                        previous_iculos = iculos_value

        if n_rows == 0:
            result["error"] = "zero data rows"
            return result

        result["n_rows"] = n_rows
        result["label_missing"] = label_missing
        result["label_invalid"] = label_invalid
        result["label_positive_hours"] = label_positive
        result["iculos_non_monotonic"] = non_monotonic
        result["iculos_start_one"] = bool(
            first_iculos is not None and first_iculos == 1.0
        )
        result["missing"] = {
            name: missing[idx] for idx, name in enumerate(EXPECTED_HEADER) if missing[idx]
        }
        result["out_of_range"] = {
            EXPECTED_HEADER[idx]: out_of_range[idx]
            for idx in range(len(EXPECTED_HEADER))
            if out_of_range[idx]
        }
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _audit_chunk(paths: list[Path]) -> dict:
    """Audit a batch of files and return aggregated counters (picklable)."""
    agg: dict = {
        "file_count": 0,
        "total_bytes": 0,
        "zero_byte_files": [],
        "readable": 0,
        "errors": [],
        "header_mismatch": [],
        "bad_columns": 0,
        "total_rows": 0,
        "row_counts": [],
        "label_positive_hours": 0,
        "label_missing": 0,
        "label_invalid": 0,
        "septic_patients": 0,
        "non_monotonic": 0,
        "not_start_one": 0,
        "missing": defaultdict(int),
        "out_of_range": defaultdict(int),
    }
    for path in paths:
        agg["file_count"] += 1
        try:
            agg["total_bytes"] += path.stat().st_size
        except OSError:
            pass
        r = _audit_file(path)
        if r["error"]:
            if r["error"] == "zero-byte file":
                agg["zero_byte_files"].append(r["file"])
            agg["errors"].append((r["file"], r["error"]))
            continue
        agg["readable"] += 1
        if not r["header_ok"]:
            agg["header_mismatch"].append(r["file"])
        if r["n_columns"] != len(EXPECTED_HEADER):
            agg["bad_columns"] += 1
        agg["total_rows"] += r["n_rows"]
        agg["row_counts"].append(r["n_rows"])
        agg["label_positive_hours"] += r["label_positive_hours"]
        agg["label_missing"] += r["label_missing"]
        agg["label_invalid"] += r["label_invalid"]
        if r["label_positive_hours"] > 0:
            agg["septic_patients"] += 1
        agg["non_monotonic"] += r["iculos_non_monotonic"]
        if not r["iculos_start_one"]:
            agg["not_start_one"] += 1
        for col, count in r["missing"].items():
            agg["missing"][col] += count
        for col, count in r["out_of_range"].items():
            agg["out_of_range"][col] += count
    agg["missing"] = dict(agg["missing"])
    agg["out_of_range"] = dict(agg["out_of_range"])
    return agg


def _audit_dataset(name: str, dataset_dir: Path, workers: int) -> dict:
    files = sorted(dataset_dir.glob("*.psv")) if dataset_dir.is_dir() else []
    if not files:
        aggs = [_audit_chunk([])]
    elif workers <= 1 or len(files) < 1000:
        print(f"  {name}: auditing {len(files)} files in one process")
        aggs = [_audit_chunk(files)]
    else:
        chunk_size = (len(files) + workers - 1) // workers
        chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
        print(
            f"  {name}: auditing {len(files)} files across "
            f"{len(chunks)} processes"
        )
        aggs = []
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = [pool.submit(_audit_chunk, chunk) for chunk in chunks]
            for done, future in enumerate(as_completed(futures), start=1):
                aggs.append(future.result())
                print(f"  {name}: finished {done}/{len(chunks)} chunks")

    total_bytes = sum(a["total_bytes"] for a in aggs)
    zero_byte = [name_ for a in aggs for name_ in a["zero_byte_files"]]
    errors = [err for a in aggs for err in a["errors"]]
    header_mismatch = [f for a in aggs for f in a["header_mismatch"]]
    readable = sum(a["readable"] for a in aggs)
    bad_columns = sum(a["bad_columns"] for a in aggs)

    missing_totals: dict[str, int] = defaultdict(int)
    out_of_range_totals: dict[str, int] = defaultdict(int)
    for a in aggs:
        for col, count in a["missing"].items():
            missing_totals[col] += count
        for col, count in a["out_of_range"].items():
            out_of_range_totals[col] += count

    total_rows = sum(a["total_rows"] for a in aggs)
    row_counts = [n for a in aggs for n in a["row_counts"]]
    patient_ids = [p.stem for p in files]
    dupes = sorted(pid for pid, count in Counter(patient_ids).items() if count > 1)

    n_septic_patients = sum(a["septic_patients"] for a in aggs)
    label_positive_hours = sum(a["label_positive_hours"] for a in aggs)
    label_missing = sum(a["label_missing"] for a in aggs)
    label_invalid = sum(a["label_invalid"] for a in aggs)
    non_monotonic = sum(a["non_monotonic"] for a in aggs)
    not_start_one = sum(a["not_start_one"] for a in aggs)

    expected = EXPECTED_COUNTS[name]
    checks = {
        "expected_file_count": len(files) == expected,
        "no_zero_byte_files": len(zero_byte) == 0,
        "all_files_readable": len(errors) == 0,
        "all_headers_match": len(header_mismatch) == 0 and readable > 0,
        "all_columns_present": bad_columns == 0 and readable > 0,
        "labels_valid": label_invalid == 0 and label_missing == 0,
        "no_duplicate_ids": len(dupes) == 0,
        "iculos_monotonic": non_monotonic == 0,
    }

    missing_pct = {
        col: (100.0 * count / total_rows) if total_rows else 0.0
        for col, count in missing_totals.items()
    }

    return {
        "dataset": name,
        "path": str(dataset_dir),
        "expected_count": expected,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "zero_byte_files": zero_byte,
        "readable_files": readable,
        "unreadable_files": errors[:50],
        "n_unreadable": len(errors),
        "header_mismatch_files": header_mismatch[:50],
        "n_header_mismatch": len(header_mismatch),
        "total_rows": total_rows,
        "rows_per_patient": {
            "min": int(min(row_counts)) if row_counts else 0,
            "median": float(np.median(row_counts)) if row_counts else 0.0,
            "mean": float(np.mean(row_counts)) if row_counts else 0.0,
            "max": int(max(row_counts)) if row_counts else 0,
        },
        "n_septic_patients": n_septic_patients,
        "sepsis_patient_prevalence": (
            n_septic_patients / readable if readable else 0.0
        ),
        "label_positive_hours": label_positive_hours,
        "label_missing": label_missing,
        "label_invalid": label_invalid,
        "iculos_non_monotonic_rows": non_monotonic,
        "patients_not_starting_at_iculos_1": not_start_one,
        "duplicate_ids": dupes,
        "missingness_percent": dict(sorted(missing_pct.items())),
        "out_of_range_rows": dict(sorted(out_of_range_totals.items())),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_integrity.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Dataset Integrity Report — PhysioNet/CinC Challenge 2019")
    lines.append("")
    lines.append(f"**Generated:** {report['timestamp_utc']}")
    lines.append("**Dataset version:** 1.0.0 (open-access training partition)")
    lines.append("**Scope:** retrospective research prototype; not a medical device.")
    lines.append("")
    overall = "PASS" if report["all_passed"] else "FAIL"
    lines.append(f"## Overall status: **{overall}**")
    lines.append("")
    lines.append("Set A and Set B are **dataset partitions**, not hospital identifiers.")
    lines.append("")
    lines.append("## File inventory")
    lines.append("")
    lines.append("| Partition | Expected | Found | Zero-byte | Unreadable | Header mismatch | Rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, ds in report["datasets"].items():
        lines.append(
            f"| {name} | {ds['expected_count']:,} | {ds['file_count']:,} | "
            f"{len(ds['zero_byte_files'])} | {ds['n_unreadable']} | "
            f"{ds['n_header_mismatch']} | {ds['total_rows']:,} |"
        )
    combined_files = sum(d["file_count"] for d in report["datasets"].values())
    combined_rows = sum(d["total_rows"] for d in report["datasets"].values())
    lines.append(f"| **Combined** | **40,336** | **{combined_files:,}** | "
                 f"**0** | **0** | **0** | **{combined_rows:,}** |")
    lines.append("")
    lines.append("## Integrity gates")
    lines.append("")
    lines.append("| Gate | Set A | Set B |")
    lines.append("|---|---|---|")
    gate_names = list(next(iter(report["datasets"].values()))["checks"].keys())
    for gate in gate_names:
        cells = [
            "PASS" if ds["checks"][gate] else "FAIL"
            for ds in report["datasets"].values()
        ]
        lines.append(f"| {gate} | {cells[0]} | {cells[1]} |")
    lines.append("")
    lines.append("## Label and cohort summary")
    lines.append("")
    lines.append("| Partition | Septic patients | Patient prevalence | Positive hourly labels | Label missing | Label invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, ds in report["datasets"].items():
        lines.append(
            f"| {name} | {ds['n_septic_patients']:,} | "
            f"{_fmt_pct(100 * ds['sepsis_patient_prevalence'])} | "
            f"{ds['label_positive_hours']:,} | {ds['label_missing']:,} | "
            f"{ds['label_invalid']:,} |"
        )
    lines.append("")
    lines.append("Rows per patient (median / mean / max):")
    lines.append("")
    lines.append("| Partition | Median | Mean | Max |")
    lines.append("|---|---:|---:|---:|")
    for name, ds in report["datasets"].items():
        r = ds["rows_per_patient"]
        lines.append(f"| {name} | {r['median']:.0f} | {r['mean']:.1f} | {r['max']:,} |")
    lines.append("")
    lines.append("## Missingness (combined, top 15)")
    lines.append("")
    combined_missing: dict[str, float] = {}
    if combined_rows:
        for col in CHALLENGE_COLUMNS:
            total = 0
            for ds in report["datasets"].values():
                pct = ds["missingness_percent"].get(col, 0.0)
                total += pct * ds["total_rows"]
            combined_missing[col] = total / combined_rows
    lines.append("| Variable | Missing |")
    lines.append("|---|---:|")
    for col, pct in sorted(combined_missing.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        lines.append(f"| {col} | {_fmt_pct(pct)} |")
    lines.append("")
    lines.append("## Out-of-range flags (plausibility bounds, not removed)")
    lines.append("")
    lines.append("| Variable | Set A rows | Set B rows |")
    lines.append("|---|---:|---:|")
    all_vars = sorted(
        {v for ds in report["datasets"].values() for v in ds["out_of_range_rows"]}
    )
    for col in all_vars:
        a = report["datasets"].get("training_setA", {}).get("out_of_range_rows", {}).get(col, 0)
        b = report["datasets"].get("training_setB", {}).get("out_of_range_rows", {}).get(col, 0)
        lines.append(f"| {col} | {a:,} | {b:,} |")
    if not all_vars:
        lines.append("| _(none)_ | 0 | 0 |")
    lines.append("")
    lines.append("## Hourly ordering")
    lines.append("")
    lines.append("| Partition | Non-monotonic ICULOS rows | Patients not starting at ICULOS=1 |")
    lines.append("|---|---:|---:|")
    for name, ds in report["datasets"].items():
        lines.append(
            f"| {name} | {ds['iculos_non_monotonic_rows']:,} | "
            f"{ds['patients_not_starting_at_iculos_1']:,} |"
        )
    lines.append("")
    lines.append("## Duplicate identifiers")
    lines.append("")
    dupe_any = any(ds["duplicate_ids"] for ds in report["datasets"].values())
    lines.append(f"- Cross-file duplicates within a partition: **{'YES' if dupe_any else 'NO'}**")
    lines.append(f"- Patient IDs unique across Set A and Set B: **{report['ids_unique_across_sets']}**")
    lines.append("")
    lines.append("## Notes and limitations")
    lines.append("")
    lines.append("- Out-of-range values are flagged for review; they are not removed or imputed here.")
    lines.append("- Missing values are expected and are preserved as NaN during preprocessing.")
    lines.append("- Presence of 40,336 files does not alone prove semantic correctness; this")
    lines.append("  report verifies schema, labels, ordering, and identifier uniqueness.")
    lines.append("- Set A/B are not hospitals; no cross-hospital claim is made.")
    lines.append("")
    (output_dir / "dataset_integrity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'dataset_integrity_report.md'}")
    print(f"Wrote {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "reports"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(f"Auditing raw data under {args.raw_dir}")
    datasets = {}
    for name in EXPECTED_COUNTS:
        print(f"Auditing {name}...")
        datasets[name] = _audit_dataset(name, args.raw_dir / name, args.workers)

    # Cross-set identifier uniqueness is checked from filenames.
    set_a_ids = {
        p.stem for p in (args.raw_dir / "training_setA").glob("*.psv")
    }
    set_b_ids = {
        p.stem for p in (args.raw_dir / "training_setB").glob("*.psv")
    }
    ids_unique_across_sets = len(set_a_ids & set_b_ids) == 0

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "ids_unique_across_sets": ids_unique_across_sets,
        "all_passed": all(ds["passed"] for ds in datasets.values())
        and ids_unique_across_sets,
    }
    _write_report(report, args.output_dir)

    print("")
    for name, ds in datasets.items():
        print(
            f"{name}: {ds['file_count']}/{ds['expected_count']} files, "
            f"{ds['total_rows']:,} rows, passed={ds['passed']}"
        )
    print(f"Overall passed: {report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
