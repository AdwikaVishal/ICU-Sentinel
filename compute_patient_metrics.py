"""Pre-compute data-quality metrics for all patients from source parquet files.

Reads clinical columns from each patient parquet and computes:
  * total_hours   - number of hourly records (row count)
  * recorded_hours- rows with at least one non-null clinical observation
  * missing_hours - total_hours - recorded_hours
  * observation_count - non-null clinical measurements (ICULOS excluded)

Results are written to a JSON cache file for fast backend loading.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

# Clinical variables from P1 schema, excluding ICULOS per task requirements.
CLINICAL_VARIABLES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]
OBS_COLUMNS = [v for v in CLINICAL_VARIABLES if v != "ICULOS"]


def _compute_one(file_path: str):
    """Compute metrics for a single patient parquet file."""
    pid = os.path.basename(file_path).replace(".parquet", "")
    try:
        table = pq.read_table(file_path, columns=OBS_COLUMNS)
        df = table.to_pandas()
        total = int(len(df))
        if total == 0:
            return pid, 0, 0, 0
        obs_frame = df[OBS_COLUMNS]
        recorded = int(obs_frame.notna().any(axis=1).sum())
        obs_count = int(obs_frame.notna().sum().sum())
        return pid, total, recorded, obs_count
    except Exception:
        return pid, 0, 0, 0


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel\data\processed"
    )
    output = sys.argv[2] if len(sys.argv) > 2 else (
        r"C:\Users\Saatvika Reddy\ICU-Sentinel\icu-sentinel\data\reports\patient_metrics_cache.json"
    )

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"Found {len(files)} patient parquet files")

    metrics = {}
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_compute_one, f): f for f in files}
        for fut in as_completed(futures):
            pid, total, recorded, obs = fut.result()
            metrics[pid] = {
                "total_hours": total,
                "recorded_hours": recorded,
                "missing_hours": max(0, total - recorded),
                "observation_count": obs,
            }
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {done}/{len(files)} in {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nComputed metrics for {len(metrics)} patients in {elapsed:.1f}s")

    # Write cache
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f)
    print(f"Cache written to {output}")


if __name__ == "__main__":
    main()
