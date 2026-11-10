#!/usr/bin/env python3
"""Generate synthetic PhysioNet/CinC Challenge 2019 data for P1 pipeline validation.

This script creates synthetic .psv files that mimic the real dataset structure,
enabling end-to-end testing of the P1 pipeline without requiring the actual
PhysioNet data. The synthetic data is clearly marked and should NEVER be used
for scientific results.

Usage:
    python scripts/generate_synthetic_data.py --n-patients 100 --output-dir data/raw
    python scripts/generate_synthetic_data.py --n-patients 1000 --sepsis-rate 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICU_SENTINEL_ROOT = _REPO_ROOT / "icu-sentinel"
if str(_ICU_SENTINEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICU_SENTINEL_ROOT))

import numpy as np
import pandas as pd

from ml.p1_clinical.ingestion.dataset_loader import (
    REQUIRED_COLUMNS,
    write_synthetic_psv,
    build_synthetic_patient_file,
)


def generate_synthetic_dataset(
    output_dir: Path,
    n_patients: int,
    sepsis_rate: float = 0.15,
    min_hours: int = 12,
    max_hours: int = 48,
    seed: int = 42,
) -> None:
    """Generate synthetic patient files.

    Parameters
    ----------
    output_dir : Path
        Directory to write .psv files.
    n_patients : int
        Number of patient files to generate.
    sepsis_rate : float
        Fraction of patients who develop sepsis.
    min_hours : int
        Minimum ICU stay hours.
    max_hours : int
        Maximum ICU stay hours.
    seed : int
        Random seed for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_septic = int(n_patients * sepsis_rate)
    septic_indices = set(rng.choice(n_patients, size=n_septic, replace=False))

    print(f"Generating {n_patients} synthetic patients ({n_septic} septic)...")

    for i in range(n_patients):
        patient_id = f"p{i+1:05d}"
        n_hours = int(rng.integers(min_hours, max_hours + 1))
        is_septic = i in septic_indices

        if is_septic:
            # Sepsis onset between hour 6 and n_hours - 6 (to allow 6h window)
            onset_hour = int(rng.integers(6, max(7, n_hours - 5)))
        else:
            onset_hour = None

        df = build_synthetic_patient_file(
            patient_id=patient_id,
            n_hours=n_hours,
            seed=seed + i,
            sepsis_onset_hour=onset_hour,
        )

        filepath = output_dir / f"{patient_id}.psv"
        write_synthetic_psv(df, filepath)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_patients} patients...")

    print(f"Done. {n_patients} patient files written to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
        help="Output directory for .psv files",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=100,
        help="Number of synthetic patients to generate",
    )
    parser.add_argument(
        "--sepsis-rate",
        type=float,
        default=0.15,
        help="Fraction of patients with sepsis (0-1)",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=12,
        help="Minimum ICU stay hours",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=48,
        help="Maximum ICU stay hours",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not (0 <= args.sepsis_rate <= 1):
        print("ERROR: sepsis-rate must be between 0 and 1", file=sys.stderr)
        return 1

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        n_patients=args.n_patients,
        sepsis_rate=args.sepsis_rate,
        min_hours=args.min_hours,
        max_hours=args.max_hours,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())