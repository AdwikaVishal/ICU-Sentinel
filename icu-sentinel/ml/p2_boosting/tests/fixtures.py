"""Synthetic fixtures for P2 tests. No real patient data is used."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.p1_clinical.classical_ml.data_loader import (
    LoadedDataset,
    SplitData,
)
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN
from ml.p2_boosting.config import (
    ExtraTreesConfig,
    LightGBMConfig,
    P2Config,
    XGBoostConfig,
)

FEATURE_NAMES = [f"f{i}" for i in range(6)]


def make_split(
    split: str,
    n_rows: int = 120,
    seed: int = 42,
    patient_prefix: str | None = None,
) -> SplitData:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n_rows, len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
    )
    X.iloc[::11, 0] = np.nan
    noise = rng.normal(0, 0.5, size=n_rows)
    y = (X["f0"].fillna(0) + 0.5 * X["f1"] + noise > 0).astype(int)
    if y.nunique() < 2:
        y.iloc[0], y.iloc[1] = 0, 1
    prefix = patient_prefix or split
    patient_ids = pd.Series(
        [f"{prefix}_{i:03d}" for i in range(n_rows)]
    )
    return SplitData(
        split=split,
        X=X,
        y=y,
        patient_ids=patient_ids,
        hours=pd.Series(range(n_rows)),
        target_column=DERIVED_TARGET_COLUMN,
    )


def make_dataset(seed: int = 42) -> LoadedDataset:
    return LoadedDataset(
        train=make_split("train", seed=seed, patient_prefix="tr"),
        validation=make_split("validation", seed=seed + 1, patient_prefix="va"),
        test=make_split("test", seed=seed + 2, patient_prefix="te"),
        feature_names=list(FEATURE_NAMES),
        n_features=len(FEATURE_NAMES),
        n_patients_total=360,
        source="synthetic",
        target_column=DERIVED_TARGET_COLUMN,
    )


def fast_config(enable_lightgbm: bool = False) -> P2Config:
    """Fast test config. LightGBM disabled by default due to Windows DLL issues."""
    return P2Config(
        random_seed=42,
        train_extra_trees=True,
        train_lightgbm=enable_lightgbm,
        train_xgboost=True,
        extra_trees=ExtraTreesConfig(n_estimators=10),
        lightgbm=LightGBMConfig(n_estimators=10, verbosity=-1),
        xgboost=XGBoostConfig(n_estimators=10),
    )
