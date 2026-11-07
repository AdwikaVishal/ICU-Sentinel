"""Feature and target contract shared by P2 models.

Reuses P1's identifier and target definitions. P2 adds ``hospital_id`` to
the non-feature set because P1 stores it only when genuinely available and
it must never become a model input.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ml.p1_clinical.classical_ml.data_loader import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
)
from ml.p1_clinical.features.feature_schema import FeatureSchema
from ml.p1_clinical.preprocessing.targets import DERIVED_TARGET_COLUMN

P2_METADATA_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys([*IDENTIFIER_COLUMNS, "hospital_id"])
)
P2_NON_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys([*P2_METADATA_COLUMNS, *TARGET_COLUMNS])
)
P2_TARGET_COLUMNS: tuple[str, ...] = tuple(TARGET_COLUMNS)
P2_DEFAULT_TARGET: str = DERIVED_TARGET_COLUMN


def resolve_feature_names(
    columns: Iterable[str],
    schema: FeatureSchema | None = None,
) -> list[str]:
    """Return model feature names, excluding identifiers and targets."""
    candidates = [c for c in columns if c not in P2_NON_FEATURE_COLUMNS]
    if schema is not None:
        allowed = set(schema.feature_names()) - set(P2_NON_FEATURE_COLUMNS)
        candidates = [c for c in candidates if c in allowed]
    return candidates


def validate_feature_names(feature_names: Iterable[str]) -> list[str]:
    """Validate an explicit feature list and preserve its order."""
    names = list(feature_names)
    if not names:
        raise ValueError("At least one model feature is required.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    leaked = [
        name
        for name in names
        if name in P2_NON_FEATURE_COLUMNS or name in P2_TARGET_COLUMNS
    ]
    if leaked:
        raise ValueError(
            "Identifier/target columns cannot be model features: "
            f"{sorted(leaked)}"
        )
    return names


def validate_feature_frame(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    allow_extra_columns: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Validate and return a feature-only DataFrame with preserved names.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame containing at least the required feature columns.
    feature_names : Iterable[str]
        Required feature column names in the desired order.
    allow_extra_columns : bool, default=True
        If True, extra columns beyond feature_names are allowed.
        If False, raises ValueError if extra columns are present.
    copy : bool, default=True
        If True, returns a copy of the selected columns (safe, backward-compatible).
        If False, returns a view when possible (memory-efficient, caller must not
        mutate the result). Use copy=False only when:
        - The caller will not modify the returned DataFrame
        - The source DataFrame will not be modified afterward
        - Memory efficiency is critical (e.g., large training sets)
    
    Returns
    -------
    pd.DataFrame
        Feature-only DataFrame with columns in the order specified by feature_names.
        May be a view if copy=False.
    
    Notes
    -----
    Phase 3 optimization: Added copy=False option to avoid unnecessary memory
    allocation when the caller guarantees no mutation. The default copy=True
    preserves backward compatibility and safety.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Model input must be a pandas DataFrame.")

    names = validate_feature_names(feature_names)
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise ValueError(
            "Feature schema mismatch; missing required columns: "
            f"{missing}"
        )

    if not allow_extra_columns:
        extra = [c for c in frame.columns if c not in names]
        if extra:
            raise ValueError(
                f"Unexpected columns not in feature schema: {extra}"
            )

    # Select columns: copy() if requested, otherwise return view
    if copy:
        selected = frame.loc[:, names].copy()
    else:
        # This returns a view when frame's column order matches names
        selected = frame.loc[:, names]
    
    if len(selected) == 0:
        raise ValueError("Model input contains zero rows.")

    non_numeric = [
        col
        for col in names
        if not (is_numeric_dtype(selected[col]) or is_bool_dtype(selected[col]))
    ]
    if non_numeric:
        raise ValueError(
            "Model features must be numeric; non-numeric columns: "
            f"{non_numeric}"
        )

    return selected


def sanitize_infinities(
    frame: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Replace +/-inf with NaN.

    P1 preserves NaN and documents ``inf`` as a never-observed sentinel for
    hours-since-last features. Tree models accept NaN; replacing infinities
    avoids estimator-specific failures without fitting anything on
    validation or test data.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame potentially containing infinite values.
    inplace : bool, default=False
        If True, modifies the input DataFrame in-place and returns it (memory-efficient).
        If False, creates and returns a copy (safe, backward-compatible).
        
        Use inplace=True only when:
        - The caller owns the DataFrame and will not use it again
        - Memory efficiency is critical (e.g., large training sets)
        - The caller explicitly wants mutation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with infinities replaced by NaN. Same object as input if
        inplace=True, otherwise a new DataFrame.
    
    Notes
    -----
    Phase 3 optimization: Added inplace=True option to avoid unnecessary memory
    allocation when the caller guarantees the input is no longer needed.
    The default inplace=False preserves backward compatibility and safety.
    
    WARNING: Cannot use inplace=True with sklearn pipelines. The imputer
    validation fails when the DataFrame is modified in place. Always use
    inplace=False (default) for data that will pass through sklearn estimators.
    
    Examples
    --------
    Safe usage (default, backward-compatible):
    >>> cleaned = sanitize_infinities(df)
    
    Memory-efficient usage when df is no longer needed (NOT for sklearn):
    >>> df = sanitize_infinities(df, inplace=True)
    """
    if inplace:
        # pandas.replace with inplace=True returns None, so we need to call it
        # and return the frame
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame
    else:
        return frame.replace([np.inf, -np.inf], np.nan, inplace=False)


def validate_training_target(y: pd.Series | np.ndarray, name: str) -> pd.Series:
    """Validate a binary training target and return it as an int Series."""
    series = pd.Series(y).reset_index(drop=True)
    if len(series) == 0:
        raise ValueError("Training target is empty.")
    if series.isna().any():
        raise ValueError("Training target contains missing values.")
    invalid = sorted(set(series.unique()) - {0, 1})
    if invalid:
        raise ValueError(
            f"Training target '{name}' must be binary; found {invalid}."
        )
    if series.nunique() < 2:
        raise ValueError(
            "Training target must contain both classes; "
            f"found only {series.unique().tolist()}."
        )
    return series.astype(int)


def validate_prediction_target(
    y: pd.Series | np.ndarray,
    name: str,
    *,
    require_both: bool = True,
) -> pd.Series:
    """Validate a binary evaluation target."""
    series = pd.Series(y).reset_index(drop=True)
    if len(series) == 0:
        raise ValueError("Evaluation target is empty.")
    if series.isna().any():
        raise ValueError("Evaluation target contains missing values.")
    invalid = sorted(set(series.unique()) - {0, 1})
    if invalid:
        raise ValueError(
            f"Evaluation target '{name}' must be binary; found {invalid}."
        )
    if require_both and series.nunique() < 2:
        raise ValueError(
            "Evaluation target must contain both classes to compute AUROC/AUPRC; "
            f"found only {series.unique().tolist()}."
        )
    return series.astype(int)
