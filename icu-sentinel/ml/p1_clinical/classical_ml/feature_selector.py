"""Feature selection for P1 classical ML models.

Detects and removes constant, duplicate, all-missing, and infinite
columns. All detection is fitted on training data only and applied to
validation/test without refitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.feature_schema import FeatureSchema
from .data_loader import IDENTIFIER_COLUMNS, TARGET_COLUMN

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.feature_selector")


def _column_content_digest(series: pd.Series) -> str:
    """Return a digest of the complete column contents.

    ``pd.util.hash_pandas_object`` hashes every value (NaN-safe) rather than
    only the first value. The resulting uint64 array is digested to keep the
    per-column key small.
    """
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.blake2b(hashed.tobytes(), digest_size=16).hexdigest()


def _columns_have_identical_values(left: pd.Series, right: pd.Series) -> bool:
    """Compare complete column contents, treating NaNs in the same place equal.

    ``Series.equals`` requires identical length and dtype and considers NaN
    values at the same position equal. This acts as a collision check on top
    of the content digest so a hash collision can never drop a valid column.
    """
    if len(left) != len(right):
        return False
    return bool(left.equals(right))


@dataclass
class FeatureSelectionReport:
    """Report of feature selection decisions."""

    n_features_input: int = 0
    n_features_output: int = 0
    removed_constant: list[str] = field(default_factory=list)
    removed_duplicate: list[str] = field(default_factory=list)
    removed_all_missing: list[str] = field(default_factory=list)
    removed_infinite: list[str] = field(default_factory=list)
    removed_highly_correlated: list[str] = field(default_factory=list)
    kept_features: list[str] = field(default_factory=list)
    fit_on_train_only: bool = True

    def to_dict(self) -> dict:
        return {
            "n_features_input": self.n_features_input,
            "n_features_output": self.n_features_output,
            "n_removed": self.n_features_input - self.n_features_output,
            "removed_constant": list(self.removed_constant),
            "removed_duplicate": list(self.removed_duplicate),
            "removed_all_missing": list(self.removed_all_missing),
            "removed_infinite": list(self.removed_infinite),
            "removed_highly_correlated": list(self.removed_highly_correlated),
            "kept_features": list(self.kept_features),
            "fit_on_train_only": self.fit_on_train_only,
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class FeatureSelector:
    """Feature selection fitted on training data only."""

    def __init__(
        self,
        remove_constant: bool = True,
        remove_duplicate: bool = True,
        remove_all_missing: bool = True,
        remove_infinite: bool = True,
        correlation_threshold: float | None = None,
    ) -> None:
        self.remove_constant = remove_constant
        self.remove_duplicate = remove_duplicate
        self.remove_all_missing = remove_all_missing
        self.remove_infinite = remove_infinite
        self.correlation_threshold = correlation_threshold

        # Fitted state (from training data only).
        self.kept_features: list[str] = []
        self.removed_features: dict[str, list[str]] = {
            "constant": [],
            "duplicate": [],
            "all_missing": [],
            "infinite": [],
            "highly_correlated": [],
        }
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        feature_schema: FeatureSchema | None = None,
    ) -> FeatureSelector:
        """Fit feature selection on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (identifiers/target already excluded).
        feature_schema : FeatureSchema, optional
            Schema for cross-validation of feature names.

        Returns
        -------
        self
        """
        # Start with all columns in X.
        all_features = list(X.columns)

        # Cross-reference with schema tabular features if provided.
        if feature_schema is not None:
            schema_feature_names = {
                f
                for f in feature_schema.feature_names()
                if f not in IDENTIFIER_COLUMNS and f != TARGET_COLUMN
            }
            all_features = [f for f in all_features if f in schema_feature_names]

        kept: list[str] = []
        removed: dict[str, list[str]] = {
            "constant": [],
            "duplicate": [],
            "all_missing": [],
            "infinite": [],
            "highly_correlated": [],
        }

        # Full-column content digest -> candidate column names. The candidate
        # list handles the (rare) possibility of a digest collision.
        seen_columns: dict[str, list[str]] = {}

        for col in all_features:
            col_data = X[col]

            # 1. All-missing check.
            if self.remove_all_missing and col_data.isna().all():
                removed["all_missing"].append(col)
                continue

            # 2. Infinite check.
            if self.remove_infinite:
                try:
                    has_inf = np.isinf(col_data.astype(float)).any()
                except (TypeError, ValueError):
                    has_inf = False
                if has_inf:
                    removed["infinite"].append(col)
                    continue

            # 3. Constant check (after removing all-missing and infinite).
            try:
                nunique = col_data.nunique(dropna=False)
            except TypeError:
                nunique = 1
            if self.remove_constant and nunique <= 1:
                removed["constant"].append(col)
                continue

            # 4. Duplicate check (compare complete column contents).
            if self.remove_duplicate:
                col_hash = _column_content_digest(col_data)
                candidates = seen_columns.get(col_hash, [])
                duplicate_of = next(
                    (
                        candidate
                        for candidate in candidates
                        if _columns_have_identical_values(
                            col_data, X[candidate]
                        )
                    ),
                    None,
                )
                if duplicate_of is not None:
                    removed["duplicate"].append(col)
                    continue
                seen_columns.setdefault(col_hash, []).append(col)

            kept.append(col)

        # 5. Highly correlated features (optional).
        if self.correlation_threshold is not None and len(kept) > 1:
            try:
                corr_matrix = X[kept].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = set()
                for col in upper.columns:
                    high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
                    for other in high_corr:
                        if col not in to_drop and other not in to_drop:
                            to_drop.add(other)
                            removed["highly_correlated"].append(other)
                kept = [c for c in kept if c not in to_drop]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Correlation-based selection failed: %s", exc)

        self.kept_features = kept
        self.removed_features = removed
        self._fitted = True

        logger.info(
            "Feature selection: %d -> %d features "
            "(constant=%d, duplicate=%d, all_missing=%d, infinite=%d, "
            "highly_correlated=%d)",
            len(all_features),
            len(kept),
            len(removed["constant"]),
            len(removed["duplicate"]),
            len(removed["all_missing"]),
            len(removed["infinite"]),
            len(removed["highly_correlated"]),
        )
        return self

    def transform(self, X: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Apply fitted feature selection to a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.
        copy : bool
            When True (default) return a new frame. When False, drop unused
            columns in place and return ``X`` to avoid a full-matrix copy on
            large datasets.

        Returns
        -------
        pd.DataFrame
            X with only kept features.
        """
        if not self._fitted:
            raise ValueError(
                "FeatureSelector must be fitted before transform. "
                "Call fit() on training data first."
            )
        available = [c for c in self.kept_features if c in X.columns]
        missing = [c for c in self.kept_features if c not in X.columns]
        if missing:
            logger.warning(
                "%d kept features missing from transform data: %s",
                len(missing),
                missing[:5],
            )
        if copy:
            return X[available].copy()
        drop = [c for c in X.columns if c not in available]
        if drop:
            X.drop(columns=drop, inplace=True)
        return X

    def fit_transform(
        self,
        X: pd.DataFrame,
        feature_schema: FeatureSchema | None = None,
        copy: bool = True,
    ) -> pd.DataFrame:
        """Fit on X and transform X in one call (training data only)."""
        self.fit(X, feature_schema=feature_schema)
        return self.transform(X, copy=copy)

    def report(self) -> FeatureSelectionReport:
        """Return a FeatureSelectionReport."""
        return FeatureSelectionReport(
            n_features_input=len(self.kept_features)
            + sum(len(v) for v in self.removed_features.values()),
            n_features_output=len(self.kept_features),
            removed_constant=list(self.removed_features["constant"]),
            removed_duplicate=list(self.removed_features["duplicate"]),
            removed_all_missing=list(self.removed_features["all_missing"]),
            removed_infinite=list(self.removed_features["infinite"]),
            removed_highly_correlated=list(
                self.removed_features["highly_correlated"]
            ),
            kept_features=list(self.kept_features),
            fit_on_train_only=True,
        )

    @property
    def n_input_features(self) -> int:
        return len(self.kept_features) + sum(
            len(v) for v in self.removed_features.values()
        )

    @property
    def n_output_features(self) -> int:
        return len(self.kept_features)
