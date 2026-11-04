"""Feature selection for P1 classical ML models.

Detects and removes constant, duplicate, all-missing, and infinite
columns. All detection is fitted on training data only and applied to
validation/test without refitting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.feature_schema import FeatureSchema
from .data_loader import IDENTIFIER_COLUMNS, TARGET_COLUMN

logger = logging.getLogger("icu_sentinel.p1_clinical.classical_ml.feature_selector")


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
            set(feature_schema.tabular_features().__class__ and [])
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

        seen_columns: dict[str, str] = {}  # column values hash -> first name

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

            # 4. Duplicate check (compare value patterns).
            if self.remove_duplicate:
                col_hash = pd.util.hash_pandas_object(col_data).iloc[0]
                if col_hash in seen_columns:
                    removed["duplicate"].append(col)
                    continue
                seen_columns[col_hash] = col

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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted feature selection to a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

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
        return X[available].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        feature_schema: FeatureSchema | None = None,
    ) -> pd.DataFrame:
        """Fit on X and transform X in one call (training data only)."""
        self.fit(X, feature_schema=feature_schema)
        return self.transform(X)

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
