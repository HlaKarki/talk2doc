"""Data profiling service for generating dataset statistics and insights."""
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import Dataset, AnalysisResult
from services.data_service import get_data_service


class DataProfilingService:
    """Service for profiling datasets and generating statistics."""

    def __init__(self):
        self.data_service = get_data_service()

    async def profile_dataset(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession
    ) -> Optional[AnalysisResult]:
        """
        Generate a comprehensive profile for a dataset.

        Includes:
        - Basic stats (row count, column count, memory usage)
        - Per-column statistics
        - Missing value analysis
        - Correlation matrix (for numeric columns)
        - Data quality issues
        """
        # Load the dataframe
        df = await self.data_service.get_dataframe(dataset_id, db)
        if df is None:
            return None

        # Generate profile
        profile = {
            "basic_stats": self._get_basic_stats(df),
            "column_profiles": self._profile_columns(df),
            "missing_values": self._analyze_missing_values(df),
            "correlations": self._compute_correlations(df),
            "data_quality": self._assess_data_quality(df),
            "value_distributions": self._get_value_distributions(df),
        }

        # Store result
        result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type="profile",
            parameters={},
            results=profile,
            status="completed",
            completed_at=datetime.utcnow()
        )

        db.add(result)
        await db.flush()

        # Update dataset status
        dataset = await self.data_service.get_dataset_by_id(dataset_id, db)
        if dataset:
            dataset.status = "profiled"
            dataset.profiled_at = datetime.utcnow()
            await db.flush()

        return result

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_row_percentage": round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
        }

    def _profile_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate detailed profile for each column."""
        profiles = []

        for col in df.columns:
            series = df[col]
            profile = {
                "name": str(col),
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "missing_percentage": round(series.isna().sum() / len(series) * 100, 2) if len(series) > 0 else 0,
                "unique": int(series.nunique()),
                "unique_percentage": round(series.nunique() / len(series) * 100, 2) if len(series) > 0 else 0,
            }

            # Add type-specific stats
            if pd.api.types.is_bool_dtype(series):
                # Handle boolean separately
                profile.update(self._boolean_stats(series))
                profile["semantic_type"] = "boolean"
            elif pd.api.types.is_numeric_dtype(series):
                profile.update(self._numeric_stats(series))
                profile["semantic_type"] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.update(self._datetime_stats(series))
                profile["semantic_type"] = "datetime"
            else:
                profile.update(self._categorical_stats(series))
                # Determine if categorical or text
                if series.nunique() < 50 and series.nunique() / len(series) < 0.1:
                    profile["semantic_type"] = "categorical"
                else:
                    profile["semantic_type"] = "text"

            profiles.append(profile)

        return profiles

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for numeric columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        stats = {
            "mean": self._safe_float(clean.mean()),
            "std": self._safe_float(clean.std()),
            "min": self._safe_float(clean.min()),
            "max": self._safe_float(clean.max()),
            "median": self._safe_float(clean.median()),
            "q1": self._safe_float(clean.quantile(0.25)),
            "q3": self._safe_float(clean.quantile(0.75)),
            "skewness": self._safe_float(clean.skew()),
            "kurtosis": self._safe_float(clean.kurtosis()),
            "zeros": int((clean == 0).sum()),
            "negatives": int((clean < 0).sum()),
        }

        # Detect outliers using IQR
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
        stats["outlier_count"] = len(outliers)
        stats["outlier_percentage"] = round(len(outliers) / len(clean) * 100, 2) if len(clean) > 0 else 0

        return stats

    def _boolean_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for boolean columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        true_count = int(clean.sum())
        false_count = len(clean) - true_count

        return {
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": round(true_count / len(clean) * 100, 2) if len(clean) > 0 else 0,
        }

    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for datetime columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        return {
            "min": str(clean.min()),
            "max": str(clean.max()),
            "range_days": (clean.max() - clean.min()).days if len(clean) > 0 else 0,
        }

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute statistics for categorical/text columns."""
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}

        value_counts = clean.value_counts()
        top_values = value_counts.head(10).to_dict()

        return {
            "top_values": {str(k): int(v) for k, v in top_values.items()},
            "avg_length": round(clean.str.len().mean(), 2),
            "min_length": int(clean.str.len().min()),
            "max_length": int(clean.str.len().max()),
        }

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values across the dataset."""
        missing_counts = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()

        return {
            "total_missing": int(total_missing),
            "total_cells": total_cells,
            "missing_percentage": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
            "columns_with_missing": int((missing_counts > 0).sum()),
            "complete_rows": int((~df.isna().any(axis=1)).sum()),
            "complete_row_percentage": round((~df.isna().any(axis=1)).sum() / len(df) * 100, 2) if len(df) > 0 else 0,
            "by_column": {
                str(col): {
                    "count": int(missing_counts[col]),
                    "percentage": round(missing_counts[col] / len(df) * 100, 2) if len(df) > 0 else 0
                }
                for col in df.columns if missing_counts[col] > 0
            }
        }

    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation matrix for numeric columns."""
        # Exclude boolean columns from correlation
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns.tolist()
            if not pd.api.types.is_bool_dtype(df[col])
        ]

        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation"}

        # Limit to first 20 numeric columns for performance
        numeric_cols = numeric_cols[:20]
        corr_matrix = df[numeric_cols].corr()

        # Convert to serializable format
        correlations = {}
        for col1 in numeric_cols:
            correlations[col1] = {}
            for col2 in numeric_cols:
                val = corr_matrix.loc[col1, col2]
                correlations[col1][col2] = self._safe_float(val)

        # Find highly correlated pairs (excluding self-correlation)
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val > 0.7:
                    high_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": self._safe_float(corr_matrix.loc[col1, col2])
                    })

        return {
            "matrix": correlations,
            "high_correlations": high_correlations,
            "columns_analyzed": numeric_cols
        }

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality and identify issues."""
        issues = []

        # Check for high missing value columns
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100 if len(df) > 0 else 0
            if missing_pct > 50:
                issues.append({
                    "type": "high_missing",
                    "column": str(col),
                    "severity": "high",
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values"
                })
            elif missing_pct > 20:
                issues.append({
                    "type": "moderate_missing",
                    "column": str(col),
                    "severity": "medium",
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values"
                })

        # Check for high cardinality categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio > 0.9 and df[col].nunique() > 100:
                issues.append({
                    "type": "high_cardinality",
                    "column": str(col),
                    "severity": "low",
                    "message": f"Column '{col}' has very high cardinality ({df[col].nunique()} unique values)"
                })

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append({
                    "type": "constant_column",
                    "column": str(col),
                    "severity": "medium",
                    "message": f"Column '{col}' has only one unique value"
                })

        # Check for duplicate rows
        dup_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
        if dup_pct > 10:
            issues.append({
                "type": "high_duplicates",
                "severity": "medium",
                "message": f"Dataset has {dup_pct:.1f}% duplicate rows"
            })

        # Calculate quality score (simple heuristic)
        total_issues = len(issues)
        high_severity = len([i for i in issues if i.get("severity") == "high"])
        medium_severity = len([i for i in issues if i.get("severity") == "medium"])

        quality_score = max(0, 100 - (high_severity * 20) - (medium_severity * 10) - (total_issues * 2))

        return {
            "quality_score": quality_score,
            "total_issues": total_issues,
            "issues": issues
        }

    def _get_value_distributions(self, df: pd.DataFrame, max_bins: int = 20) -> Dict[str, Any]:
        """Get value distributions for visualization."""
        distributions = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if pd.api.types.is_numeric_dtype(series):
                # Histogram for numeric
                try:
                    hist, bin_edges = np.histogram(series, bins=min(max_bins, len(series.unique())))
                    distributions[str(col)] = {
                        "type": "histogram",
                        "counts": hist.tolist(),
                        "bin_edges": [self._safe_float(e) for e in bin_edges.tolist()]
                    }
                except Exception:
                    pass
            else:
                # Value counts for categorical (top 20)
                value_counts = series.astype(str).value_counts().head(20)
                distributions[str(col)] = {
                    "type": "bar",
                    "labels": value_counts.index.tolist(),
                    "counts": value_counts.values.tolist()
                }

        return distributions

    def _safe_float(self, val) -> Optional[float]:
        """Convert to float, handling NaN and inf."""
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), 4)

    async def get_profile(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get existing profile for a dataset, or generate one."""
        # Check for existing profile
        result = await db.execute(
            select(AnalysisResult)
            .where(AnalysisResult.dataset_id == dataset_id)
            .where(AnalysisResult.analysis_type == "profile")
            .order_by(AnalysisResult.created_at.desc())
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing.results

        # Generate new profile
        profile_result = await self.profile_dataset(dataset_id, db)
        if profile_result:
            return profile_result.results

        return None


# Singleton instance
_profiling_service: Optional[DataProfilingService] = None


def get_profiling_service() -> DataProfilingService:
    """Get the singleton DataProfilingService instance."""
    global _profiling_service
    if _profiling_service is None:
        _profiling_service = DataProfilingService()
    return _profiling_service
