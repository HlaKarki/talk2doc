"""Service for generating Plotly visualizations."""
from typing import Any, Dict, List, Optional
import uuid
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Dataset
from services.storage import get_storage_service


def fig_to_dict(fig) -> Dict[str, Any]:
    """Convert Plotly figure to JSON-serializable dict."""
    # Use Plotly's JSON encoder which handles numpy arrays properly
    return json.loads(fig.to_json())


class VisualizationService:
    """Service for generating data visualizations using Plotly."""

    async def _load_dataset(
        self, dataset_id: uuid.UUID, db: AsyncSession
    ) -> pd.DataFrame:
        """Load dataset from storage."""
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        storage = get_storage_service()
        file_bytes = await storage.download(dataset.file_path)

        if dataset.file_type in ("text/csv", "csv"):
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(file_bytes))

        return df

    def generate_distribution_plot(
        self,
        df: pd.DataFrame,
        column: str,
        plot_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate distribution plot for a column.

        Args:
            df: DataFrame with the data
            column: Column name to visualize
            plot_type: 'histogram', 'box', 'violin', or 'auto' (auto-detect)

        Returns:
            Plotly figure as JSON dict
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")

        data = df[column].dropna()

        # Auto-detect plot type based on data
        if plot_type == "auto":
            if pd.api.types.is_numeric_dtype(data):
                unique_ratio = len(data.unique()) / len(data)
                plot_type = "histogram" if unique_ratio > 0.1 else "box"
            else:
                plot_type = "bar"  # Categorical data

        if plot_type == "histogram":
            fig = px.histogram(
                df, x=column,
                title=f"Distribution of {column}",
                labels={column: column},
                template="plotly_white"
            )
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count",
                showlegend=False
            )
        elif plot_type == "box":
            fig = px.box(
                df, y=column,
                title=f"Box Plot of {column}",
                template="plotly_white"
            )
        elif plot_type == "violin":
            fig = px.violin(
                df, y=column,
                title=f"Violin Plot of {column}",
                template="plotly_white",
                box=True
            )
        else:  # bar chart for categorical
            value_counts = df[column].value_counts().head(20)
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Distribution of {column}",
                labels={"x": column, "y": "Count"},
                template="plotly_white"
            )

        return fig_to_dict(fig)

    def generate_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate scatter plot for two columns.

        Args:
            df: DataFrame with the data
            x_col: X-axis column
            y_col: Y-axis column
            color_by: Optional column to color points by
            size_by: Optional column to size points by

        Returns:
            Plotly figure as JSON dict
        """
        for col in [x_col, y_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset")

        if color_by and color_by not in df.columns:
            raise ValueError(f"Column '{color_by}' not found in dataset")

        if size_by and size_by not in df.columns:
            raise ValueError(f"Column '{size_by}' not found in dataset")

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_by,
            size=size_by,
            title=f"{y_col} vs {x_col}",
            template="plotly_white",
            hover_data=df.columns.tolist()[:5]  # Show first 5 columns on hover
        )

        return fig_to_dict(fig)

    def generate_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate correlation matrix heatmap.

        Args:
            df: DataFrame with the data
            columns: Optional list of columns to include (defaults to all numeric)

        Returns:
            Plotly figure as JSON dict
        """
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        if columns:
            missing = [c for c in columns if c not in numeric_df.columns]
            if missing:
                raise ValueError(f"Columns not found or not numeric: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty or len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")

        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix",
            template="plotly_white",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title=""
        )

        return fig_to_dict(fig)

    def generate_cluster_visualization(
        self,
        coords: np.ndarray,
        labels: List[int],
        feature_columns: List[str],
        centroids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate 2D cluster visualization from PCA coordinates.

        Args:
            coords: 2D coordinates from PCA (n_samples, 2)
            labels: Cluster labels for each point
            feature_columns: Original feature names for title
            centroids: Optional cluster centroids in PCA space

        Returns:
            Plotly figure as JSON dict
        """
        df_viz = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]
        })

        fig = px.scatter(
            df_viz, x="PC1", y="PC2",
            color="Cluster",
            title=f"Cluster Visualization (Features: {', '.join(feature_columns[:3])}{'...' if len(feature_columns) > 3 else ''})",
            template="plotly_white"
        )

        # Add centroids if provided
        if centroids is not None and len(centroids) > 0:
            fig.add_trace(go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=15,
                    color="black",
                    line=dict(width=2)
                ),
                name="Centroids"
            ))

        fig.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2"
        )

        return fig_to_dict(fig)

    def generate_feature_importance_chart(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        top_n: int = 15
    ) -> Dict[str, Any]:
        """
        Generate horizontal bar chart for feature importance.

        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            top_n: Number of top features to show

        Returns:
            Plotly figure as JSON dict
        """
        # Create dataframe and sort by importance
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_scores
        }).sort_values("Importance", ascending=True).tail(top_n)

        fig = px.bar(
            df, x="Importance", y="Feature",
            orientation="h",
            title=f"Top {min(top_n, len(df))} Feature Importance",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            showlegend=False
        )

        return fig_to_dict(fig)

    def generate_time_series_plot(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate time series line chart.

        Args:
            df: DataFrame with the data
            date_col: Column containing dates
            value_col: Column with values to plot
            group_by: Optional column to create multiple lines

        Returns:
            Plotly figure as JSON dict
        """
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in dataset")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in dataset")

        # Try to parse dates
        df_copy = df.copy()
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be parsed as dates")

        df_copy = df_copy.sort_values(date_col)

        fig = px.line(
            df_copy, x=date_col, y=value_col,
            color=group_by,
            title=f"{value_col} Over Time",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title=date_col,
            yaxis_title=value_col
        )

        return fig_to_dict(fig)

    def generate_pairplot(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        color_by: Optional[str] = None,
        max_columns: int = 5
    ) -> Dict[str, Any]:
        """
        Generate pairwise scatter plot matrix.

        Args:
            df: DataFrame with the data
            columns: Columns to include (defaults to first N numeric)
            color_by: Optional column to color points by
            max_columns: Maximum number of columns to include

        Returns:
            Plotly figure as JSON dict
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            columns = [c for c in columns if c in numeric_cols]
        else:
            columns = numeric_cols[:max_columns]

        if len(columns) < 2:
            raise ValueError("Need at least 2 numeric columns for pairplot")

        fig = px.scatter_matrix(
            df, dimensions=columns,
            color=color_by,
            title="Pairwise Relationships",
            template="plotly_white"
        )

        fig.update_traces(diagonal_visible=False)

        return fig_to_dict(fig)

    def generate_confusion_matrix_plot(
        self,
        matrix: List[List[int]],
        labels: List[str]
    ) -> Dict[str, Any]:
        """
        Generate confusion matrix heatmap.

        Args:
            matrix: Confusion matrix as 2D list
            labels: Class labels

        Returns:
            Plotly figure as JSON dict
        """
        # Convert labels to strings
        labels = [str(l) for l in labels]

        fig = px.imshow(
            matrix,
            x=labels,
            y=labels,
            text_auto=True,
            title="Confusion Matrix",
            template="plotly_white",
            color_continuous_scale="Blues",
            aspect="equal"
        )

        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )

        return fig_to_dict(fig)

    async def auto_generate_visualizations(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession,
        max_charts: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate relevant visualizations based on data types.

        Args:
            dataset_id: Dataset ID
            db: Database session
            max_charts: Maximum number of charts to generate

        Returns:
            List of Plotly figures as JSON dicts with metadata
        """
        df = await self._load_dataset(dataset_id, db)
        charts = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = []

        # Detect datetime columns
        for col in df.columns:
            if col not in numeric_cols:
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except Exception:
                    pass

        # 1. Distribution plots for numeric columns
        for col in numeric_cols[:2]:
            if len(charts) >= max_charts:
                break
            charts.append({
                "type": "distribution",
                "column": col,
                "figure": self.generate_distribution_plot(df, col, "histogram")
            })

        # 2. Bar charts for categorical columns
        for col in categorical_cols[:2]:
            if len(charts) >= max_charts:
                break
            if df[col].nunique() <= 20:  # Only if not too many categories
                charts.append({
                    "type": "distribution",
                    "column": col,
                    "figure": self.generate_distribution_plot(df, col, "bar")
                })

        # 3. Correlation matrix if enough numeric columns
        if len(numeric_cols) >= 3 and len(charts) < max_charts:
            try:
                charts.append({
                    "type": "correlation",
                    "columns": numeric_cols[:10],
                    "figure": self.generate_correlation_matrix(df, numeric_cols[:10])
                })
            except Exception:
                pass

        # 4. Scatter plot of first two numeric columns
        if len(numeric_cols) >= 2 and len(charts) < max_charts:
            color_col = categorical_cols[0] if categorical_cols and df[categorical_cols[0]].nunique() <= 10 else None
            charts.append({
                "type": "scatter",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1],
                "color_by": color_col,
                "figure": self.generate_scatter_plot(df, numeric_cols[0], numeric_cols[1], color_col)
            })

        # 5. Time series if datetime detected
        if datetime_cols and numeric_cols and len(charts) < max_charts:
            try:
                charts.append({
                    "type": "time_series",
                    "date_column": datetime_cols[0],
                    "value_column": numeric_cols[0],
                    "figure": self.generate_time_series_plot(df, datetime_cols[0], numeric_cols[0])
                })
            except Exception:
                pass

        return charts


# Singleton instance
_visualization_service: Optional[VisualizationService] = None


def get_visualization_service() -> VisualizationService:
    """Get the visualization service singleton."""
    global _visualization_service
    if _visualization_service is None:
        _visualization_service = VisualizationService()
    return _visualization_service
