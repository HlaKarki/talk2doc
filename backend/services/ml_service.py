"""Machine Learning service for training and prediction."""
import io
import uuid
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import MLModel
from services.data_service import get_data_service
from services.storage import get_storage_service


# Supported algorithms
ALGORITHMS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
}

# Default hyperparameters
DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "svm": {"kernel": "rbf", "probability": True, "random_state": 42},
}


class MLService:
    """Service for training and using ML models."""

    def __init__(self):
        self.data_service = get_data_service()
        self.storage = get_storage_service()

    async def train_classifier(
        self,
        dataset_id: uuid.UUID,
        target_column: str,
        feature_columns: List[str],
        algorithm: Literal["random_forest", "logistic_regression", "svm"] = "random_forest",
        parameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        db: AsyncSession = None
    ) -> MLModel:
        """
        Train a classification model on the dataset.

        :param dataset_id: Dataset UUID
        :param target_column: Column to predict
        :param feature_columns: Columns to use as features
        :param algorithm: Classification algorithm to use
        :param parameters: Optional hyperparameters (overrides defaults)
        :param test_size: Fraction of data for testing (default 0.2)
        :param db: Database session
        :return: Trained MLModel record
        """
        # Load dataset
        df = await self.data_service.get_dataframe(dataset_id, db)
        if df is None:
            raise ValueError("Dataset not found")

        # Validate columns
        missing_cols = set(feature_columns + [target_column]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")

        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Drop rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            raise ValueError("Need at least 10 samples for training")

        # Preprocess features
        X_processed, preprocessing_info = self._preprocess_features(X)

        # Encode target if categorical
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            preprocessing_info["target_classes"] = target_encoder.classes_.tolist()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )

        # Get algorithm and parameters
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: {list(ALGORITHMS.keys())}")

        model_class = ALGORITHMS[algorithm]
        model_params = DEFAULT_PARAMS[algorithm].copy()
        if parameters:
            model_params.update(parameters)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='accuracy')

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "matrix": cm.tolist(),
            "labels": preprocessing_info.get("target_classes", list(range(cm.shape[0])))
        }

        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_columns, algorithm)

        # Serialize model and preprocessing info
        model_bundle = {
            "model": model,
            "preprocessing": preprocessing_info,
            "target_encoder": target_encoder,
            "feature_columns": feature_columns,
            "target_column": target_column,
        }

        # Save to R2
        model_bytes = pickle.dumps(model_bundle)
        artifact_key = self.storage.generate_key("models", f"{algorithm}.pkl")
        await self.storage.upload(model_bytes, artifact_key, content_type="application/octet-stream")

        # Create database record
        ml_model = MLModel(
            dataset_id=dataset_id,
            model_type="classifier",
            algorithm=algorithm,
            target_column=target_column,
            feature_columns=feature_columns,
            parameters=model_params,
            preprocessing=preprocessing_info,
            metrics=metrics,
            feature_importance=feature_importance,
            confusion_matrix=cm_dict,
            artifact_path=artifact_key,
            status="trained",
            trained_at=datetime.utcnow()
        )

        db.add(ml_model)
        await db.flush()

        return ml_model

    def _preprocess_features(self, X: pd.DataFrame) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess features: encode categorical, scale numeric, handle missing.

        Returns processed array and preprocessing info for later use.
        """
        preprocessing_info = {
            "columns": list(X.columns),
            "encoders": {},
            "scaler_mean": None,
            "scaler_scale": None,
        }

        X_processed = X.copy()

        # Handle each column
        for col in X.columns:
            # Check if column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(X[col])

            if not is_numeric:
                # Encode categorical/string columns
                encoder = LabelEncoder()
                # Handle missing values for categorical
                X_processed[col] = X_processed[col].fillna("__MISSING__").astype(str)
                X_processed[col] = encoder.fit_transform(X_processed[col])
                preprocessing_info["encoders"][col] = encoder.classes_.tolist()
            else:
                # Fill numeric missing with median
                median_val = float(X_processed[col].median())
                X_processed[col] = X_processed[col].fillna(median_val)
                preprocessing_info["encoders"][col] = {"type": "numeric", "fill_value": median_val}

        # Scale all features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        preprocessing_info["scaler_mean"] = scaler.mean_.tolist()
        preprocessing_info["scaler_scale"] = scaler.scale_.tolist()

        return X_scaled, preprocessing_info

    def _get_feature_importance(
        self,
        model,
        feature_columns: List[str],
        algorithm: str
    ) -> List[Dict[str, Any]]:
        """Extract feature importance from the model."""
        importance_scores = None

        if algorithm == "random_forest":
            importance_scores = model.feature_importances_
        elif algorithm == "logistic_regression":
            # Use absolute coefficients as importance
            importance_scores = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0])
        elif algorithm == "svm":
            # SVM doesn't have straightforward feature importance
            # Return equal importance as placeholder
            importance_scores = np.ones(len(feature_columns)) / len(feature_columns)

        if importance_scores is not None:
            # Normalize to sum to 1
            importance_scores = importance_scores / importance_scores.sum()

            return [
                {"feature": col, "importance": round(float(score), 4)}
                for col, score in sorted(
                    zip(feature_columns, importance_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

        return []

    async def predict(
        self,
        model_id: uuid.UUID,
        data: List[Dict[str, Any]],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.

        :param model_id: MLModel UUID
        :param data: List of feature dictionaries
        :param db: Database session
        :return: Predictions and probabilities
        """
        # Load model record
        result = await db.execute(
            select(MLModel).where(MLModel.id == model_id)
        )
        ml_model = result.scalar_one_or_none()

        if ml_model is None:
            raise ValueError("Model not found")

        # Download and deserialize model
        model_bytes = await self.storage.download(ml_model.artifact_path)
        model_bundle = pickle.loads(model_bytes)

        model = model_bundle["model"]
        preprocessing = model_bundle["preprocessing"]
        target_encoder = model_bundle["target_encoder"]
        feature_columns = model_bundle["feature_columns"]

        # Convert input to DataFrame
        df = pd.DataFrame(data)

        # Validate columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        # Select and order columns
        X = df[feature_columns].copy()

        # Apply preprocessing
        X_processed = self._apply_preprocessing(X, preprocessing)

        # Predict
        predictions = model.predict(X_processed)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_processed)
            probabilities = proba.tolist()

        # Decode predictions if we have a target encoder
        if target_encoder is not None:
            predictions = target_encoder.inverse_transform(predictions)

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "probabilities": probabilities,
            "classes": preprocessing.get("target_classes"),
        }

    def _apply_preprocessing(self, X: pd.DataFrame, preprocessing: Dict[str, Any]) -> np.ndarray:
        """Apply saved preprocessing to new data."""
        X_processed = X.copy()

        # Apply encoding
        for col in X.columns:
            encoder_info = preprocessing["encoders"].get(col)
            if encoder_info is None:
                continue

            if isinstance(encoder_info, list):
                # Categorical encoding
                X_processed[col] = X_processed[col].fillna("__MISSING__")
                # Map to encoded values
                class_to_idx = {cls: idx for idx, cls in enumerate(encoder_info)}
                X_processed[col] = X_processed[col].map(
                    lambda x: class_to_idx.get(x, 0)  # Default to 0 for unknown
                )
            elif isinstance(encoder_info, dict) and encoder_info.get("type") == "numeric":
                # Numeric - fill with saved value
                X_processed[col] = X_processed[col].fillna(encoder_info["fill_value"])

        # Apply scaling
        X_array = X_processed.values.astype(float)
        mean = np.array(preprocessing["scaler_mean"])
        scale = np.array(preprocessing["scaler_scale"])
        X_scaled = (X_array - mean) / scale

        return X_scaled

    async def get_model(self, model_id: uuid.UUID, db: AsyncSession) -> Optional[MLModel]:
        """Get a model by ID."""
        result = await db.execute(
            select(MLModel).where(MLModel.id == model_id)
        )
        return result.scalar_one_or_none()

    async def list_models(
        self,
        dataset_id: Optional[uuid.UUID],
        db: AsyncSession
    ) -> List[MLModel]:
        """List models, optionally filtered by dataset."""
        query = select(MLModel).order_by(MLModel.created_at.desc())
        if dataset_id:
            query = query.where(MLModel.dataset_id == dataset_id)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def delete_model(self, model_id: uuid.UUID, db: AsyncSession) -> bool:
        """Delete a model and its artifact."""
        ml_model = await self.get_model(model_id, db)
        if ml_model is None:
            return False

        # Delete artifact from R2
        try:
            await self.storage.delete(ml_model.artifact_path)
        except Exception as e:
            print(f"Warning: Could not delete model artifact: {e}")

        # Delete from database
        await db.delete(ml_model)
        return True

    # =========================================================================
    # Clustering Methods
    # =========================================================================

    async def cluster_data(
        self,
        dataset_id: uuid.UUID,
        feature_columns: List[str],
        algorithm: Literal["kmeans", "dbscan", "hierarchical"] = "kmeans",
        n_clusters: Optional[int] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Perform clustering on dataset features.

        :param dataset_id: Dataset UUID
        :param feature_columns: Columns to use for clustering
        :param algorithm: Clustering algorithm (kmeans, dbscan, hierarchical)
        :param n_clusters: Number of clusters (auto-detected if None for kmeans)
        :param db: Database session
        :return: Clustering results with labels, metrics, and visualization data
        """
        # Load dataset
        df = await self.data_service.get_dataframe(dataset_id, db)
        if df is None:
            raise ValueError("Dataset not found")

        # Validate columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")

        # Prepare features
        X = df[feature_columns].copy()

        # Preprocess (handle missing, scale)
        X_processed, preprocessing_info = self._preprocess_for_clustering(X)

        if len(X_processed) < 10:
            raise ValueError("Need at least 10 samples for clustering")

        # Determine optimal clusters if not specified (for kmeans)
        if algorithm == "kmeans" and n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_processed)

        # Perform clustering
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_processed)
            centroids = model.cluster_centers_.tolist()
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_processed)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            centroids = None
        elif algorithm == "hierarchical":
            if n_clusters is None:
                n_clusters = 3  # Default for hierarchical
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_processed)
            centroids = None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Calculate metrics (only if we have valid clusters)
        unique_labels = set(labels)
        n_valid_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        metrics = {}
        if n_valid_clusters >= 2:
            # Filter out noise points (-1) for metrics
            valid_mask = labels != -1
            if valid_mask.sum() >= 2:
                metrics["silhouette_score"] = round(
                    silhouette_score(X_processed[valid_mask], labels[valid_mask]), 4
                )
                metrics["calinski_harabasz_score"] = round(
                    calinski_harabasz_score(X_processed[valid_mask], labels[valid_mask]), 4
                )
                metrics["davies_bouldin_score"] = round(
                    davies_bouldin_score(X_processed[valid_mask], labels[valid_mask]), 4
                )

        # Generate cluster profiles
        cluster_profiles = self._generate_cluster_profiles(
            df[feature_columns], labels, feature_columns
        )

        # PCA for visualization (reduce to 2D)
        viz_data = self._generate_cluster_visualization(X_processed, labels)

        return {
            "algorithm": algorithm,
            "n_clusters": int(n_valid_clusters),
            "cluster_labels": [int(x) for x in labels],
            "metrics": metrics,
            "cluster_profiles": cluster_profiles,
            "centroids": centroids,
            "visualization": viz_data,
            "feature_columns": feature_columns,
            "total_samples": int(len(labels)),
            "noise_points": int((labels == -1).sum()) if algorithm == "dbscan" else 0
        }

    def _preprocess_for_clustering(self, X: pd.DataFrame) -> tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess features for clustering (numeric only, scaled)."""
        preprocessing_info = {"columns": list(X.columns)}

        X_processed = X.copy()

        # Handle each column
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # Encode categorical
                encoder = LabelEncoder()
                X_processed[col] = X_processed[col].fillna("__MISSING__").astype(str)
                X_processed[col] = encoder.fit_transform(X_processed[col])
            else:
                # Fill numeric missing with median
                median_val = X_processed[col].median()
                X_processed[col] = X_processed[col].fillna(median_val)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        return X_scaled, preprocessing_info

    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method with silhouette."""
        max_k = min(max_k, len(X) - 1)
        if max_k < 2:
            return 2

        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        # Find k with highest silhouette score
        best_idx = np.argmax(silhouette_scores)
        optimal_k = k_range[best_idx]

        return optimal_k

    def _generate_cluster_profiles(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate statistical profiles for each cluster."""
        profiles = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            if label == -1:
                # Noise points in DBSCAN
                cluster_name = "noise"
            else:
                cluster_name = f"cluster_{label}"

            mask = labels == label
            cluster_data = df[mask]

            profile = {
                "cluster": int(label),
                "name": cluster_name,
                "size": int(mask.sum()),
                "percentage": round(float(mask.sum()) / len(labels) * 100, 2),
                "feature_stats": {}
            }

            # Calculate stats for each feature
            for col in feature_columns:
                col_data = cluster_data[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    profile["feature_stats"][col] = {
                        "mean": round(float(col_data.mean()), 4),
                        "std": round(float(col_data.std()), 4),
                        "min": round(float(col_data.min()), 4),
                        "max": round(float(col_data.max()), 4),
                    }
                else:
                    # For categorical, show mode
                    mode_val = col_data.mode()
                    profile["feature_stats"][col] = {
                        "mode": str(mode_val.iloc[0]) if len(mode_val) > 0 else None,
                        "unique_count": int(col_data.nunique())
                    }

            profiles.append(profile)

        return profiles

    def _generate_cluster_visualization(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Generate 2D visualization data using PCA."""
        # Reduce to 2D using PCA
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X)

        return {
            "x": [float(x) for x in coords[:, 0]],
            "y": [float(y) for y in coords[:, 1]] if n_components > 1 else [0.0] * len(coords),
            "labels": [int(l) for l in labels],
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_]
        }


# Singleton instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get the singleton MLService instance."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
