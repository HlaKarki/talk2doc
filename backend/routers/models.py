"""API endpoints for ML model management."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from services.ml_service import get_ml_service


router = APIRouter(prefix="/models", tags=["models"])


# Request models
class TrainClassifierRequest(BaseModel):
    """Request to train a classifier."""
    target_column: str
    feature_columns: List[str]
    algorithm: str = "random_forest"  # random_forest, logistic_regression, svm
    parameters: Optional[dict] = None
    test_size: float = 0.2


class PredictRequest(BaseModel):
    """Request to make predictions."""
    data: List[dict]


# Response models
class MetricsResponse(BaseModel):
    """Model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    train_samples: int
    test_samples: int


class FeatureImportanceItem(BaseModel):
    """Feature importance item."""
    feature: str
    importance: float


class ConfusionMatrixResponse(BaseModel):
    """Confusion matrix data."""
    matrix: List[List[int]]
    labels: List


class ModelResponse(BaseModel):
    """Response for a trained model."""
    id: str
    dataset_id: str
    model_type: str
    algorithm: str
    target_column: str
    feature_columns: List[str]
    metrics: dict
    feature_importance: List[dict]
    confusion_matrix: Optional[dict]
    status: str
    created_at: str
    trained_at: Optional[str]


class ModelListResponse(BaseModel):
    """Response for listing models."""
    models: List[ModelResponse]
    total: int


class TrainResponse(BaseModel):
    """Response after training a model."""
    model_id: str
    algorithm: str
    metrics: dict
    feature_importance: List[dict]
    confusion_matrix: dict
    message: str


class PredictResponse(BaseModel):
    """Response for predictions."""
    predictions: List
    probabilities: Optional[List[List[float]]]
    classes: Optional[List]


@router.post("/train/{dataset_id}", response_model=TrainResponse)
async def train_classifier(
    dataset_id: uuid.UUID,
    request: TrainClassifierRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Train a classification model on a dataset.

    Supported algorithms:
    - random_forest: Random Forest Classifier
    - logistic_regression: Logistic Regression
    - svm: Support Vector Machine

    The model is trained on 80% of data (by default) and evaluated on 20%.
    Cross-validation (5-fold) is also performed.
    """
    ml_service = get_ml_service()

    try:
        model = await ml_service.train_classifier(
            dataset_id=dataset_id,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            algorithm=request.algorithm,
            parameters=request.parameters,
            test_size=request.test_size,
            db=db
        )
        await db.commit()

        return TrainResponse(
            model_id=str(model.id),
            algorithm=model.algorithm,
            metrics=model.metrics,
            feature_importance=model.feature_importance,
            confusion_matrix=model.confusion_matrix,
            message=f"Model trained successfully with {model.metrics['accuracy']*100:.1f}% accuracy"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(
    model_id: uuid.UUID,
    request: PredictRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Make predictions using a trained model.

    Provide data as a list of dictionaries with feature values.
    Returns predictions and class probabilities (if available).
    """
    ml_service = get_ml_service()

    try:
        result = await ml_service.predict(model_id, request.data, db)
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("", response_model=ModelListResponse)
async def list_models(
    dataset_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all trained models.

    Optionally filter by dataset_id.
    """
    ml_service = get_ml_service()
    models = await ml_service.list_models(dataset_id, db)

    return ModelListResponse(
        models=[
            ModelResponse(
                id=str(m.id),
                dataset_id=str(m.dataset_id),
                model_type=m.model_type,
                algorithm=m.algorithm,
                target_column=m.target_column,
                feature_columns=m.feature_columns,
                metrics=m.metrics,
                feature_importance=m.feature_importance,
                confusion_matrix=m.confusion_matrix,
                status=m.status,
                created_at=m.created_at.isoformat(),
                trained_at=m.trained_at.isoformat() if m.trained_at else None
            )
            for m in models
        ],
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific model."""
    ml_service = get_ml_service()
    model = await ml_service.get_model(model_id, db)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelResponse(
        id=str(model.id),
        dataset_id=str(model.dataset_id),
        model_type=model.model_type,
        algorithm=model.algorithm,
        target_column=model.target_column,
        feature_columns=model.feature_columns,
        metrics=model.metrics,
        feature_importance=model.feature_importance,
        confusion_matrix=model.confusion_matrix,
        status=model.status,
        created_at=model.created_at.isoformat(),
        trained_at=model.trained_at.isoformat() if model.trained_at else None
    )


@router.delete("/{model_id}")
async def delete_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a model and its stored artifact."""
    ml_service = get_ml_service()
    deleted = await ml_service.delete_model(model_id, db)

    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")

    await db.commit()
    return {"message": "Model deleted successfully", "id": str(model_id)}
