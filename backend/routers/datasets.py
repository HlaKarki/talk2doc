"""API endpoints for dataset management."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from services.data_service import get_data_service
from services.data_profiling_service import get_profiling_service


router = APIRouter(prefix="/datasets", tags=["datasets"])


# Response models
class ColumnSchema(BaseModel):
    """Schema for a single column."""
    name: str
    dtype: str
    semantic_type: str
    nullable: bool
    unique_count: int
    sample_values: List


class DatasetSchema(BaseModel):
    """Schema information for a dataset."""
    columns: List[ColumnSchema]
    column_names: List[str]


class DatasetResponse(BaseModel):
    """Response model for dataset information."""
    id: str
    filename: str
    file_type: str
    file_size: Optional[int]
    row_count: Optional[int]
    column_count: Optional[int]
    column_schema: Optional[dict] = None
    status: str
    error_message: Optional[str]
    upload_date: str
    profiled_at: Optional[str]

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Response for listing datasets."""
    datasets: List[DatasetResponse]
    total: int


class PreviewResponse(BaseModel):
    """Response for dataset preview."""
    columns: List[str]
    rows: List[dict]
    total_rows: int
    preview_rows: int


class ProfileResponse(BaseModel):
    """Response for dataset profile."""
    basic_stats: dict
    column_profiles: List[dict]
    missing_values: dict
    correlations: dict
    data_quality: dict
    value_distributions: dict


class UploadResponse(BaseModel):
    """Response after uploading a dataset."""
    id: str
    filename: str
    status: str
    row_count: Optional[int]
    column_count: Optional[int]
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a CSV or Excel file as a dataset.

    Supported formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    """
    data_service = get_data_service()

    try:
        dataset = await data_service.upload_dataset(file, db)
        await db.commit()

        return UploadResponse(
            id=str(dataset.id),
            filename=dataset.filename,
            status=dataset.status,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            message=f"Dataset uploaded successfully with {dataset.row_count} rows and {dataset.column_count} columns"
            if dataset.status == "uploaded"
            else f"Dataset upload failed: {dataset.error_message}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    db: AsyncSession = Depends(get_db)
):
    """List all uploaded datasets."""
    data_service = get_data_service()
    datasets = await data_service.get_datasets(db)

    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                id=str(ds.id),
                filename=ds.filename,
                file_type=ds.file_type,
                file_size=ds.file_size,
                row_count=ds.row_count,
                column_count=ds.column_count,
                column_schema=ds.schema,
                status=ds.status,
                error_message=ds.error_message,
                upload_date=ds.upload_date.isoformat(),
                profiled_at=ds.profiled_at.isoformat() if ds.profiled_at else None
            )
            for ds in datasets
        ],
        total=len(datasets)
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific dataset."""
    data_service = get_data_service()
    dataset = await data_service.get_dataset_by_id(dataset_id, db)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(
        id=str(dataset.id),
        filename=dataset.filename,
        file_type=dataset.file_type,
        file_size=dataset.file_size,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        column_schema=dataset.schema,
        status=dataset.status,
        error_message=dataset.error_message,
        upload_date=dataset.upload_date.isoformat(),
        profiled_at=dataset.profiled_at.isoformat() if dataset.profiled_at else None
    )


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    dataset_id: uuid.UUID,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a preview of the dataset rows.

    Args:
        dataset_id: The dataset UUID
        limit: Maximum number of rows to return (default: 100, max: 1000)
    """
    if limit > 1000:
        limit = 1000

    data_service = get_data_service()
    preview = await data_service.preview_dataset(dataset_id, db, limit)

    if preview is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return PreviewResponse(**preview)


@router.get("/{dataset_id}/profile", response_model=ProfileResponse)
async def get_dataset_profile(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the statistical profile of a dataset.

    Includes:
    - Basic stats (row count, memory usage, duplicates)
    - Per-column statistics (mean, std, min, max, etc.)
    - Missing value analysis
    - Correlation matrix
    - Data quality assessment
    - Value distributions
    """
    # Check if dataset exists
    data_service = get_data_service()
    dataset = await data_service.get_dataset_by_id(dataset_id, db)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get or generate profile
    profiling_service = get_profiling_service()
    profile = await profiling_service.get_profile(dataset_id, db)
    await db.commit()

    if profile is None:
        raise HTTPException(status_code=500, detail="Failed to generate profile")

    return ProfileResponse(**profile)


@router.post("/{dataset_id}/profile", response_model=ProfileResponse)
async def regenerate_profile(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Regenerate the statistical profile for a dataset.

    Use this to refresh the profile after data changes.
    """
    # Check if dataset exists
    data_service = get_data_service()
    dataset = await data_service.get_dataset_by_id(dataset_id, db)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Force regenerate profile
    profiling_service = get_profiling_service()
    result = await profiling_service.profile_dataset(dataset_id, db)
    await db.commit()

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate profile")

    return ProfileResponse(**result.results)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a dataset and its associated file."""
    data_service = get_data_service()
    deleted = await data_service.delete_dataset(dataset_id, db)

    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await db.commit()

    return {"message": "Dataset deleted successfully", "id": str(dataset_id)}
