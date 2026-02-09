"""Data service for managing dataset uploads and operations."""
import io
import uuid
from typing import List, Optional, Dict, Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import UploadFile

from database.models import Dataset, AnalysisResult
from services.storage import get_storage_service


# Supported file types
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Content types for R2
CONTENT_TYPES = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
}


class DataService:
    """Service for managing dataset uploads and operations."""

    def __init__(self):
        self.storage = get_storage_service()

    async def upload_dataset(
        self,
        file: UploadFile,
        db: AsyncSession
    ) -> Dataset:
        """
        Upload and process a dataset file.

        :param file: Uploaded file (CSV or Excel)
        :param db: Database session
        :return: Created Dataset record
        """
        # Validate file extension
        filename = file.filename or "unnamed"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ext = f".{ext}"

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Generate R2 key and upload
        r2_key = self.storage.generate_key("datasets", filename)
        await self.storage.upload(
            data=content,
            key=r2_key,
            content_type=CONTENT_TYPES.get(ext)
        )

        # Parse file to extract metadata
        try:
            df = self._read_from_bytes(content, ext)
            row_count = len(df)
            column_count = len(df.columns)
            schema = self._extract_schema(df)
            status = "uploaded"
            error_message = None
        except Exception as e:
            row_count = None
            column_count = None
            schema = {}
            status = "error"
            error_message = str(e)

        # Create database record (file_path stores R2 key)
        dataset = Dataset(
            filename=filename,
            file_path=r2_key,
            file_type=ext.lstrip("."),
            file_size=file_size,
            row_count=row_count,
            column_count=column_count,
            schema=schema,
            status=status,
            error_message=error_message
        )

        db.add(dataset)
        await db.flush()

        return dataset

    def _read_from_bytes(self, content: bytes, ext: str) -> pd.DataFrame:
        """Read file content into pandas DataFrame."""
        buffer = io.BytesIO(content)
        if ext == ".csv":
            return pd.read_csv(buffer)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(buffer)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def _extract_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema information from DataFrame."""
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)

            # Determine semantic type
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    semantic_type = "integer"
                else:
                    semantic_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                semantic_type = "datetime"
            elif pd.api.types.is_bool_dtype(df[col]):
                semantic_type = "boolean"
            else:
                # Check if it's categorical (few unique values relative to size)
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio < 0.05 and df[col].nunique() < 50:
                    semantic_type = "categorical"
                else:
                    semantic_type = "text"

            columns.append({
                "name": str(col),
                "dtype": dtype,
                "semantic_type": semantic_type,
                "nullable": bool(df[col].isna().any()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            })

        return {
            "columns": columns,
            "column_names": list(df.columns)
        }

    async def get_datasets(self, db: AsyncSession) -> List[Dataset]:
        """Get all datasets."""
        result = await db.execute(
            select(Dataset).order_by(Dataset.upload_date.desc())
        )
        return list(result.scalars().all())

    async def get_dataset_by_id(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession
    ) -> Optional[Dataset]:
        """Get a dataset by ID."""
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def delete_dataset(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession
    ) -> bool:
        """Delete a dataset and its file from R2."""
        dataset = await self.get_dataset_by_id(dataset_id, db)
        if not dataset:
            return False

        # Delete file from R2
        try:
            await self.storage.delete(dataset.file_path)
        except Exception as e:
            print(f"Warning: Could not delete R2 object {dataset.file_path}: {e}")

        # Delete from database
        await db.execute(
            delete(Dataset).where(Dataset.id == dataset_id)
        )

        return True

    async def get_dataframe(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession
    ) -> Optional[pd.DataFrame]:
        """Load dataset as pandas DataFrame from R2."""
        dataset = await self.get_dataset_by_id(dataset_id, db)
        if not dataset:
            return None

        # Download from R2
        content = await self.storage.download(dataset.file_path)
        ext = f".{dataset.file_type}"
        return self._read_from_bytes(content, ext)

    async def preview_dataset(
        self,
        dataset_id: uuid.UUID,
        db: AsyncSession,
        limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Get a preview of the dataset."""
        df = await self.get_dataframe(dataset_id, db)
        if df is None:
            return None

        # Get first N rows
        preview_df = df.head(limit)

        return {
            "columns": list(preview_df.columns),
            "rows": preview_df.to_dict(orient="records"),
            "total_rows": len(df),
            "preview_rows": len(preview_df)
        }


# Singleton instance
_data_service: Optional[DataService] = None


def get_data_service() -> DataService:
    """Get the singleton DataService instance."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service
