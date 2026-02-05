"""Document routes for file upload and management"""
from typing import List
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database.database import get_db
from services import document_service

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a PDF document.
    - **file**: PDF file to upload (max 50MB)

    Returns the created document metadata.
    """
    document = await document_service.upload_document(file, db)

    return {
        "id": str(document.id),
        "filename": document.filename,
        "file_type": document.file_type,
        "status": document.status,
        "upload_date": document.upload_date.isoformat(),
        "metadata": document.doc_metadata,
    }

@router.get("", response_model=List[dict])
async def get_documents(db: AsyncSession = Depends(get_db)):
    """
    Get all uploaded documents.

    :param db: Database session
    :return: A list of all documents with their metadata.
    """
    documents = await document_service.get_document(db)

    return [
        {
            "id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "status": document.status,
            "upload_date": document.upload_date.isoformat(),
            "metadata": document.doc_metadata,
        } for document in documents
    ]

@router.get("/{document_id}", response_model=dict)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific document by ID."""

    document = await document_service.get_document_by_id(document_id, db)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": str(document.id),
        "filename": document.filename,
        "file_type": document.file_type,
        "status": document.status,
        "upload_date": document.upload_date.isoformat(),
        "metadata": document.doc_metadata,
    }

@router.delete("/{document_id}", response_model=dict)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document by ID."""
    success = await document_service.delete_document(document_id, db)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "success": True,
        "message": "Document deleted successfully"
    }
