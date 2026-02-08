"""Document routes for file upload and management"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from services import document_service
from services import vector_store
from services import rag_service


class SearchRequest(BaseModel):
    """Request body for search endpoints."""
    query: str
    k: int = 5
    score_threshold: Optional[float] = None


class QueryRequest(BaseModel):
    """Request body for RAG query endpoints."""
    query: str
    k: int = 5  # Number of chunks to retrieve for context
    preprocess: bool = False  # Whether to rewrite query for better retrieval
    rerank: bool = False  # Whether to re-rank chunks using LLM scoring

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


@router.post("/search", response_model=List[dict])
async def search_documents(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search across all documents using semantic similarity.

    - **query**: The search query text
    - **k**: Number of results to return (default: 5)
    - **score_threshold**: Minimum similarity score (0-1, optional)
    """
    if request.score_threshold is not None:
        results = await vector_store.similarity_search_with_score(
            query=request.query,
            db=db,
            k=request.k,
            score_threshold=request.score_threshold
        )
    else:
        results = await vector_store.similarity_search(
            query=request.query,
            db=db,
            k=request.k
        )

    return [
        {
            "content": r.content,
            "score": r.score,
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "chunk_index": r.chunk_index,
            "metadata": r.chunk_metadata
        }
        for r in results
    ]


@router.post("/{document_id}/search", response_model=List[dict])
async def search_document(
    document_id: UUID,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search within a specific document using semantic similarity.
    - **document_id**: UUID of the document to search in
    - **query**: The search query text
    - **k**: Number of results to return (default: 5)
    """
    # Verify document exists
    document = await document_service.get_document_by_id(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    results = await vector_store.similarity_search(
        query=request.query,
        db=db,
        k=request.k,
        document_id=document_id
    )

    return [
        {
            "content": r.content,
            "score": r.score,
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "chunk_index": r.chunk_index,
            "metadata": r.chunk_metadata
        }
        for r in results
    ]


@router.post("/query", response_model=dict)
async def query_all_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question across all documents using RAG.

    - **query**: Your question
    - **k**: Number of context chunks to use (default: 5)

    Returns an AI-generated answer with source citations.
    """
    response = await rag_service.query_documents(
        query=request.query,
        db=db,
        k=request.k,
        preprocess=request.preprocess,
        rerank=request.rerank
    )

    return {
        "answer": response.answer,
        "sources": [
            {
                "chunk_index": s.chunk_index,
                "content": s.content,
                "score": s.score,
                "document_id": str(s.document_id)
            }
            for s in response.sources
        ]
    }


@router.post("/{document_id}/query", response_model=dict)
async def query_document(
    document_id: UUID,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question about a specific document using RAG.

    - **document_id**: UUID of the document to query
    - **query**: Your question
    - **k**: Number of context chunks to use (default: 5)

    Returns an AI-generated answer with source citations.
    """
    # Verify document exists
    document = await document_service.get_document_by_id(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    response = await rag_service.query_documents(
        query=request.query,
        db=db,
        document_id=document_id,
        k=request.k,
        preprocess=request.preprocess,
        rerank=request.rerank
    )

    return {
        "answer": response.answer,
        "sources": [
            {
                "chunk_index": s.chunk_index,
                "content": s.content,
                "score": s.score,
                "document_id": str(s.document_id)
            }
            for s in response.sources
        ]
    }


@router.post("/query/stream")
async def stream_query_all_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream a question answer across all documents using RAG.

    - **query**: Your question
    - **k**: Number of context chunks to use (default: 5)

    Returns Server-Sent Events (SSE) stream with:
    - sources: Retrieved document chunks
    - token: Individual response tokens
    - [DONE]: End of stream marker
    """
    return StreamingResponse(
        rag_service.stream_query_documents(
            query=request.query,
            db=db,
            k=request.k,
            preprocess=request.preprocess,
            rerank=request.rerank
        ),
        media_type="text/event-stream"
    )


@router.post("/{document_id}/query/stream")
async def stream_query_document(
    document_id: UUID,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream a question answer about a specific document using RAG.

    - **document_id**: UUID of the document to query
    - **query**: Your question
    - **k**: Number of context chunks to use (default: 5)

    Returns Server-Sent Events (SSE) stream.
    """
    # Verify document exists
    document = await document_service.get_document_by_id(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return StreamingResponse(
        rag_service.stream_query_documents(
            query=request.query,
            db=db,
            document_id=document_id,
            k=request.k,
            preprocess=request.preprocess,
            rerank=request.rerank
        ),
        media_type="text/event-stream"
    )
