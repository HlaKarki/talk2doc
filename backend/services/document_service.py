import fitz
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.models import Document
from typing import List, Optional
from uuid import UUID

async def extract_pdf_text(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")

        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()

        pdf_document.close()
        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF text: {str(e)}")

async def upload_document(
        file: UploadFile,
        db: AsyncSession,
) -> Document:
    """Upload and process a document"""

    if not file.filename.lower().endswith((".pdf")):
        raise HTTPException(status_code=400, detail="File extension not supported")

    contents = await file.read()
    max_size = 50 * 1024 * 1024 # 50 MB
    if len(contents) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")

    # Extract text from PDF
    try:
        extracted_text = await extract_pdf_text(contents)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")


    # Create document record
    document = Document(
        filename=file.filename,
        file_type="application/pdf",
        status="pending",
        doc_metadata={
            "text_length": len(extracted_text),
            "original_size_bytes": len(contents),
        }
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    return document


async def get_document(db: AsyncSession) -> List[Document]:
    """Get all documents"""
    result = await db.execute(select(Document))
    return result.scalars().all()

async def get_document_by_id(document_id: UUID, db: AsyncSession) -> Optional[Document]:
    """Get document by id"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )

    return result.scalars().one_or_none()

async def delete_document(document_id: UUID, db: AsyncSession) -> bool:
    """Delete document by id"""
    document = await get_document_by_id(document_id, db)

    if not document:
        return False

    await db.delete(document)
    await db.commit()

    return True