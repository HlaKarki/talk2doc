from typing import List
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from database.models import DocumentChunk


async def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    :param text: The text to chunk
    :return: List of text chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI

    :param texts: List of text strings to embed
    :return: List of embeddings
    """
    embeddings_model = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
    )
    embeddings = await embeddings_model.aembed_documents(texts)
    return embeddings


async def store_chunks(
    document_id: UUID,
    chunks: List[str],
    embeddings: List[List[float]],
    db: AsyncSession
) -> List[DocumentChunk]:
    """
    Store text chunks with their embeddings in the database
    :param document_id: the UUID of the parent document
    :param chunks: List of text chunks
    :param embeddings: List of embedding vectors
    :param db: Database session
    :return: List of created DocumentChunk objects
    """

    chunk_objects = []
    for idx, (chunked_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=idx,
            content=chunked_text,
            embedding=embedding,
            chunk_metadata={
                "chunk_length": len(chunked_text),
                "chunk_position": idx,
            }
        )

        db.add(chunk)
        chunk_objects.append(chunk)

    await db.commit()
    return chunk_objects

async def process_document_text(
    document_id: UUID,
    text: str,
    db: AsyncSession
) -> int:
    """
    Complete pipeline: chunk text, generate embeddings, and store.
    :param document_id:  The UUID of the parent document
    :param text: The extracted text to process
    :param db: Database session
    :return: Number of chunks created
    """
    # Chunk the text
    chunks = await chunk_text(text)
    if not chunks:
        return 0
    # Generate embeddings
    embeddings = await generate_embeddings(chunks)
    # Store chunks with embeddings
    await store_chunks(document_id, chunks, embeddings, db)

    return len(chunks)