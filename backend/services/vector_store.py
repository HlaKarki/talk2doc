from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config


@dataclass
class SearchResult:
    """Search result with content and metadata."""
    content: str
    score: float
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    chunk_metadata: dict


async def get_query_embedding(query: str) -> List[float]:
    """Generate embedding for a search query."""
    embeddings_model = OpenAIEmbeddings(model=config.embedding_model)
    embedding = await embeddings_model.aembed_query(query)
    return embedding


async def similarity_search(
    query: str,
    db: AsyncSession,
    k: int = 5,
    document_id: Optional[UUID] = None,
) -> List[SearchResult]:
    """Find chunks most similar to the query."""
    query_embedding = await get_query_embedding(query)
    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    if document_id:
        sql = text("""
            SELECT
                id,
                document_id,
                chunk_index,
                content,
                chunk_metadata,
                embedding <=> :embedding AS distance
            FROM document_chunks
            WHERE document_id = :doc_id
            ORDER BY distance
            LIMIT :limit
        """)
        result = await db.execute(
            sql,
            {"embedding": embedding_str, "doc_id": str(document_id), "limit": k}
        )
    else:
        sql = text("""
            SELECT
                id,
                document_id,
                chunk_index,
                content,
                chunk_metadata,
                embedding <=> :embedding AS distance
            FROM document_chunks
            ORDER BY distance
            LIMIT :limit
        """)
        result = await db.execute(sql, {"embedding": embedding_str, "limit": k})

    rows = result.fetchall()

    results = [
        SearchResult(
            content=row.content,
            score=(1 - row.distance),
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            chunk_metadata=row.chunk_metadata or {}
        ) for row in rows
    ]

    return results


async def similarity_search_with_score(
    query: str,
    db: AsyncSession,
    k: int = 5,
    score_threshold: float = 0.0,
    document_id: Optional[UUID] = None
) -> List[SearchResult]:
    """Find similar chunks with minimum score threshold."""
    results = await similarity_search(query, db, k, document_id)
    return [r for r in results if r.score >= score_threshold]
