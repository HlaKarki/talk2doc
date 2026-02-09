"""Semantic memory service for storing past interactions with embeddings."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, text

from database.models import SemanticMemory
from services.embedding_service import generate_embeddings


@dataclass
class SemanticMemoryEntry:
    """Structured semantic memory entry."""
    id: str
    query: str
    response: str
    context_summary: Optional[str]
    intent: Optional[str]
    agent_used: Optional[str]
    relevance_score: float
    similarity: float  # Similarity to search query
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "context_summary": self.context_summary,
            "intent": self.intent,
            "agent_used": self.agent_used,
            "relevance_score": self.relevance_score,
            "similarity": self.similarity,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class SemanticMemoryService:
    """
    Service for semantic memory - storing and retrieving past interactions by similarity.

    Enables:
    - Storing query/response pairs with embeddings
    - Semantic similarity search over past interactions
    - Finding relevant past conversations for context
    """

    def __init__(self):
        pass

    async def store_interaction(
        self,
        db: AsyncSession,
        query: str,
        response: str,
        conversation_id: Optional[UUID] = None,
        context_summary: Optional[str] = None,
        intent: Optional[str] = None,
        agent_used: Optional[str] = None,
        relevance_score: float = 1.0
    ) -> SemanticMemory:
        """
        Store a query/response interaction with embedding.

        :param db: Database session
        :param query: User's query
        :param response: Assistant's response
        :param conversation_id: Related conversation
        :param context_summary: Summary of the interaction context
        :param intent: Classified intent
        :param agent_used: Which agent handled this
        :param relevance_score: How relevant/important this interaction is
        :return: Created semantic memory
        """
        # Generate embedding for the query
        embeddings = await generate_embeddings([query])
        query_embedding = embeddings[0] if embeddings else None

        memory = SemanticMemory(
            query=query,
            response=response,
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            context_summary=context_summary,
            intent=intent,
            agent_used=agent_used,
            relevance_score=relevance_score
        )

        db.add(memory)
        await db.flush()
        return memory

    async def search_similar_interactions(
        self,
        db: AsyncSession,
        query: str,
        k: int = 5,
        threshold: float = 0.5
    ) -> List[SemanticMemoryEntry]:
        """
        Search for similar past interactions using vector similarity.

        :param db: Database session
        :param query: Search query
        :param k: Number of results to return
        :param threshold: Minimum similarity threshold (0-1, higher = more similar)
        :return: List of similar past interactions
        """
        # Generate embedding for the search query
        embeddings = await generate_embeddings([query])
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # Vector similarity search using pgvector
        sql = """
            SELECT id, query, response, context_summary, intent, agent_used,
                   relevance_score, created_at, conversation_id,
                   1 - (query_embedding <=> :embedding) AS similarity
            FROM semantic_memories
            WHERE query_embedding IS NOT NULL
            ORDER BY query_embedding <=> :embedding
            LIMIT :limit
        """

        result = await db.execute(
            text(sql),
            {"embedding": embedding_str, "limit": k * 2}  # Fetch more to filter
        )
        rows = result.mappings().fetchall()

        # Filter by threshold and return top k
        memories = []
        for row in rows:
            similarity = row["similarity"]
            if similarity >= threshold:
                memories.append(SemanticMemoryEntry(
                    id=str(row["id"]),
                    query=row["query"],
                    response=row["response"],
                    context_summary=row["context_summary"],
                    intent=row["intent"],
                    agent_used=row["agent_used"],
                    relevance_score=row["relevance_score"],
                    similarity=similarity,
                    created_at=row["created_at"]
                ))

            if len(memories) >= k:
                break

        return memories

    async def get_relevant_memories(
        self,
        db: AsyncSession,
        query: str,
        k: int = 3,
        threshold: float = 0.6
    ) -> List[SemanticMemoryEntry]:
        """
        Get memories relevant to the current query.
        Higher threshold than search for more precise matching.

        :param db: Database session
        :param query: Current query
        :param k: Number of memories to return
        :param threshold: Minimum similarity threshold
        :return: Relevant past interactions
        """
        memories = await self.search_similar_interactions(
            db=db,
            query=query,
            k=k,
            threshold=threshold
        )

        # Update access tracking for retrieved memories
        for mem in memories:
            await self._update_access(db, UUID(mem.id))

        return memories

    async def _update_access(self, db: AsyncSession, memory_id: UUID) -> None:
        """Update access tracking for a memory."""
        result = await db.execute(
            select(SemanticMemory).where(SemanticMemory.id == memory_id)
        )
        memory = result.scalar_one_or_none()
        if memory:
            memory.last_accessed = datetime.utcnow()
            memory.access_count += 1
            await db.flush()

    async def get_by_conversation(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> List[SemanticMemoryEntry]:
        """Get all semantic memories for a conversation."""
        result = await db.execute(
            select(SemanticMemory)
            .where(SemanticMemory.conversation_id == conversation_id)
            .order_by(SemanticMemory.created_at)
        )
        memories = result.scalars().all()

        return [
            SemanticMemoryEntry(
                id=str(m.id),
                query=m.query,
                response=m.response,
                context_summary=m.context_summary,
                intent=m.intent,
                agent_used=m.agent_used,
                relevance_score=m.relevance_score,
                similarity=1.0,  # Exact match for conversation
                created_at=m.created_at
            )
            for m in memories
        ]

    async def get_by_intent(
        self,
        db: AsyncSession,
        intent: str,
        k: int = 10
    ) -> List[SemanticMemoryEntry]:
        """Get semantic memories filtered by intent."""
        result = await db.execute(
            select(SemanticMemory)
            .where(SemanticMemory.intent == intent)
            .order_by(SemanticMemory.created_at.desc())
            .limit(k)
        )
        memories = result.scalars().all()

        return [
            SemanticMemoryEntry(
                id=str(m.id),
                query=m.query,
                response=m.response,
                context_summary=m.context_summary,
                intent=m.intent,
                agent_used=m.agent_used,
                relevance_score=m.relevance_score,
                similarity=1.0,
                created_at=m.created_at
            )
            for m in memories
        ]

    async def delete_memory(self, db: AsyncSession, memory_id: UUID) -> bool:
        """Delete a specific semantic memory."""
        result = await db.execute(
            delete(SemanticMemory).where(SemanticMemory.id == memory_id)
        )
        return result.rowcount > 0

    async def delete_by_conversation(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> int:
        """Delete all semantic memories for a conversation."""
        result = await db.execute(
            delete(SemanticMemory)
            .where(SemanticMemory.conversation_id == conversation_id)
        )
        return result.rowcount

    async def format_for_prompt(
        self,
        db: AsyncSession,
        query: str,
        k: int = 2
    ) -> str:
        """
        Format relevant past interactions for prompt injection.

        :param db: Database session
        :param query: Current query to find similar interactions
        :param k: Number of similar interactions to include
        :return: Formatted string for prompt
        """
        memories = await self.get_relevant_memories(db, query, k=k, threshold=0.65)

        if not memories:
            return ""

        parts = ["[Relevant past interactions]"]
        for mem in memories:
            parts.append(f"Previous Q: {mem.query[:100]}...")
            parts.append(f"Previous A: {mem.response[:150]}...")
            parts.append("")

        return "\n".join(parts)

    async def get_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get statistics about semantic memories."""
        result = await db.execute(select(SemanticMemory))
        memories = result.scalars().all()

        if not memories:
            return {
                "total_count": 0,
                "by_intent": {},
                "by_agent": {},
                "avg_relevance": 0
            }

        by_intent = {}
        by_agent = {}

        for m in memories:
            if m.intent:
                by_intent[m.intent] = by_intent.get(m.intent, 0) + 1
            if m.agent_used:
                by_agent[m.agent_used] = by_agent.get(m.agent_used, 0) + 1

        return {
            "total_count": len(memories),
            "by_intent": by_intent,
            "by_agent": by_agent,
            "avg_relevance": sum(m.relevance_score for m in memories) / len(memories)
        }
