"""Memory manager coordinating all memory layers."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from services.memory.short_term_memory import ShortTermMemoryService, MessageData
from services.memory.long_term_memory import LongTermMemoryService
from services.memory.semantic_memory import SemanticMemoryService


@dataclass
class MemoryContext:
    """Context assembled from all memory layers."""
    conversation_id: Optional[UUID]
    # Short-term
    summary: Optional[str]
    recent_messages: List[Dict[str, Any]]
    has_history: bool
    # Long-term
    long_term_memories: List[Dict[str, Any]]
    # Semantic
    semantic_memories: List[Dict[str, Any]]
    # Future: Graph memory
    graph_memories: List[Dict[str, Any]]

    def __post_init__(self):
        if self.long_term_memories is None:
            self.long_term_memories = []
        if self.semantic_memories is None:
            self.semantic_memories = []
        if self.graph_memories is None:
            self.graph_memories = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": str(self.conversation_id) if self.conversation_id else None,
            "summary": self.summary,
            "recent_messages": self.recent_messages,
            "has_history": self.has_history,
            "long_term_memories": self.long_term_memories,
            "semantic_memories": self.semantic_memories,
            "graph_memories": self.graph_memories
        }


class MemoryManager:
    """
    Central manager coordinating all memory layers.

    Memory Layers:
    1. Short-term: Conversation buffer (recent messages)
    2. Long-term: User preferences and learned facts
    3. Semantic: Past interactions searchable by similarity
    4. Graph: Concept relationships (future iteration)

    This class provides a unified interface for:
    - Retrieving context from all memory layers
    - Updating memories after interactions
    - Memory prioritization and consolidation
    """

    def __init__(self):
        self.short_term = ShortTermMemoryService()
        self.long_term = LongTermMemoryService()
        self.semantic = SemanticMemoryService()
        # Future: self.graph = GraphMemoryService()

    async def get_context_for_query(
        self,
        query: str,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None
    ) -> MemoryContext:
        """
        Assemble context from all memory layers for a query.

        :param query: The user's query (used for semantic search)
        :param db: Database session
        :param conversation_id: Optional conversation ID
        :return: MemoryContext with all relevant memories
        """
        # Short-term memory
        summary = None
        recent_messages = []
        has_history = False

        if conversation_id:
            short_term_context = await self.short_term.get_context_for_query(
                db, conversation_id
            )
            summary = short_term_context.get("summary")
            recent_messages = short_term_context.get("recent_messages", [])
            has_history = short_term_context.get("has_history", False)

        # Long-term memory - get important facts about the user
        long_term_entries = await self.long_term.get_important_memories(db, k=5)
        long_term_memories = [m.to_dict() for m in long_term_entries]

        # Semantic memory - find similar past interactions
        semantic_entries = await self.semantic.get_relevant_memories(
            db=db,
            query=query,
            k=2,
            threshold=0.6
        )
        semantic_memories = [m.to_dict() for m in semantic_entries]

        return MemoryContext(
            conversation_id=conversation_id,
            summary=summary,
            recent_messages=recent_messages,
            has_history=has_history or bool(long_term_memories) or bool(semantic_memories),
            long_term_memories=long_term_memories,
            semantic_memories=semantic_memories,
            graph_memories=[]  # Future
        )

    async def update_memory(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update all memory layers after an interaction.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param query: The user's query
        :param response: The assistant's response
        :param metadata: Optional metadata (sources, agent used, etc.)
        """
        metadata = metadata or {}

        # 1. Update short-term memory (conversation buffer)
        await self.short_term.add_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"type": "query"}
        )

        await self.short_term.add_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=response,
            metadata=metadata
        )

        # 2. Extract and store long-term memories (facts, preferences)
        await self.long_term.extract_memories_from_conversation(
            db=db,
            query=query,
            response=response,
            conversation_id=conversation_id
        )

        # 3. Store in semantic memory for future similarity search
        await self.semantic.store_interaction(
            db=db,
            query=query,
            response=response,
            conversation_id=conversation_id,
            intent=metadata.get("intent"),
            agent_used=metadata.get("agent_used")
        )

    async def format_for_prompt(
        self,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Format all memory context as a string for prompt injection.

        :param db: Database session
        :param conversation_id: Optional conversation ID
        :param query: Optional query for semantic search
        :return: Formatted memory context
        """
        parts = []

        # Long-term memories (always include if available)
        long_term_formatted = await self.long_term.format_for_prompt(db, k=5)
        if long_term_formatted:
            parts.append(long_term_formatted)

        # Semantic memories (if query provided)
        if query:
            semantic_formatted = await self.semantic.format_for_prompt(db, query, k=2)
            if semantic_formatted:
                parts.append(semantic_formatted)

        # Short-term memory (conversation history)
        if conversation_id:
            short_term_formatted = await self.short_term.format_messages_for_prompt(
                db, conversation_id
            )
            if short_term_formatted:
                parts.append(short_term_formatted)

        return "\n\n".join(parts) if parts else ""

    async def start_conversation(
        self,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None,
        title: Optional[str] = None
    ) -> UUID:
        """Start or continue a conversation."""
        conversation = await self.short_term.get_or_create_conversation(
            db=db,
            conversation_id=conversation_id,
            title=title
        )
        return conversation.id

    async def clear_conversation(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> None:
        """Clear all memory for a conversation."""
        await self.short_term.clear_conversation(db, conversation_id)
        await self.semantic.delete_by_conversation(db, conversation_id)

    async def get_conversation_history(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get full conversation history."""
        messages = await self.short_term.get_conversation_buffer(
            db=db,
            conversation_id=conversation_id,
            window_size=limit
        )
        return [msg.to_dict() for msg in messages]

    async def get_all_long_term_memories(
        self,
        db: AsyncSession,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all long-term memories."""
        memories = await self.long_term.get_all_memories(db, memory_type)
        return [m.to_dict() for m in memories]

    async def store_preference(
        self,
        db: AsyncSession,
        key: str,
        value: str,
        conversation_id: Optional[UUID] = None
    ) -> None:
        """Explicitly store a user preference."""
        await self.long_term.store_preference(
            db=db,
            key=key,
            value=value,
            conversation_id=conversation_id
        )

    async def get_preference(
        self,
        db: AsyncSession,
        key: str
    ) -> Optional[str]:
        """Get a specific user preference."""
        return await self.long_term.get_preference(db, key)

    async def search_memories(
        self,
        db: AsyncSession,
        query: str,
        include_long_term: bool = True,
        include_semantic: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across memory layers.

        :param db: Database session
        :param query: Search query
        :param include_long_term: Include long-term memory search
        :param include_semantic: Include semantic memory search
        :return: Dict with results from each memory layer
        """
        results = {}

        if include_long_term:
            lt_results = await self.long_term.search_memories(db, query)
            results["long_term"] = [m.to_dict() for m in lt_results]

        if include_semantic:
            sem_results = await self.semantic.search_similar_interactions(
                db, query, k=5, threshold=0.5
            )
            results["semantic"] = [m.to_dict() for m in sem_results]

        return results

    async def get_memory_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get statistics about all memory layers."""
        lt_memories = await self.long_term.get_all_memories(db)
        sem_stats = await self.semantic.get_stats(db)

        return {
            "long_term": {
                "total_count": len(lt_memories),
                "by_type": {
                    "preference": len([m for m in lt_memories if m.memory_type == "preference"]),
                    "fact": len([m for m in lt_memories if m.memory_type == "fact"]),
                    "insight": len([m for m in lt_memories if m.memory_type == "insight"]),
                }
            },
            "semantic": sem_stats
        }


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
