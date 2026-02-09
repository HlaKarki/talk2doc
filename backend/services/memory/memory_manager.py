"""Memory manager coordinating all memory layers."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from services.memory.short_term_memory import ShortTermMemoryService, MessageData


@dataclass
class MemoryContext:
    """Context assembled from all memory layers."""
    conversation_id: Optional[UUID]
    summary: Optional[str]
    recent_messages: List[Dict[str, Any]]
    has_history: bool

    # Placeholders for future memory layers
    long_term_memories: List[Dict[str, Any]] = None
    semantic_memories: List[Dict[str, Any]] = None
    graph_memories: List[Dict[str, Any]] = None

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
    1. Short-term: Conversation buffer (current iteration)
    2. Long-term: User preferences and important facts (future)
    3. Semantic: Past interactions searchable by similarity (future)
    4. Graph: Concept relationships across conversations (future)

    This class provides a unified interface for:
    - Retrieving context from all memory layers
    - Updating memories after interactions
    - Memory prioritization and consolidation
    """

    def __init__(self):
        self.short_term = ShortTermMemoryService()
        # Future: self.long_term = LongTermMemoryService()
        # Future: self.semantic = SemanticMemoryService()
        # Future: self.graph = GraphMemoryService()

    async def get_context_for_query(
        self,
        query: str,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None
    ) -> MemoryContext:
        """
        Assemble context from all memory layers for a query.

        :param query: The user's query (used for semantic search in future)
        :param db: Database session
        :param conversation_id: Optional conversation ID
        :return: MemoryContext with all relevant memories
        """
        # Initialize with defaults
        summary = None
        recent_messages = []
        has_history = False

        if conversation_id:
            # Get short-term memory context
            short_term_context = await self.short_term.get_context_for_query(
                db, conversation_id
            )
            summary = short_term_context.get("summary")
            recent_messages = short_term_context.get("recent_messages", [])
            has_history = short_term_context.get("has_history", False)

        # Future: Get long-term memories relevant to query
        # long_term_memories = await self.long_term.get_relevant(user_id, query)

        # Future: Get semantic memories by similarity
        # semantic_memories = await self.semantic.search(user_id, query)

        # Future: Get graph memories by traversing concept graph
        # graph_memories = await self.graph.get_related(user_id, query)

        return MemoryContext(
            conversation_id=conversation_id,
            summary=summary,
            recent_messages=recent_messages,
            has_history=has_history
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
        Update memories after an interaction.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param query: The user's query
        :param response: The assistant's response
        :param metadata: Optional metadata (sources, agent used, etc.)
        """
        # Add user message
        await self.short_term.add_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"type": "query"}
        )

        # Add assistant response
        await self.short_term.add_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=response,
            metadata=metadata or {}
        )

        # Future: Extract and store long-term memories
        # if self._should_store_long_term(query, response):
        #     await self.long_term.store(user_id, query, response)

        # Future: Store in semantic memory
        # await self.semantic.store(user_id, query, response, context)

        # Future: Update graph memory with new concepts
        # await self.graph.update(user_id, query, response)

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
        if not conversation_id:
            return ""

        # Get short-term memory formatted
        short_term_formatted = await self.short_term.format_messages_for_prompt(
            db, conversation_id
        )

        # Future: Add other memory layers
        # long_term_formatted = await self.long_term.format_for_prompt(user_id)
        # semantic_formatted = await self.semantic.format_for_prompt(user_id, query)
        # graph_formatted = await self.graph.format_for_prompt(user_id, query)

        parts = []
        if short_term_formatted:
            parts.append(short_term_formatted)

        return "\n\n".join(parts) if parts else ""

    async def start_conversation(
        self,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None,
        title: Optional[str] = None
    ) -> UUID:
        """
        Start or continue a conversation.

        :param db: Database session
        :param conversation_id: Optional existing conversation ID
        :param title: Optional title for new conversation
        :return: Conversation ID
        """
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
        """
        Clear all memory for a conversation.

        :param db: Database session
        :param conversation_id: The conversation ID
        """
        await self.short_term.clear_conversation(db, conversation_id)

    async def get_conversation_history(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get full conversation history.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param limit: Maximum messages to return
        :return: List of messages
        """
        messages = await self.short_term.get_conversation_buffer(
            db=db,
            conversation_id=conversation_id,
            window_size=limit
        )
        return [msg.to_dict() for msg in messages]


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
