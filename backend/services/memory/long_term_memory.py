"""Long-term memory service for persistent facts and preferences."""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, or_

from core.config import config
from database.models import LongTermMemory


@dataclass
class MemoryEntry:
    """Structured memory entry."""
    id: str
    memory_type: str
    key: str
    value: str
    confidence: float
    access_count: int
    created_at: datetime
    last_accessed: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


class LongTermMemoryService:
    """
    Service for managing long-term persistent memories.

    Handles:
    - User preferences (explicit settings)
    - Facts about the user (learned from conversations)
    - Insights (derived patterns and observations)
    - Memory importance scoring and retrieval
    """

    TYPE_PREFERENCE = "preference"
    TYPE_FACT = "fact"
    TYPE_INSIGHT = "insight"

    def __init__(self):
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-load LLM for memory extraction."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=config.openai_api_key
            )
        return self._llm

    async def store_preference(
        self,
        db: AsyncSession,
        key: str,
        value: str,
        conversation_id: Optional[UUID] = None,
        confidence: float = 1.0
    ) -> LongTermMemory:
        """Store a user preference. Updates if key exists."""
        return await self._upsert_memory(
            db=db,
            memory_type=self.TYPE_PREFERENCE,
            key=key,
            value=value,
            conversation_id=conversation_id,
            confidence=confidence,
            source="explicit"
        )

    async def store_fact(
        self,
        db: AsyncSession,
        key: str,
        value: str,
        conversation_id: Optional[UUID] = None,
        confidence: float = 0.8
    ) -> LongTermMemory:
        """Store a fact about the user learned from conversation."""
        return await self._upsert_memory(
            db=db,
            memory_type=self.TYPE_FACT,
            key=key,
            value=value,
            conversation_id=conversation_id,
            confidence=confidence,
            source="conversation"
        )

    async def store_insight(
        self,
        db: AsyncSession,
        key: str,
        value: str,
        conversation_id: Optional[UUID] = None,
        confidence: float = 0.7
    ) -> LongTermMemory:
        """Store an insight derived from conversations."""
        return await self._upsert_memory(
            db=db,
            memory_type=self.TYPE_INSIGHT,
            key=key,
            value=value,
            conversation_id=conversation_id,
            confidence=confidence,
            source="inferred"
        )

    async def _upsert_memory(
        self,
        db: AsyncSession,
        memory_type: str,
        key: str,
        value: str,
        conversation_id: Optional[UUID],
        confidence: float,
        source: str
    ) -> LongTermMemory:
        """Insert or update a memory."""
        result = await db.execute(
            select(LongTermMemory)
            .where(LongTermMemory.memory_type == memory_type)
            .where(LongTermMemory.key == key)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.value = value
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = datetime.utcnow()
            existing.access_count += 1
            if conversation_id:
                existing.conversation_id = conversation_id
            await db.flush()
            return existing
        else:
            memory = LongTermMemory(
                memory_type=memory_type,
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                conversation_id=conversation_id
            )
            db.add(memory)
            await db.flush()
            return memory

    async def _update_access(self, db: AsyncSession, memory: LongTermMemory) -> None:
        """Update access tracking for a memory."""
        memory.last_accessed = datetime.utcnow()
        memory.access_count += 1
        await db.flush()

    async def get_preference(self, db: AsyncSession, key: str) -> Optional[str]:
        """Get a specific preference by key."""
        result = await db.execute(
            select(LongTermMemory)
            .where(LongTermMemory.memory_type == self.TYPE_PREFERENCE)
            .where(LongTermMemory.key == key)
        )
        memory = result.scalar_one_or_none()
        if memory:
            await self._update_access(db, memory)
            return memory.value
        return None

    async def get_fact(self, db: AsyncSession, key: str) -> Optional[str]:
        """Get a specific fact by key."""
        result = await db.execute(
            select(LongTermMemory)
            .where(LongTermMemory.memory_type == self.TYPE_FACT)
            .where(LongTermMemory.key == key)
        )
        memory = result.scalar_one_or_none()
        if memory:
            await self._update_access(db, memory)
            return memory.value
        return None

    async def get_all_memories(
        self,
        db: AsyncSession,
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get all memories, optionally filtered by type."""
        query = select(LongTermMemory)
        if memory_type:
            query = query.where(LongTermMemory.memory_type == memory_type)

        result = await db.execute(query.order_by(LongTermMemory.created_at.desc()))
        memories = result.scalars().all()

        return [
            MemoryEntry(
                id=str(m.id),
                memory_type=m.memory_type,
                key=m.key,
                value=m.value,
                confidence=m.confidence,
                access_count=m.access_count,
                created_at=m.created_at,
                last_accessed=m.last_accessed
            )
            for m in memories
        ]

    async def get_important_memories(self, db: AsyncSession, k: int = 10) -> List[MemoryEntry]:
        """
        Get the most important memories based on importance scoring.
        Importance = (confidence * 0.4) + (recency * 0.3) + (access_frequency * 0.3)
        """
        result = await db.execute(
            select(LongTermMemory).order_by(LongTermMemory.last_accessed.desc())
        )
        memories = result.scalars().all()

        if not memories:
            return []

        now = datetime.utcnow()
        max_access = max(m.access_count for m in memories) or 1

        scored_memories = []
        for m in memories:
            days_old = (now - m.last_accessed).days
            recency = max(0, 1 - (days_old / 30))
            frequency = m.access_count / max_access
            importance = (m.confidence * 0.4) + (recency * 0.3) + (frequency * 0.3)
            scored_memories.append((m, importance))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored_memories[:k]

        return [
            MemoryEntry(
                id=str(m.id),
                memory_type=m.memory_type,
                key=m.key,
                value=m.value,
                confidence=m.confidence,
                access_count=m.access_count,
                created_at=m.created_at,
                last_accessed=m.last_accessed
            )
            for m, _ in top_memories
        ]

    async def search_memories(self, db: AsyncSession, query: str) -> List[MemoryEntry]:
        """Search memories by key or value containing the query."""
        search_pattern = f"%{query.lower()}%"
        result = await db.execute(
            select(LongTermMemory)
            .where(
                or_(
                    LongTermMemory.key.ilike(search_pattern),
                    LongTermMemory.value.ilike(search_pattern)
                )
            )
            .order_by(LongTermMemory.confidence.desc())
        )
        memories = result.scalars().all()

        return [
            MemoryEntry(
                id=str(m.id),
                memory_type=m.memory_type,
                key=m.key,
                value=m.value,
                confidence=m.confidence,
                access_count=m.access_count,
                created_at=m.created_at,
                last_accessed=m.last_accessed
            )
            for m in memories
        ]

    async def delete_memory(self, db: AsyncSession, memory_id: UUID) -> bool:
        """Delete a specific memory."""
        result = await db.execute(
            delete(LongTermMemory).where(LongTermMemory.id == memory_id)
        )
        return result.rowcount > 0

    async def extract_memories_from_conversation(
        self,
        db: AsyncSession,
        query: str,
        response: str,
        conversation_id: Optional[UUID] = None
    ) -> List[LongTermMemory]:
        """Use LLM to extract memorable facts from a conversation turn."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze this conversation turn and extract any important facts or preferences about the user that should be remembered for future conversations.

Return a JSON array of memories to store. Each memory should have:
- "type": "fact" (things about the user) or "preference" (user preferences) or "insight" (patterns/interests)
- "key": a short descriptive key (snake_case, e.g., "user_name", "preferred_language", "interest_area")
- "value": the actual information
- "confidence": 0.0 to 1.0 (how confident you are this is accurate)

Only extract clear, explicit information. Don't make assumptions.
If there's nothing memorable, return an empty array: []

Examples:
- User says "My name is Alex" -> [{"type": "fact", "key": "user_name", "value": "Alex", "confidence": 1.0}]
- User says "I prefer Python" -> [{"type": "preference", "key": "preferred_language", "value": "Python", "confidence": 0.9}]"""),
            ("user", "User: {query}\n\nAssistant: {response}\n\nExtract memories (JSON array):")
        ])

        chain = prompt | self.llm

        try:
            result = await chain.ainvoke({"query": query, "response": response})
            content = result.content.strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            memories_data = json.loads(content)
            if not isinstance(memories_data, list):
                return []

            created_memories = []
            for mem in memories_data:
                mem_type = mem.get("type", "fact")
                key = mem.get("key", "")
                value = mem.get("value", "")
                confidence = mem.get("confidence", 0.8)

                if not key or not value:
                    continue

                if mem_type == "preference":
                    memory = await self.store_preference(db, key, value, conversation_id, confidence)
                elif mem_type == "insight":
                    memory = await self.store_insight(db, key, value, conversation_id, confidence)
                else:
                    memory = await self.store_fact(db, key, value, conversation_id, confidence)

                created_memories.append(memory)

            return created_memories

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting memories: {e}")
            return []

    async def format_for_prompt(self, db: AsyncSession, k: int = 5) -> str:
        """Format important memories as context for prompts."""
        memories = await self.get_important_memories(db, k)

        if not memories:
            return ""

        parts = ["[Known facts about the user]"]
        for mem in memories:
            parts.append(f"- {mem.key}: {mem.value}")

        return "\n".join(parts)
