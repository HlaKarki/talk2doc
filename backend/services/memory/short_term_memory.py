"""Short-term memory service for managing conversation buffers."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from core.config import config
from database.models import Conversation, Message, ConversationSummary


@dataclass
class MessageData:
    """Structured message data for memory operations."""
    role: str
    content: str
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ShortTermMemoryService:
    """
    Service for managing short-term conversation memory.

    Handles:
    - Conversation buffer (recent messages)
    - Rolling window management
    - Context summarization for long conversations
    - Message pruning
    """

    DEFAULT_WINDOW_SIZE = 10  # Number of recent messages to keep in active context
    MAX_MESSAGES_BEFORE_SUMMARY = 20  # Summarize when exceeding this count

    def __init__(self, window_size: int = None):
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-load LLM for summarization."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=config.openai_api_key
            )
        return self._llm

    async def get_or_create_conversation(
        self,
        db: AsyncSession,
        conversation_id: Optional[UUID] = None,
        title: Optional[str] = None
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if conversation_id:
            result = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation

        # Create new conversation
        conversation = Conversation(
            title=title or "New Conversation",
            conv_metadata={}
        )
        db.add(conversation)
        await db.flush()
        return conversation

    async def add_message(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation buffer.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param role: Message role ("user", "assistant", "system")
        :param content: Message content
        :param metadata: Optional metadata (sources, agent used, etc.)
        :return: Created message
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            msg_metadata=metadata or {}
        )
        db.add(message)
        await db.flush()

        # Update conversation timestamp
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.updated_at = datetime.utcnow()

        # Check if we need to summarize
        message_count = await self._get_message_count(db, conversation_id)
        if message_count > self.MAX_MESSAGES_BEFORE_SUMMARY:
            await self._update_summary(db, conversation_id)

        return message

    async def get_conversation_buffer(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        window_size: Optional[int] = None
    ) -> List[MessageData]:
        """
        Get recent messages for a conversation.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param window_size: Number of recent messages to return (default: self.window_size)
        :return: List of recent messages
        """
        limit = window_size or self.window_size

        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()

        # Reverse to get chronological order
        return [
            MessageData(
                role=msg.role,
                content=msg.content,
                metadata=msg.msg_metadata,
                created_at=msg.created_at
            )
            for msg in reversed(messages)
        ]

    async def get_context_for_query(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> Dict[str, Any]:
        """
        Get full context for a query, including summary and recent messages.

        :param db: Database session
        :param conversation_id: The conversation ID
        :return: Context dict with summary and recent messages
        """
        # Get summary if exists
        summary = await self._get_summary(db, conversation_id)

        # Get recent messages
        recent_messages = await self.get_conversation_buffer(db, conversation_id)

        return {
            "summary": summary,
            "recent_messages": [msg.to_dict() for msg in recent_messages],
            "has_history": bool(summary or recent_messages)
        }

    async def format_messages_for_prompt(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> str:
        """
        Format conversation history as a string for prompt injection.

        :param db: Database session
        :param conversation_id: The conversation ID
        :return: Formatted conversation history
        """
        context = await self.get_context_for_query(db, conversation_id)

        parts = []

        # Add summary if exists
        if context["summary"]:
            parts.append(f"[Previous conversation summary]\n{context['summary']}\n")

        # Add recent messages
        if context["recent_messages"]:
            parts.append("[Recent messages]")
            for msg in context["recent_messages"]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role_label}: {msg['content']}")

        return "\n".join(parts) if parts else ""

    async def summarize_context(
        self,
        messages: List[MessageData]
    ) -> str:
        """
        Summarize a list of messages into a concise context.

        :param messages: Messages to summarize
        :return: Summarized context
        """
        if not messages:
            return ""

        # Format messages for summarization
        formatted = "\n".join([
            f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
            for msg in messages
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a conversation summarizer. Create a concise summary of the conversation that captures:
1. Key topics discussed
2. Important questions asked
3. Key information or answers provided
4. Any decisions or conclusions reached

Keep the summary brief but informative. Focus on what would be useful context for continuing the conversation."""),
            ("user", "Please summarize this conversation:\n\n{conversation}")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({"conversation": formatted})

        return response.content

    async def prune_messages(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        keep_count: Optional[int] = None
    ) -> int:
        """
        Remove old messages from a conversation, keeping only the most recent.

        :param db: Database session
        :param conversation_id: The conversation ID
        :param keep_count: Number of messages to keep (default: window_size * 2)
        :return: Number of messages deleted
        """
        keep = keep_count or (self.window_size * 2)

        # Get IDs of messages to keep
        result = await db.execute(
            select(Message.id)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(keep)
        )
        keep_ids = [row[0] for row in result.fetchall()]

        if not keep_ids:
            return 0

        # Delete messages not in keep list
        delete_result = await db.execute(
            delete(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.id.not_in(keep_ids))
        )

        return delete_result.rowcount

    async def clear_conversation(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> None:
        """
        Clear all messages and summary for a conversation.

        :param db: Database session
        :param conversation_id: The conversation ID
        """
        # Delete summary
        await db.execute(
            delete(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
        )

        # Delete messages
        await db.execute(
            delete(Message)
            .where(Message.conversation_id == conversation_id)
        )

    async def _get_message_count(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> int:
        """Get the number of messages in a conversation."""
        result = await db.execute(
            select(func.count(Message.id))
            .where(Message.conversation_id == conversation_id)
        )
        return result.scalar() or 0

    async def _get_summary(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> Optional[str]:
        """Get the current summary for a conversation."""
        result = await db.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
        )
        summary = result.scalar_one_or_none()
        return summary.summary if summary else None

    async def _update_summary(
        self,
        db: AsyncSession,
        conversation_id: UUID
    ) -> None:
        """
        Update the conversation summary by summarizing older messages.
        Keeps the most recent messages in the buffer and summarizes the rest.
        """
        # Get all messages except the most recent window
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        all_messages = result.scalars().all()

        if len(all_messages) <= self.window_size:
            return

        # Messages to summarize (older ones)
        messages_to_summarize = all_messages[:-self.window_size]

        # Get existing summary
        existing_summary = await self._get_summary(db, conversation_id)

        # Build content to summarize
        messages_data = [
            MessageData(
                role=msg.role,
                content=msg.content,
                metadata=msg.msg_metadata,
                created_at=msg.created_at
            )
            for msg in messages_to_summarize
        ]

        # If there's an existing summary, include it
        if existing_summary:
            summary_prefix = f"Previous summary: {existing_summary}\n\nNew messages to incorporate:\n"
            formatted_messages = "\n".join([
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in messages_data
            ])

            prompt = ChatPromptTemplate.from_messages([
                ("system", """Update the conversation summary by incorporating the new messages.
Keep the summary concise but comprehensive."""),
                ("user", summary_prefix + formatted_messages)
            ])

            chain = prompt | self.llm
            response = await chain.ainvoke({})
            new_summary = response.content
        else:
            new_summary = await self.summarize_context(messages_data)

        # Upsert summary
        result = await db.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.summary = new_summary
            existing.messages_summarized = len(messages_to_summarize)
            existing.last_message_id = messages_to_summarize[-1].id
            existing.updated_at = datetime.utcnow()
        else:
            summary_record = ConversationSummary(
                conversation_id=conversation_id,
                summary=new_summary,
                messages_summarized=len(messages_to_summarize),
                last_message_id=messages_to_summarize[-1].id
            )
            db.add(summary_record)

        # Delete summarized messages to save space
        message_ids_to_delete = [msg.id for msg in messages_to_summarize]
        await db.execute(
            delete(Message)
            .where(Message.id.in_(message_ids_to_delete))
        )

        await db.flush()
