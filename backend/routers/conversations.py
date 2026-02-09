"""Conversation management routes."""
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from database.session import get_db
from database.models import Conversation, Message
from services.memory.memory_manager import get_memory_manager


class ConversationResponse(BaseModel):
    """Response model for conversation."""
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageResponse(BaseModel):
    """Response model for message."""
    id: str
    role: str
    content: str
    created_at: str
    metadata: dict = {}


class ConversationDetailResponse(BaseModel):
    """Response model for conversation with messages."""
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: List[MessageResponse]
    summary: Optional[str] = None


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all conversations, ordered by most recent.
    """
    result = await db.execute(
        select(Conversation)
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(limit)
    )
    conversations = result.scalars().all()

    response = []
    for conv in conversations:
        # Get message count
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
        )
        message_count = len(msg_result.scalars().all())

        response.append(ConversationResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            message_count=message_count
        ))

    return response


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a conversation with all its messages.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    # Get summary if exists
    memory_manager = get_memory_manager()
    context = await memory_manager.get_context_for_query(
        query="",
        db=db,
        conversation_id=conversation_id
    )

    return ConversationDetailResponse(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        messages=[
            MessageResponse(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at.isoformat(),
                metadata=msg.msg_metadata or {}
            )
            for msg in messages
        ],
        summary=context.summary
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a conversation and all its messages.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)

    return {"message": "Conversation deleted", "id": str(conversation_id)}


@router.post("/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear all messages from a conversation but keep the conversation.
    """
    memory_manager = get_memory_manager()

    # Check conversation exists
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await memory_manager.clear_conversation(db, conversation_id)

    return {"message": "Conversation cleared", "id": str(conversation_id)}


@router.patch("/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: UUID,
    title: str = Query(..., min_length=1, max_length=255),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the title of a conversation.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.title = title

    return {"message": "Title updated", "id": str(conversation_id), "title": title}
