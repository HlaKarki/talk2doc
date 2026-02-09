"""Chat routes for agent-based Q&A."""
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import json

from database.session import get_db
from services import agent_executor


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=dict)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message to the AI agent.

    The agent will:
    1. Classify your intent (document query, knowledge graph, general, etc.)
    2. Route to the appropriate specialized agent
    3. Return a response with sources

    - **query**: Your question or message
    - **document_id**: Optional - restrict to a specific document
    - **conversation_id**: Optional - for conversation tracking
    """
    try:
        result = await agent_executor.execute_workflow(
            query=request.query,
            db=db,
            document_id=request.document_id,
            conversation_id=request.conversation_id
        )

        return {
            "response": result.response,
            "intent": result.intent,
            "agent_used": result.agent_used,
            "confidence": result.confidence,
            "entities": result.entities,
            "sources": result.sources,
            "conversation_id": result.conversation_id,
            "metadata": result.metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message and stream the response.

    Returns Server-Sent Events (SSE) with:
    - routing: Intent classification result
    - processing: Which agent is handling the query
    - response: Final response with sources
    """
    async def generate():
        try:
            async for event in agent_executor.execute_workflow_stream(
                query=request.query,
                db=db,
                document_id=request.document_id,
                conversation_id=request.conversation_id
            ):
                yield f"data: {json.dumps(event)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
