"""Memory inspection and management routes."""
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from services.memory.memory_manager import get_memory_manager


class MemoryResponse(BaseModel):
    """Response model for memory entry."""
    id: str
    memory_type: str
    key: str
    value: str
    confidence: float
    access_count: int


class SemanticMemoryResponse(BaseModel):
    """Response model for semantic memory."""
    id: str
    query: str
    response: str
    intent: Optional[str]
    agent_used: Optional[str]
    similarity: float


router = APIRouter(
    prefix="/memory",
    tags=["memory"],
)


@router.get("/long-term", response_model=List[MemoryResponse])
async def get_long_term_memories(
    memory_type: Optional[str] = Query(default=None, description="Filter by type: preference, fact, insight"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all long-term memories (facts, preferences, insights).
    """
    memory_manager = get_memory_manager()
    memories = await memory_manager.get_all_long_term_memories(db, memory_type)

    return [
        MemoryResponse(
            id=m["id"],
            memory_type=m["memory_type"],
            key=m["key"],
            value=m["value"],
            confidence=m["confidence"],
            access_count=m["access_count"]
        )
        for m in memories
    ]


@router.get("/search")
async def search_memories(
    query: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db)
):
    """
    Search across all memory layers.
    """
    memory_manager = get_memory_manager()
    results = await memory_manager.search_memories(db, query)

    return results


@router.get("/stats")
async def get_memory_stats(
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about all memory layers.
    """
    memory_manager = get_memory_manager()
    stats = await memory_manager.get_memory_stats(db)

    return stats


@router.post("/preference")
async def store_preference(
    key: str = Query(..., min_length=1),
    value: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db)
):
    """
    Explicitly store a user preference.
    """
    memory_manager = get_memory_manager()
    await memory_manager.store_preference(db, key, value)

    return {"message": "Preference stored", "key": key, "value": value}


@router.get("/preference/{key}")
async def get_preference(
    key: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific preference.
    """
    memory_manager = get_memory_manager()
    value = await memory_manager.get_preference(db, key)

    if value is None:
        raise HTTPException(status_code=404, detail=f"Preference '{key}' not found")

    return {"key": key, "value": value}


@router.delete("/long-term/{memory_id}")
async def delete_long_term_memory(
    memory_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a specific long-term memory.
    """
    memory_manager = get_memory_manager()
    deleted = await memory_manager.long_term.delete_memory(db, memory_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"message": "Memory deleted", "id": str(memory_id)}
