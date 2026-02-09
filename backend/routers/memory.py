"""Memory inspection and management routes."""
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
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


# ============================================================================
# Graph Memory Endpoints
# ============================================================================

@router.get("/graph")
async def get_memory_graph(
    limit: int = Query(default=50, le=200, description="Maximum nodes to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the full memory graph (nodes and edges).

    Useful for visualization.
    """
    memory_manager = get_memory_manager()
    nodes = await memory_manager.graph.get_all_nodes(db, limit=limit)
    edges = await memory_manager.graph.get_all_edges(db, limit=limit * 4)

    return {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges]
    }


@router.get("/graph/concept/{concept}")
async def get_concept_graph(
    concept: str,
    depth: int = Query(default=2, ge=1, le=4, description="Traversal depth"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a concept and its related concepts via graph traversal.
    """
    memory_manager = get_memory_manager()
    result = await memory_manager.graph.traverse_memory_graph(
        db=db,
        start_concept=concept,
        depth=depth
    )

    if not result:
        raise HTTPException(status_code=404, detail=f"Concept '{concept}' not found")

    return result.to_dict()


@router.get("/graph/search")
async def search_graph_concepts(
    query: str = Query(..., min_length=1),
    k: int = Query(default=10, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for concepts in the memory graph using semantic similarity.
    """
    memory_manager = get_memory_manager()
    concepts = await memory_manager.graph.search_concepts(
        db=db,
        query=query,
        k=k,
        threshold=0.4
    )

    return {
        "query": query,
        "concepts": [c.to_dict() for c in concepts]
    }


class LearnConceptRequest(BaseModel):
    """Request body for learning a concept."""
    concept: str
    concept_type: str = "topic"
    related_to: Optional[List[str]] = None


@router.post("/learn")
async def learn_concept(
    request: LearnConceptRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Explicitly teach the system a concept and its relationships.
    """
    memory_manager = get_memory_manager()

    # Add the main concept
    node = await memory_manager.graph.add_memory_node(
        db=db,
        concept=request.concept,
        concept_type=request.concept_type
    )

    # Add relationships if provided
    edges_created = 0
    if request.related_to:
        for related_concept in request.related_to:
            edge = await memory_manager.graph.add_memory_edge(
                db=db,
                source_concept=request.concept,
                target_concept=related_concept,
                relationship_type="related_to"
            )
            if edge:
                edges_created += 1

    return {
        "message": "Concept learned",
        "concept": request.concept,
        "concept_type": request.concept_type,
        "edges_created": edges_created
    }


@router.delete("/graph/concept/{concept}")
async def forget_concept(
    concept: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a concept from the memory graph.

    This also removes all edges connected to this concept.
    """
    memory_manager = get_memory_manager()
    deleted = await memory_manager.graph.delete_node(db, concept)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Concept '{concept}' not found")

    return {"message": "Concept forgotten", "concept": concept}
