"""Knowledge Graph routes for entity extraction and graph querying."""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from services import knowledge_graph_service


class SearchRequest(BaseModel):
    """Request body for entity search."""
    query: str
    k: int = 10
    entity_type: Optional[str] = None


class SubgraphRequest(BaseModel):
    """Request body for subgraph extraction."""
    node_ids: List[UUID]
    include_connections: bool = True


router = APIRouter(
    prefix="/kg",
    tags=["knowledge-graph"],
)


@router.post("/extract/{document_id}", response_model=dict)
async def extract_knowledge_graph(
    document_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract knowledge graph from a document.

    This processes all chunks and extracts:
    - Entities (nodes): People, Organizations, Concepts, etc.
    - Relationships (edges): How entities relate to each other

    Note: This will clear any existing graph for the document.
    """
    try:
        result = await knowledge_graph_service.build_graph_from_document(document_id, db)
        return {
            "success": True,
            "document_id": str(document_id),
            "nodes_created": result["nodes_created"],
            "edges_created": result["edges_created"]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/documents/{document_id}/graph", response_model=dict)
async def get_document_graph(
    document_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the knowledge graph for a document.

    Returns nodes and edges in a format suitable for visualization.
    """
    graph = await knowledge_graph_service.get_document_graph(document_id, db)

    if not graph["nodes"] and not graph["edges"]:
        raise HTTPException(
            status_code=404,
            detail="No graph found for this document. Run extraction first."
        )

    return {
        "document_id": str(document_id),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "nodes": graph["nodes"],
        "edges": graph["edges"]
    }


@router.get("/nodes/{node_id}", response_model=dict)
async def get_node(
    node_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific node."""
    node = await knowledge_graph_service.get_node_by_id(node_id, db)

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    return node


@router.get("/edges/{edge_id}", response_model=dict)
async def get_edge(
    edge_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific edge."""
    edge = await knowledge_graph_service.get_edge_by_id(edge_id, db)

    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")

    return edge


@router.post("/search", response_model=List[dict])
async def search_entities(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search entities across all documents using semantic similarity.

    - **query**: Search query text
    - **k**: Number of results to return (default: 10)
    - **entity_type**: Optional filter by entity type (Person, Organization, etc.)
    """
    results = await knowledge_graph_service.search_entities(
        query=request.query,
        db=db,
        k=request.k,
        entity_type=request.entity_type
    )

    return results


@router.get("/nodes/{node_id}/neighbors", response_model=dict)
async def get_node_neighbors(
    node_id: UUID,
    depth: int = 1,
    direction: str = "both",
    db: AsyncSession = Depends(get_db)
):
    """
    Get neighboring nodes connected to a specific node.

    - **node_id**: The node to find neighbors for
    - **depth**: How many hops to traverse (1 = immediate neighbors, default: 1)
    - **direction**: "in" (incoming edges), "out" (outgoing), or "both" (default)
    """
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=400, detail="Depth must be between 1 and 5")

    if direction not in ("in", "out", "both"):
        raise HTTPException(status_code=400, detail="Direction must be 'in', 'out', or 'both'")

    result = await knowledge_graph_service.get_node_neighbors(
        node_id=node_id,
        db=db,
        depth=depth,
        direction=direction
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Node not found")

    return result


@router.get("/path", response_model=dict)
async def find_path(
    from_node: UUID,
    to_node: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Find the shortest path between two nodes.

    - **from_node**: Starting node ID
    - **to_node**: Target node ID

    Returns the path with all nodes and edges traversed.
    """
    result = await knowledge_graph_service.find_shortest_path(
        from_node_id=from_node,
        to_node_id=to_node,
        db=db
    )

    if result is None:
        raise HTTPException(status_code=404, detail="One or both nodes not found")

    return result


@router.post("/subgraph", response_model=dict)
async def get_subgraph(
    request: SubgraphRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract a subgraph containing specific nodes.

    - **node_ids**: List of node IDs to include in the subgraph
    - **include_connections**: If true, include edges between the specified nodes (default: true)

    Useful for visualizing relationships between a set of entities.
    """
    if not request.node_ids:
        raise HTTPException(status_code=400, detail="At least one node_id is required")

    if len(request.node_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 nodes allowed")

    result = await knowledge_graph_service.get_subgraph(
        node_ids=request.node_ids,
        db=db,
        include_connections=request.include_connections
    )

    return result
