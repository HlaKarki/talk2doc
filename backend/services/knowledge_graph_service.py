"""Knowledge Graph service for entity extraction and graph management."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document as LCDocument
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from core.config import config
from database.models import Document, DocumentChunk, KnowledgeGraphNode, KnowledgeGraphEdge
from services.embedding_service import generate_embeddings


# Entity types we want to extract
ALLOWED_NODES = ["Person", "Organization", "Location", "Concept", "Technology", "Event", "Date"]
ALLOWED_RELATIONSHIPS = [
    "works_at", "located_in", "related_to", "part_of", "created_by",
    "uses", "depends_on", "occurred_on", "affiliated_with", "mentions"
]


@dataclass
class ExtractedNode:
    """Extracted entity from text."""
    name: str
    type: str
    properties: Dict[str, Any]


@dataclass
class ExtractedEdge:
    """Extracted relationship from text."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]


@dataclass
class ExtractionResult:
    """Result of knowledge graph extraction."""
    nodes: List[ExtractedNode]
    edges: List[ExtractedEdge]


def get_graph_transformer() -> LLMGraphTransformer:
    """Create and configure the LLMGraphTransformer."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=config.openai_api_key
    )

    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=["description"],
        relationship_properties=["description"],
    )


async def extract_from_text(text: str) -> ExtractionResult:
    """
    Extract entities and relationships from text using LLMGraphTransformer.
    """
    transformer = get_graph_transformer()

    # LLMGraphTransformer expects LangChain Document objects
    lc_doc = LCDocument(page_content=text)

    # Extract graph - this is sync, so we run it directly
    # (LLMGraphTransformer doesn't have async support yet)
    graph_documents = transformer.convert_to_graph_documents([lc_doc])

    nodes = []
    edges = []

    for graph_doc in graph_documents:
        # Extract nodes
        for node in graph_doc.nodes:
            nodes.append(ExtractedNode(
                name=node.id,
                type=node.type,
                properties=node.properties if hasattr(node, 'properties') else {}
            ))

        # Extract relationships
        for rel in graph_doc.relationships:
            edges.append(ExtractedEdge(
                source=rel.source.id,
                target=rel.target.id,
                relationship=rel.type,
                properties=rel.properties if hasattr(rel, 'properties') else {}
            ))

    return ExtractionResult(nodes=nodes, edges=edges)


async def build_graph_from_document(
    document_id: UUID,
    db: AsyncSession
) -> Dict[str, int]:
    """
    Extract knowledge graph from all chunks of a document.
    Returns count of nodes and edges created.
    """
    # Get document
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()
    if not document:
        raise ValueError(f"Document {document_id} not found")

    # Get all chunks
    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for document {document_id}")

    # Clear existing graph for this document (re-extraction)
    await db.execute(
        delete(KnowledgeGraphEdge).where(KnowledgeGraphEdge.source_document_id == document_id)
    )
    await db.execute(
        delete(KnowledgeGraphNode).where(KnowledgeGraphNode.source_document_id == document_id)
    )

    all_nodes: Dict[str, KnowledgeGraphNode] = {}  # name -> node (for deduplication)
    all_edges: List[tuple] = []  # (source_name, target_name, rel_type, props, chunk_id)

    # Extract from each chunk
    for chunk in chunks:
        try:
            result = await extract_from_text(chunk.content)

            # Collect nodes (deduplicate by name)
            for node in result.nodes:
                node_key = f"{node.name.lower()}:{node.type.lower()}"
                if node_key not in all_nodes:
                    # Generate embedding for the entity name
                    embeddings = await generate_embeddings([node.name])
                    embedding = embeddings[0] if embeddings else None

                    db_node = KnowledgeGraphNode(
                        entity_name=node.name,
                        entity_type=node.type,
                        properties=node.properties,
                        embedding=embedding,
                        source_document_id=document_id,
                        source_chunk_id=chunk.id
                    )
                    db.add(db_node)
                    all_nodes[node_key] = db_node

            # Collect edges
            for edge in result.edges:
                all_edges.append((
                    edge.source,
                    edge.target,
                    edge.relationship,
                    edge.properties,
                    chunk.id
                ))

        except Exception as e:
            # Log but continue with other chunks
            print(f"Error extracting from chunk {chunk.id}: {e}")
            continue

    # Flush to get node IDs
    await db.flush()

    # Create edges
    edges_created = 0
    for source_name, target_name, rel_type, props, chunk_id in all_edges:
        # Find source and target nodes
        source_key = None
        target_key = None

        for key, node in all_nodes.items():
            if key.startswith(source_name.lower() + ":"):
                source_key = key
            if key.startswith(target_name.lower() + ":"):
                target_key = key

        if source_key and target_key:
            source_node = all_nodes[source_key]
            target_node = all_nodes[target_key]

            db_edge = KnowledgeGraphEdge(
                source_node_id=source_node.id,
                target_node_id=target_node.id,
                relationship_type=rel_type,
                properties=props,
                source_document_id=document_id
            )
            db.add(db_edge)
            edges_created += 1

    await db.flush()

    return {
        "nodes_created": len(all_nodes),
        "edges_created": edges_created
    }


async def get_document_graph(
    document_id: UUID,
    db: AsyncSession
) -> Dict[str, Any]:
    """
    Get the knowledge graph for a document.
    Returns nodes and edges in a format suitable for visualization.
    """
    # Get nodes
    nodes_result = await db.execute(
        select(KnowledgeGraphNode).where(KnowledgeGraphNode.source_document_id == document_id)
    )
    nodes = nodes_result.scalars().all()

    # Get edges
    edges_result = await db.execute(
        select(KnowledgeGraphEdge).where(KnowledgeGraphEdge.source_document_id == document_id)
    )
    edges = edges_result.scalars().all()

    return {
        "nodes": [
            {
                "id": str(node.id),
                "name": node.entity_name,
                "type": node.entity_type,
                "properties": node.properties
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": str(edge.id),
                "source": str(edge.source_node_id),
                "target": str(edge.target_node_id),
                "relationship": edge.relationship_type,
                "properties": edge.properties,
                "confidence": edge.confidence
            }
            for edge in edges
        ]
    }


async def get_node_by_id(
    node_id: UUID,
    db: AsyncSession
) -> Optional[Dict[str, Any]]:
    """Get a specific node by ID."""
    result = await db.execute(
        select(KnowledgeGraphNode).where(KnowledgeGraphNode.id == node_id)
    )
    node = result.scalar_one_or_none()

    if not node:
        return None

    return {
        "id": str(node.id),
        "name": node.entity_name,
        "type": node.entity_type,
        "properties": node.properties,
        "document_id": str(node.source_document_id),
        "created_at": node.created_at.isoformat()
    }


async def get_edge_by_id(
    edge_id: UUID,
    db: AsyncSession
) -> Optional[Dict[str, Any]]:
    """Get a specific edge by ID."""
    result = await db.execute(
        select(KnowledgeGraphEdge).where(KnowledgeGraphEdge.id == edge_id)
    )
    edge = result.scalar_one_or_none()

    if not edge:
        return None

    return {
        "id": str(edge.id),
        "source_node_id": str(edge.source_node_id),
        "target_node_id": str(edge.target_node_id),
        "relationship": edge.relationship_type,
        "properties": edge.properties,
        "confidence": edge.confidence,
        "document_id": str(edge.source_document_id),
        "created_at": edge.created_at.isoformat()
    }


async def search_entities(
    query: str,
    db: AsyncSession,
    k: int = 10,
    entity_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search entities by semantic similarity.
    """
    from sqlalchemy import text

    # Generate query embedding
    embeddings = await generate_embeddings([query])
    if not embeddings:
        return []

    query_embedding = embeddings[0]
    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    # Build query with optional type filter
    sql = """
        SELECT id, entity_name, entity_type, properties, source_document_id,
               embedding <=> :embedding AS distance
        FROM kg_nodes
        WHERE embedding IS NOT NULL
    """

    if entity_type:
        sql += " AND entity_type = :entity_type"

    sql += " ORDER BY distance LIMIT :limit"

    params = {"embedding": embedding_str, "limit": k}
    if entity_type:
        params["entity_type"] = entity_type

    result = await db.execute(text(sql), params)
    rows = result.fetchall()

    return [
        {
            "id": str(row.id),
            "name": row.entity_name,
            "type": row.entity_type,
            "properties": row.properties,
            "document_id": str(row.source_document_id),
            "score": 1 - row.distance  # Convert distance to similarity
        }
        for row in rows
    ]
