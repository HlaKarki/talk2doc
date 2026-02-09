"""Graph memory service for storing and traversing concept relationships."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, text
from sqlalchemy.orm import selectinload

from database.models import MemoryGraphNode, MemoryGraphEdge
from services.embedding_service import generate_embeddings
from core.config import config


@dataclass
class GraphNode:
    """Structured graph node for API responses."""
    id: str
    concept: str
    concept_type: str
    properties: Dict[str, Any]
    mention_count: int
    first_mentioned: datetime
    last_mentioned: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "concept": self.concept,
            "concept_type": self.concept_type,
            "properties": self.properties,
            "mention_count": self.mention_count,
            "first_mentioned": self.first_mentioned.isoformat() if self.first_mentioned else None,
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None,
        }


@dataclass
class GraphEdge:
    """Structured graph edge for API responses."""
    id: str
    source_concept: str
    target_concept: str
    relationship_type: str
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_concept": self.source_concept,
            "target_concept": self.target_concept,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
        }


@dataclass
class GraphMemoryResult:
    """Result from graph memory traversal."""
    node: GraphNode
    related_nodes: List[GraphNode]
    edges: List[GraphEdge]
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "related_nodes": [n.to_dict() for n in self.related_nodes],
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
        }


class GraphMemoryService:
    """
    Service for graph-based memory storing concept relationships.

    Enables:
    - Storing concepts from conversations
    - Connecting related concepts
    - Traversing the memory graph
    - Finding related memories through graph structure
    """

    # Common relationship types
    RELATIONSHIP_TYPES = [
        "related_to",  # General relationship
        "used_for",  # Tool/skill used for a purpose
        "part_of",  # Component relationship
        "interested_in",  # User interest
        "works_with",  # Technologies that work together
        "similar_to",  # Conceptual similarity
        "requires",  # Dependency relationship
        "produces",  # Output relationship
    ]

    # Concept types
    CONCEPT_TYPES = [
        "topic",  # General topic (e.g., "machine learning")
        "skill",  # User skill (e.g., "Python programming")
        "tool",  # Tool or technology (e.g., "TensorFlow")
        "project",  # User's project
        "person",  # Person mentioned
        "organization",  # Company or org
        "preference",  # User preference
        "goal",  # User goal or objective
    ]

    async def add_memory_node(
            self,
            db: AsyncSession,
            concept: str,
            concept_type: str = "topic",
            properties: Optional[Dict[str, Any]] = None,
            conversation_id: Optional[UUID] = None,
            generate_embedding: bool = True
    ) -> MemoryGraphNode:
        """
        Add a concept node to the memory graph.

        If the concept already exists, updates mention count and last_mentioned.

        :param db: Database session
        :param concept: The concept name
        :param concept_type: Type of concept (topic, skill, tool, etc.)
        :param properties: Additional properties
        :param conversation_id: Source conversation
        :param generate_embedding: Whether to generate embedding for the concept
        :return: Created or updated node
        """
        # Normalize concept name
        concept_normalized = concept.lower().strip()

        # Check if concept already exists
        result = await db.execute(
            select(MemoryGraphNode).where(
                MemoryGraphNode.concept == concept_normalized
            )
        )
        existing_node = result.scalar_one_or_none()

        if existing_node:
            # Update existing node
            existing_node.mention_count += 1
            existing_node.last_mentioned = datetime.utcnow()
            if properties:
                existing_node.properties = {**existing_node.properties, **properties}
            await db.flush()
            return existing_node

        # Generate embedding for new concept
        embedding = None
        if generate_embedding:
            embeddings = await generate_embeddings([concept])
            embedding = embeddings[0] if embeddings else None

        # Create new node
        node = MemoryGraphNode(
            concept=concept_normalized,
            concept_type=concept_type,
            properties=properties or {},
            embedding=embedding,
            source_conversation_id=conversation_id
        )

        db.add(node)
        await db.flush()
        return node

    async def add_memory_edge(
            self,
            db: AsyncSession,
            source_concept: str,
            target_concept: str,
            relationship_type: str = "related_to",
            properties: Optional[Dict[str, Any]] = None,
            weight: float = 1.0,
            conversation_id: Optional[UUID] = None
    ) -> Optional[MemoryGraphEdge]:
        """
        Add a relationship between two concepts.

        Creates nodes if they don't exist.

        :param db: Database session
        :param source_concept: Source concept name
        :param target_concept: Target concept name
        :param relationship_type: Type of relationship
        :param properties: Additional properties
        :param weight: Relationship strength
        :param conversation_id: Source conversation
        :return: Created edge or None if same concept
        """
        if source_concept.lower().strip() == target_concept.lower().strip():
            return None  # No self-loops

        # Ensure both nodes exist
        source_node = await self.add_memory_node(
            db, source_concept, conversation_id=conversation_id
        )
        target_node = await self.add_memory_node(
            db, target_concept, conversation_id=conversation_id
        )

        # Check if edge already exists
        result = await db.execute(
            select(MemoryGraphEdge).where(
                and_(
                    MemoryGraphEdge.source_node_id == source_node.id,
                    MemoryGraphEdge.target_node_id == target_node.id,
                    MemoryGraphEdge.relationship_type == relationship_type
                )
            )
        )
        existing_edge = result.scalar_one_or_none()

        if existing_edge:
            # Strengthen existing edge
            existing_edge.weight = min(existing_edge.weight + 0.1, 5.0)
            existing_edge.updated_at = datetime.utcnow()
            await db.flush()
            return existing_edge

        # Create new edge
        edge = MemoryGraphEdge(
            source_node_id=source_node.id,
            target_node_id=target_node.id,
            relationship_type=relationship_type,
            properties=properties or {},
            weight=weight,
            source_conversation_id=conversation_id
        )

        db.add(edge)
        await db.flush()
        return edge

    async def get_node_by_concept(
            self,
            db: AsyncSession,
            concept: str
    ) -> Optional[MemoryGraphNode]:
        """Get a node by concept name."""
        result = await db.execute(
            select(MemoryGraphNode).where(
                MemoryGraphNode.concept == concept.lower().strip()
            )
        )
        return result.scalar_one_or_none()

    async def traverse_memory_graph(
            self,
            db: AsyncSession,
            start_concept: str,
            depth: int = 2,
            max_nodes: int = 20
    ) -> Optional[GraphMemoryResult]:
        """
        Traverse the memory graph starting from a concept.

        :param db: Database session
        :param start_concept: Starting concept
        :param depth: How many edges to traverse
        :param max_nodes: Maximum nodes to return
        :return: GraphMemoryResult with traversed nodes and edges
        """
        start_node = await self.get_node_by_concept(db, start_concept)
        if not start_node:
            return None

        visited_ids: Set[UUID] = {start_node.id}
        related_nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        current_level = [start_node]

        for current_depth in range(depth):
            if len(related_nodes) >= max_nodes:
                break

            next_level = []
            for node in current_level:
                # Get outgoing edges
                result = await db.execute(
                    select(MemoryGraphEdge)
                    .options(selectinload(MemoryGraphEdge.target_node))
                    .where(MemoryGraphEdge.source_node_id == node.id)
                )
                outgoing = result.scalars().all()

                # Get incoming edges
                result = await db.execute(
                    select(MemoryGraphEdge)
                    .options(selectinload(MemoryGraphEdge.source_node))
                    .where(MemoryGraphEdge.target_node_id == node.id)
                )
                incoming = result.scalars().all()

                # Process outgoing edges
                for edge in outgoing:
                    target = edge.target_node
                    if target.id not in visited_ids:
                        visited_ids.add(target.id)
                        related_nodes.append(self._node_to_graph_node(target))
                        next_level.append(target)
                    edges.append(GraphEdge(
                        id=str(edge.id),
                        source_concept=node.concept,
                        target_concept=target.concept,
                        relationship_type=edge.relationship_type,
                        weight=edge.weight
                    ))

                # Process incoming edges
                for edge in incoming:
                    source = edge.source_node
                    if source.id not in visited_ids:
                        visited_ids.add(source.id)
                        related_nodes.append(self._node_to_graph_node(source))
                        next_level.append(source)
                    edges.append(GraphEdge(
                        id=str(edge.id),
                        source_concept=source.concept,
                        target_concept=node.concept,
                        relationship_type=edge.relationship_type,
                        weight=edge.weight
                    ))

            current_level = next_level[:max_nodes - len(related_nodes)]

        return GraphMemoryResult(
            node=self._node_to_graph_node(start_node),
            related_nodes=related_nodes,
            edges=edges,
            depth=depth
        )

    async def get_related_memories(
            self,
            db: AsyncSession,
            concept: str,
            relationship_type: Optional[str] = None
    ) -> List[Tuple[GraphNode, str, float]]:
        """
        Get memories related to a concept.

        :param db: Database session
        :param concept: Concept to find relations for
        :param relationship_type: Optional filter by relationship type
        :return: List of (related_node, relationship_type, weight)
        """
        node = await self.get_node_by_concept(db, concept)
        if not node:
            return []

        results = []

        # Get outgoing relationships
        query = select(MemoryGraphEdge).options(
            selectinload(MemoryGraphEdge.target_node)
        ).where(MemoryGraphEdge.source_node_id == node.id)

        if relationship_type:
            query = query.where(MemoryGraphEdge.relationship_type == relationship_type)

        result = await db.execute(query)
        for edge in result.scalars().all():
            results.append((
                self._node_to_graph_node(edge.target_node),
                edge.relationship_type,
                edge.weight
            ))

        # Get incoming relationships
        query = select(MemoryGraphEdge).options(
            selectinload(MemoryGraphEdge.source_node)
        ).where(MemoryGraphEdge.target_node_id == node.id)

        if relationship_type:
            query = query.where(MemoryGraphEdge.relationship_type == relationship_type)

        result = await db.execute(query)
        for edge in result.scalars().all():
            results.append((
                self._node_to_graph_node(edge.source_node),
                edge.relationship_type,
                edge.weight
            ))

        # Sort by weight (strongest relationships first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    async def search_concepts(
            self,
            db: AsyncSession,
            query: str,
            k: int = 10,
            threshold: float = 0.5
    ) -> List[GraphNode]:
        """
        Search for concepts using semantic similarity.

        :param db: Database session
        :param query: Search query
        :param k: Number of results
        :param threshold: Minimum similarity threshold
        :return: List of matching concepts
        """
        # Generate embedding for search query
        embeddings = await generate_embeddings([query])
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        sql = """
            SELECT id, concept, concept_type, properties, mention_count,
                   first_mentioned, last_mentioned,
                   1 - (embedding <=> :embedding) AS similarity
            FROM memory_graph_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """

        result = await db.execute(
            text(sql),
            {"embedding": embedding_str, "limit": k * 2}
        )
        rows = result.mappings().fetchall()

        nodes = []
        for row in rows:
            if row["similarity"] >= threshold:
                nodes.append(GraphNode(
                    id=str(row["id"]),
                    concept=row["concept"],
                    concept_type=row["concept_type"],
                    properties=row["properties"],
                    mention_count=row["mention_count"],
                    first_mentioned=row["first_mentioned"],
                    last_mentioned=row["last_mentioned"]
                ))
            if len(nodes) >= k:
                break

        return nodes

    async def extract_concepts_from_text(
            self,
            db: AsyncSession,
            text: str,
            conversation_id: Optional[UUID] = None
    ) -> List[MemoryGraphNode]:
        """
        Extract concepts from text using LLM and add to graph.

        :param db: Database session
        :param text: Text to extract concepts from
        :param conversation_id: Source conversation
        :return: List of created/updated nodes
        """
        from langchain_openai import ChatOpenAI
        import json

        llm = ChatOpenAI(model=config.openai_model, temperature=0)

        prompt = f"""Extract key concepts from the following text. Return a JSON array of concepts.

Text: {text}

For each concept, identify:
- concept: The concept name (lowercase, simple terms)
- type: One of: topic, skill, tool, project, person, organization, preference, goal
- relationships: Array of related concepts already mentioned (if any)

Return ONLY valid JSON array. Example:
[
  {{"concept": "python", "type": "tool", "relationships": ["machine learning", "data science"]}},
  {{"concept": "tensorflow", "type": "tool", "relationships": ["python", "machine learning"]}}
]

If no meaningful concepts found, return empty array: []

Concepts:"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            concepts_data = json.loads(content)

            nodes = []
            for concept_info in concepts_data:
                concept = concept_info.get("concept", "")
                concept_type = concept_info.get("type", "topic")
                relationships = concept_info.get("relationships", [])

                if not concept:
                    continue

                # Add the concept node
                node = await self.add_memory_node(
                    db=db,
                    concept=concept,
                    concept_type=concept_type,
                    conversation_id=conversation_id
                )
                nodes.append(node)

                # Add relationships
                for related_concept in relationships:
                    if related_concept:
                        await self.add_memory_edge(
                            db=db,
                            source_concept=concept,
                            target_concept=related_concept,
                            relationship_type="related_to",
                            conversation_id=conversation_id
                        )

            return nodes

        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return []

    async def get_all_nodes(
            self,
            db: AsyncSession,
            limit: int = 100
    ) -> List[GraphNode]:
        """Get all nodes in the memory graph."""
        result = await db.execute(
            select(MemoryGraphNode)
            .order_by(MemoryGraphNode.mention_count.desc())
            .limit(limit)
        )
        nodes = result.scalars().all()
        return [self._node_to_graph_node(n) for n in nodes]

    async def get_all_edges(
            self,
            db: AsyncSession,
            limit: int = 200
    ) -> List[GraphEdge]:
        """Get all edges in the memory graph."""
        result = await db.execute(
            select(MemoryGraphEdge)
            .options(
                selectinload(MemoryGraphEdge.source_node),
                selectinload(MemoryGraphEdge.target_node)
            )
            .order_by(MemoryGraphEdge.weight.desc())
            .limit(limit)
        )
        edges = result.scalars().all()
        return [
            GraphEdge(
                id=str(e.id),
                source_concept=e.source_node.concept,
                target_concept=e.target_node.concept,
                relationship_type=e.relationship_type,
                weight=e.weight
            )
            for e in edges
        ]

    async def get_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get statistics about the memory graph."""
        nodes_result = await db.execute(select(MemoryGraphNode))
        nodes = nodes_result.scalars().all()

        edges_result = await db.execute(select(MemoryGraphEdge))
        edges = edges_result.scalars().all()

        # Count by type
        by_type = {}
        for node in nodes:
            by_type[node.concept_type] = by_type.get(node.concept_type, 0) + 1

        # Count by relationship
        by_relationship = {}
        for edge in edges:
            by_relationship[edge.relationship_type] = by_relationship.get(edge.relationship_type, 0) + 1

        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes_by_type": by_type,
            "edges_by_relationship": by_relationship,
            "most_mentioned": [
                {"concept": n.concept, "count": n.mention_count}
                for n in sorted(nodes, key=lambda x: x.mention_count, reverse=True)[:5]
            ]
        }

    async def format_for_prompt(
            self,
            db: AsyncSession,
            query: str,
            k: int = 3
    ) -> str:
        """
        Format graph memory context for prompt injection.

        :param db: Database session
        :param query: Current query to find related concepts
        :param k: Number of concepts to include
        :return: Formatted string for prompt
        """
        # Search for related concepts
        related_concepts = await self.search_concepts(db, query, k=k, threshold=0.5)

        if not related_concepts:
            return ""

        parts = ["[Related concepts from memory]"]
        for concept in related_concepts:
            # Get relationships for this concept
            relations = await self.get_related_memories(db, concept.concept)
            relation_strs = [f"{r[1]}: {r[0].concept}" for r in relations[:3]]

            parts.append(f"- {concept.concept} ({concept.concept_type})")
            if relation_strs:
                parts.append(f"  Connected to: {', '.join(relation_strs)}")

        return "\n".join(parts)

    async def delete_node(
            self,
            db: AsyncSession,
            concept: str
    ) -> bool:
        """Delete a concept node and its edges."""
        node = await self.get_node_by_concept(db, concept)
        if not node:
            return False

        await db.delete(node)
        await db.flush()
        return True

    def _node_to_graph_node(self, node: MemoryGraphNode) -> GraphNode:
        """Convert database node to GraphNode dataclass."""
        return GraphNode(
            id=str(node.id),
            concept=node.concept,
            concept_type=node.concept_type,
            properties=node.properties,
            mention_count=node.mention_count,
            first_mentioned=node.first_mentioned,
            last_mentioned=node.last_mentioned
        )
