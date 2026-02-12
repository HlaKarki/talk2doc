"""Runtime guard tests for workflow routing and empty-response fallbacks.

Run with:
    cd backend && .venv/bin/python tests/test_workflow_guards.py

These tests do not require a running API server, database, or network.
"""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# Ensure backend package modules are importable when running this file directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workflows.agent_workflow import knowledge_graph_node, router_node
from workflows.state import Intent
from services import rag_service
from services.vector_store import SearchResult


def _check(condition: bool, name: str, details: str = "") -> None:
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
    if details:
        print(f"       {details}")
    if not condition:
        raise AssertionError(name)


async def test_router_override_for_explicit_kg_extraction() -> None:
    import workflows.agent_workflow as workflow

    async def fake_classify(query, db, context=None):
        return SimpleNamespace(
            intent=Intent.DOCUMENT_QUERY,
            confidence=0.61,
            entities=["knowledge graph"],
            reasoning="default router output",
        )

    original = workflow.router_agent.classify
    workflow.router_agent.classify = fake_classify
    try:
        state = {
            "query": "hi, what's my name? and can you extract the knowledge graph for this document?",
            "document_id": str(uuid4()),
            "context": {},
        }
        out = await router_node(state, db=None)
        _check(out["intent"] == Intent.KNOWLEDGE_GRAPH.value, "router override sets knowledge_graph")
        _check(out["confidence"] >= 0.95, "router override boosts confidence", f"confidence={out['confidence']}")
    finally:
        workflow.router_agent.classify = original


async def test_kg_extraction_response_passthrough() -> None:
    import services

    stub = types.ModuleType("services.knowledge_graph_service")

    async def build_graph_from_document(document_id, db):
        return {"nodes_created": 7, "edges_created": 11}

    async def search_entities(query, db, k=5):  # pragma: no cover
        return []

    async def get_node_neighbors(node_id, db, depth=1):  # pragma: no cover
        return {"neighbors": []}

    stub.build_graph_from_document = build_graph_from_document
    stub.search_entities = search_entities
    stub.get_node_neighbors = get_node_neighbors

    old_submodule = sys.modules.get("services.knowledge_graph_service")
    old_attr = getattr(services, "knowledge_graph_service", None)
    sys.modules["services.knowledge_graph_service"] = stub
    services.knowledge_graph_service = stub

    try:
        state = {
            "query": "extract the knowledge graph for this document",
            "document_id": str(uuid4()),
            "entities": [],
        }
        out = await knowledge_graph_node(state, db=None)
        _check(out.get("agent_used") == "knowledge_graph_agent", "kg node uses knowledge_graph_agent")
        _check(out.get("skip_synthesis") is True, "kg extraction sets skip_synthesis")
        _check("Nodes created: 7" in out.get("response", ""), "kg extraction includes node count")
        _check("edges created: 11" in out.get("response", ""), "kg extraction includes edge count")
    finally:
        if old_submodule is not None:
            sys.modules["services.knowledge_graph_service"] = old_submodule
        else:
            sys.modules.pop("services.knowledge_graph_service", None)

        if old_attr is not None:
            services.knowledge_graph_service = old_attr
        elif hasattr(services, "knowledge_graph_service"):
            delattr(services, "knowledge_graph_service")


async def test_rag_fallback_when_primary_returns_empty() -> None:
    async def fake_similarity_search(query, db, k=5, document_id=None):
        return [
            SearchResult(
                content="Document says the intro covers architecture and usage.",
                score=0.9,
                chunk_id=uuid4(),
                document_id=uuid4(),
                chunk_index=0,
                chunk_metadata={},
            )
        ]

    def primary_llm(streaming=False):
        return RunnableLambda(lambda _: AIMessage(content=""))

    def fallback_llm(streaming=False):
        return RunnableLambda(lambda _: AIMessage(content="Fallback visible answer"))

    original_similarity_search = rag_service.vector_store.similarity_search
    original_get_llm = rag_service.get_llm
    original_get_fallback_llm = rag_service.get_fallback_llm

    rag_service.vector_store.similarity_search = fake_similarity_search
    rag_service.get_llm = primary_llm
    rag_service.get_fallback_llm = fallback_llm

    try:
        out = await rag_service.query_documents(
            query="what can you tell me about this document?",
            db=None,
            document_id=None,
            preprocess=False,
            rerank=False,
        )
        _check(out.answer == "Fallback visible answer", "rag fallback returns visible answer", f"answer={out.answer!r}")
        _check(len(out.sources) == 1, "rag fallback keeps sources", f"sources={len(out.sources)}")
    finally:
        rag_service.vector_store.similarity_search = original_similarity_search
        rag_service.get_llm = original_get_llm
        rag_service.get_fallback_llm = original_get_fallback_llm


async def main() -> None:
    print("=" * 60)
    print("Workflow Guard Tests")
    print("=" * 60)

    await test_router_override_for_explicit_kg_extraction()
    await test_kg_extraction_response_passthrough()
    await test_rag_fallback_when_primary_returns_empty()

    print("=" * 60)
    print("All workflow guard tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
