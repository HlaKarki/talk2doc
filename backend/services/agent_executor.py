"""Agent executor service for running the LangGraph workflow."""
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from workflows.agent_workflow import create_workflow
from workflows.state import AgentState


@dataclass
class ExecutionResult:
    """Result from executing the agent workflow."""
    response: str
    intent: str
    agent_used: str
    sources: list
    confidence: float
    entities: list
    metadata: Dict[str, Any]


# Create the compiled workflow (singleton)
_workflow = None


def get_workflow():
    """Get or create the compiled workflow."""
    global _workflow
    if _workflow is None:
        graph = create_workflow()
        _workflow = graph.compile()
    return _workflow


async def execute_workflow(
    query: str,
    db: AsyncSession,
    document_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    """
    Execute the agent workflow for a given query.

    :param query: The user's query
    :param db: Database session
    :param document_id: Optional specific document to query
    :param conversation_id: Optional conversation ID for tracking
    :param context: Optional additional context
    :return: ExecutionResult with the response and metadata
    """
    workflow = get_workflow()

    # Build initial state
    initial_state: AgentState = {
        "query": query,
        "document_id": document_id,
        "conversation_id": conversation_id,
        "context": context or {},
    }

    # Create a wrapper that passes db to each node
    # LangGraph nodes need to be sync or use invoke pattern
    # We'll run the workflow step by step

    # For now, use a simpler approach: run nodes directly
    from workflows.agent_workflow import (
        router_node, document_node, knowledge_graph_node,
        data_analysis_node, general_node, route_by_intent
    )

    # Step 1: Router
    state = await router_node(initial_state, db)

    # Step 2: Route to appropriate agent
    route = route_by_intent(state)

    if route == "document":
        state = await document_node(state, db)
    elif route == "knowledge_graph":
        state = await knowledge_graph_node(state, db)
    elif route == "data_analysis":
        state = await data_analysis_node(state, db)
    else:
        state = await general_node(state, db)

    # Build result
    return ExecutionResult(
        response=state.get("response", ""),
        intent=state.get("intent", "unknown"),
        agent_used=state.get("agent_used", "unknown"),
        sources=state.get("sources", []),
        confidence=state.get("confidence", 0.0),
        entities=state.get("entities", []),
        metadata={
            "routing_reasoning": state.get("routing_reasoning"),
            "conversation_id": conversation_id,
            "document_id": document_id,
        }
    )


async def execute_workflow_stream(
    query: str,
    db: AsyncSession,
    document_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Execute the workflow and stream intermediate results.

    Yields events like:
    - {"type": "routing", "intent": "document_query", "confidence": 0.95}
    - {"type": "processing", "agent": "document_agent"}
    - {"type": "response", "content": "...", "sources": [...]}
    """
    from workflows.agent_workflow import (
        router_node, document_node, knowledge_graph_node,
        data_analysis_node, general_node, route_by_intent
    )

    initial_state: AgentState = {
        "query": query,
        "document_id": document_id,
        "conversation_id": conversation_id,
    }

    # Step 1: Router
    yield {"type": "routing", "status": "classifying intent..."}

    state = await router_node(initial_state, db)

    yield {
        "type": "routing",
        "intent": state.get("intent"),
        "confidence": state.get("confidence"),
        "entities": state.get("entities"),
        "reasoning": state.get("routing_reasoning")
    }

    # Step 2: Route to agent
    route = route_by_intent(state)

    yield {"type": "processing", "agent": f"{route}_agent"}

    if route == "document":
        state = await document_node(state, db)
    elif route == "knowledge_graph":
        state = await knowledge_graph_node(state, db)
    elif route == "data_analysis":
        state = await data_analysis_node(state, db)
    else:
        state = await general_node(state, db)

    # Final response
    yield {
        "type": "response",
        "content": state.get("response", ""),
        "sources": state.get("sources", []),
        "agent_used": state.get("agent_used")
    }
