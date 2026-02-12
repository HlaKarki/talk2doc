"""Agent executor service for running the LangGraph workflow."""
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from workflows.agent_workflow import create_workflow
from workflows.state import AgentState
from services.memory.memory_manager import get_memory_manager
from database.models import Conversation


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
    conversation_id: Optional[str] = None


# Create the compiled workflow (singleton)
_workflow = None


def get_workflow():
    """Get or create the compiled workflow."""
    global _workflow
    if _workflow is None:
        graph = create_workflow()
        _workflow = graph.compile()
    return _workflow


def _selection_type(document_id: Optional[str], dataset_id: Optional[str]) -> Optional[str]:
    """Return normalized selection type."""
    if document_id:
        return "document"
    if dataset_id:
        return "dataset"
    return None


async def _resolve_selected_resources(
    db: AsyncSession,
    conversation_id: UUID,
    document_id: Optional[str],
    dataset_id: Optional[str]
) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Resolve selected document/dataset from request or conversation metadata.

    Priority:
    1. Request payload (`document_id` / `dataset_id`)
    2. Conversation metadata fallback
    """
    if document_id and dataset_id:
        raise ValueError("Provide only one of 'document_id' or 'dataset_id'.")

    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        return document_id, dataset_id, {
            "selected_resource_source": "request" if (document_id or dataset_id) else "none",
            "selected_resource_type": _selection_type(document_id, dataset_id),
        }

    conv_meta = conversation.conv_metadata or {}
    source = "request"

    if document_id:
        conv_meta["selected_document_id"] = document_id
        conv_meta.pop("selected_dataset_id", None)
        conv_meta["selected_resource_type"] = "document"
    elif dataset_id:
        conv_meta["selected_dataset_id"] = dataset_id
        conv_meta.pop("selected_document_id", None)
        conv_meta["selected_resource_type"] = "dataset"
    else:
        document_id = conv_meta.get("selected_document_id")
        dataset_id = conv_meta.get("selected_dataset_id")

        # Handle legacy/invalid metadata where both were stored.
        if document_id and dataset_id:
            if conv_meta.get("selected_resource_type") == "dataset":
                document_id = None
            else:
                dataset_id = None

        source = "conversation" if (document_id or dataset_id) else "none"

    conversation.conv_metadata = conv_meta
    await db.flush()

    return document_id, dataset_id, {
        "selected_resource_source": source,
        "selected_resource_type": _selection_type(document_id, dataset_id),
    }


async def execute_workflow(
    query: str,
    db: AsyncSession,
    document_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    """
    Execute the agent workflow for a given query.

    :param query: The user's query
    :param db: Database session
    :param document_id: Optional specific document to query
    :param dataset_id: Optional specific dataset to analyze
    :param conversation_id: Optional conversation ID for tracking
    :param context: Optional additional context
    :return: ExecutionResult with the response and metadata
    """
    memory_manager = get_memory_manager()

    # Parse conversation_id if provided
    conv_uuid = UUID(conversation_id) if conversation_id else None

    # Start or get conversation
    conv_uuid = await memory_manager.start_conversation(
        db=db,
        conversation_id=conv_uuid
    )

    # Resolve selected resource from request/conversation metadata
    document_id, dataset_id, selection_meta = await _resolve_selected_resources(
        db=db,
        conversation_id=conv_uuid,
        document_id=document_id,
        dataset_id=dataset_id
    )

    # Build route context including selected resources
    route_context = dict(context or {})
    if document_id:
        route_context["document_id"] = document_id
    if dataset_id:
        route_context["dataset_id"] = dataset_id

    # Get memory context
    memory_context = await memory_manager.get_context_for_query(
        query=query,
        db=db,
        conversation_id=conv_uuid
    )

    # Get formatted conversation history for prompt
    conversation_history = await memory_manager.format_for_prompt(
        db=db,
        conversation_id=conv_uuid,
        query=query
    )

    # Build initial state with memory
    initial_state: AgentState = {
        "query": query,
        "document_id": document_id,
        "dataset_id": dataset_id,
        "conversation_id": str(conv_uuid),
        "context": route_context,
        "memory_context": memory_context.to_dict(),
        "conversation_history": conversation_history,
    }

    # For now, use a simpler approach: run nodes directly
    from workflows.agent_workflow import (
        router_node, document_node, knowledge_graph_node,
        data_analysis_node, general_node, synthesizer_node, route_by_intent
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

    # Step 3: Synthesize the response
    state = await synthesizer_node(state, db)

    # Save interaction to memory
    response_text = state.get("response", "")
    await memory_manager.update_memory(
        db=db,
        conversation_id=conv_uuid,
        query=query,
        response=response_text,
        metadata={
            "intent": state.get("intent"),
            "agent_used": state.get("agent_used"),
            "sources": state.get("sources", []),
            "document_id": document_id,
            "dataset_id": dataset_id,
        }
    )

    # Build result
    return ExecutionResult(
        response=response_text,
        intent=state.get("intent", "unknown"),
        agent_used=state.get("agent_used", "unknown"),
        sources=state.get("sources", []),
        confidence=state.get("confidence", 0.0),
        entities=state.get("entities", []),
        conversation_id=str(conv_uuid),
        metadata={
            "routing_reasoning": state.get("routing_reasoning"),
            "conversation_id": str(conv_uuid),
            "document_id": document_id,
            "dataset_id": dataset_id,
            **selection_meta,
            "has_history": memory_context.has_history,
            "analysis_results": state.get("analysis_results"),
            "analysis_plan": state.get("analysis_plan"),
            "visualizations": state.get("visualizations", []),
        }
    )


async def execute_workflow_stream(
    query: str,
    db: AsyncSession,
    document_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
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
        data_analysis_node, general_node, synthesizer_node, route_by_intent
    )

    memory_manager = get_memory_manager()

    # Parse conversation_id if provided
    conv_uuid = UUID(conversation_id) if conversation_id else None

    # Start or get conversation
    conv_uuid = await memory_manager.start_conversation(
        db=db,
        conversation_id=conv_uuid
    )

    # Resolve selected resource from request/conversation metadata
    document_id, dataset_id, selection_meta = await _resolve_selected_resources(
        db=db,
        conversation_id=conv_uuid,
        document_id=document_id,
        dataset_id=dataset_id
    )

    yield {"type": "memory", "status": "loading context...", "conversation_id": str(conv_uuid)}

    # Get memory context
    memory_context = await memory_manager.get_context_for_query(
        query=query,
        db=db,
        conversation_id=conv_uuid
    )

    # Get formatted conversation history
    conversation_history = await memory_manager.format_for_prompt(
        db=db,
        conversation_id=conv_uuid,
        query=query
    )

    initial_state: AgentState = {
        "query": query,
        "document_id": document_id,
        "dataset_id": dataset_id,
        "conversation_id": str(conv_uuid),
        "context": {
            "document_id": document_id,
            "dataset_id": dataset_id,
        },
        "memory_context": memory_context.to_dict(),
        "conversation_history": conversation_history,
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

    # Step 3: Synthesize
    yield {"type": "synthesizing", "status": "enhancing response..."}
    state = await synthesizer_node(state, db)

    # Save interaction to memory
    response_text = state.get("response", "")
    await memory_manager.update_memory(
        db=db,
        conversation_id=conv_uuid,
        query=query,
        response=response_text,
        metadata={
            "intent": state.get("intent"),
            "agent_used": state.get("agent_used"),
            "sources": state.get("sources", []),
            "document_id": document_id,
            "dataset_id": dataset_id,
        }
    )

    # Final response
    yield {
        "type": "response",
        "content": response_text,
        "sources": state.get("sources", []),
        "agent_used": state.get("agent_used"),
        "conversation_id": str(conv_uuid),
        "document_id": document_id,
        "dataset_id": dataset_id,
        **selection_meta
    }
