"""LangGraph workflow for agent orchestration."""
from typing import Literal

from langgraph.graph import StateGraph, END, START
from sqlalchemy.ext.asyncio import AsyncSession

from workflows.state import AgentState, Intent
from agents.router_agent import RouterAgent


# Initialize agents
router_agent = RouterAgent()


async def router_node(state: AgentState, db: AsyncSession) -> AgentState:
    """
    Router node that classifies intent and updates state.
    """
    query = state["query"]

    decision = await router_agent.classify(query, db)

    return {
        **state,
        "intent": decision.intent.value,
        "confidence": decision.confidence,
        "entities": decision.entities,
        "routing_reasoning": decision.reasoning,
    }


async def document_node(state: AgentState, db: AsyncSession) -> AgentState:
    """
    Document agent node - handles document queries using RAG.
    (Full implementation in Iteration 10)
    """
    from services import rag_service

    query = state["query"]
    document_id = state.get("document_id")

    # Use existing RAG service
    if document_id:
        from uuid import UUID
        response = await rag_service.query_documents(
            query=query,
            db=db,
            document_id=UUID(document_id)
        )
    else:
        response = await rag_service.query_documents(query=query, db=db)

    sources = [
        {
            "chunk_index": s.chunk_index,
            "content": s.content,
            "score": s.score,
            "document_id": str(s.document_id)
        }
        for s in response.sources
    ]

    return {
        **state,
        "response": response.answer,
        "sources": sources,
        "agent_used": "document_agent"
    }


async def knowledge_graph_node(state: AgentState, db: AsyncSession) -> AgentState:
    """
    Knowledge graph agent node - handles entity relationship queries.
    """
    from services import knowledge_graph_service

    query = state["query"]
    entities = state.get("entities", [])

    # Search for relevant entities
    search_results = await knowledge_graph_service.search_entities(
        query=query,
        db=db,
        k=5
    )

    if not search_results:
        return {
            **state,
            "response": "I couldn't find any relevant entities in the knowledge graph for your query.",
            "sources": [],
            "agent_used": "knowledge_graph_agent"
        }

    # Get neighbors for top entity
    top_entity = search_results[0]
    from uuid import UUID
    neighbors = await knowledge_graph_service.get_node_neighbors(
        node_id=UUID(top_entity["id"]),
        db=db,
        depth=1
    )

    # Build response
    entity_info = f"**{top_entity['name']}** ({top_entity['type']})"

    neighbor_info = ""
    if neighbors and neighbors.get("neighbors"):
        neighbor_names = [n["name"] for n in neighbors["neighbors"][:5]]
        neighbor_info = f"\n\nRelated concepts: {', '.join(neighbor_names)}"

    response = f"Found entity: {entity_info}{neighbor_info}"

    return {
        **state,
        "response": response,
        "sources": search_results[:3],
        "agent_used": "knowledge_graph_agent"
    }


async def general_node(state: AgentState, db: AsyncSession) -> AgentState:
    """
    General conversation node - handles greetings and general questions.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from core.config import config

    query = state["query"]
    conversation_history = state.get("conversation_history", "")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=config.openai_api_key
    )

    # Build system prompt with conversation history
    system_prompt = """You are a helpful assistant for a document Q&A system called Talk2Doc.

You can help users:
- Ask questions about uploaded documents
- Explore relationships in the knowledge graph
- Analyze data (coming soon)

Be friendly and helpful. If the user seems to want document-related help,
guide them on how to upload documents or ask questions."""

    if conversation_history:
        system_prompt += f"""

Previous conversation context:
{conversation_history}

Use this context to provide relevant and consistent responses."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{query}")
    ])

    chain = prompt | llm
    response = await chain.ainvoke({"query": query})

    return {
        **state,
        "response": response.content,
        "sources": [],
        "agent_used": "general_agent"
    }


async def data_analysis_node(state: AgentState, db: AsyncSession) -> AgentState:
    """
    Data analysis node - placeholder for future implementation.
    """
    return {
        **state,
        "response": "Data analysis features are coming soon! For now, I can help you with document questions or exploring the knowledge graph.",
        "sources": [],
        "agent_used": "data_analysis_agent"
    }


def route_by_intent(state: AgentState) -> Literal["document", "knowledge_graph", "data_analysis", "general"]:
    """
    Route to the appropriate agent based on classified intent.
    """
    intent = state.get("intent", "general")

    if intent == Intent.DOCUMENT_QUERY.value:
        return "document"
    elif intent == Intent.KNOWLEDGE_GRAPH.value:
        return "knowledge_graph"
    elif intent == Intent.DATA_ANALYSIS.value:
        return "data_analysis"
    else:
        return "general"


def create_workflow() -> StateGraph:
    """
    Create and return the agent workflow graph.

    Workflow structure:
        START → router → [document | knowledge_graph | data_analysis | general] → END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("document", document_node)
    workflow.add_node("knowledge_graph", knowledge_graph_node)
    workflow.add_node("data_analysis", data_analysis_node)
    workflow.add_node("general", general_node)

    # Add edges
    workflow.add_edge(START, "router")

    # Conditional routing based on intent
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "document": "document",
            "knowledge_graph": "knowledge_graph",
            "data_analysis": "data_analysis",
            "general": "general"
        }
    )

    # All agents lead to END
    workflow.add_edge("document", END)
    workflow.add_edge("knowledge_graph", END)
    workflow.add_edge("data_analysis", END)
    workflow.add_edge("general", END)

    return workflow
