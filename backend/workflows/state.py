"""State definitions for the agent workflow."""
from typing import TypedDict, Optional, List, Any, Dict
from enum import Enum


class Intent(str, Enum):
    """Possible user intents (mirrors router_agent.Intent)."""
    DOCUMENT_QUERY = "document_query"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DATA_ANALYSIS = "data_analysis"
    GENERAL = "general"


class AgentState(TypedDict, total=False):
    """
    State that flows through the agent workflow.

    This state is passed between nodes in the LangGraph workflow,
    allowing agents to share information and build on each other's work.
    """
    # Input
    query: str                          # The user's original queryx
    conversation_id: Optional[str]      # For conversation tracking

    # Router output
    intent: Optional[str]               # Classified intent
    confidence: Optional[float]         # Router confidence score
    entities: Optional[List[str]]       # Extracted entities
    routing_reasoning: Optional[str]    # Why this route was chosen

    # Agent outputs
    response: Optional[str]             # The generated response
    sources: Optional[List[dict]]       # Source citations
    agent_used: Optional[str]           # Which agent handled the query

    # Context
    context: Optional[dict]             # Additional context (e.g., from memory)
    document_id: Optional[str]          # Specific document to query

    # Memory context
    memory_context: Optional[Dict[str, Any]]  # Context from memory layers
    conversation_history: Optional[str]        # Formatted conversation history

    # Synthesizer output
    synthesized: Optional[bool]                  # Whether response was synthesized
    synthesis_metadata: Optional[Dict[str, Any]] # Synthesis details

    # Data analysis output
    analysis_results: Optional[Dict[str, Any]]   # Results from data analysis
    visualizations: Optional[List[Dict]]         # Generated visualizations
    analysis_plan: Optional[Dict[str, Any]]      # The analysis plan that was executed

    # Metadata
    error: Optional[str]                # Error message if something failed
    metadata: Optional[dict]            # Additional metadata
