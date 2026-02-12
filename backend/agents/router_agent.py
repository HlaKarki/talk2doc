"""Router agent for classifying user intent and routing to appropriate agents."""
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base_agent import BaseAgent, AgentResponse


class Intent(str, Enum):
    """Possible user intents."""
    DOCUMENT_QUERY = "document_query"      # Questions about document content
    KNOWLEDGE_GRAPH = "knowledge_graph"     # Questions about entity relationships
    DATA_ANALYSIS = "data_analysis"         # Data/analytics questions
    GENERAL = "general"                      # General conversation


class RouterDecision(BaseModel):
    """Structured output from the router."""
    intent: Intent = Field(description="The classified intent of the user query")
    confidence: float = Field(description="Confidence score from 0 to 1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")
    entities: list[str] = Field(default_factory=list, description="Key entities extracted from the query")


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a document + dataset assistant.

Analyze the user's query and classify it into one of these intents:

1. **document_query**: Questions about document content, asking for information from uploaded documents.
   Examples: "What does the document say about X?", "Summarize the key points", "Find information about Y"

2. **knowledge_graph**: Questions about relationships between concepts/entities, or requests to extract/build a graph from documents.
   Examples: "How is X related to Y?", "What concepts are connected to Z?", "Show me related topics", "Extract the knowledge graph for this document", "Build a graph from this PDF"

3. **data_analysis**: Questions about analyzing data, statistics, datasets, ML, clustering, or classification.
   Examples: "Analyze this data", "What's the correlation?", "Train a model", "Classify this dataset", "Cluster customers"

4. **general**: General conversation, greetings, or questions not related to documents/data.
   Examples: "Hello", "What can you do?", "Help me understand how to use this"

You will also receive selected resource context (if any):
- selected_document_id
- selected_dataset_id

Routing rules with selected resource context:
- If the query is ambiguous (e.g., "summarize this", "explain it"), prefer `document_query` when a document is selected.
- If the query asks for analysis/modeling and a dataset is selected, prefer `data_analysis`.
- If the query asks to extract/build/create a knowledge graph and a document is selected, prefer `knowledge_graph`.
- Do not force data/document routing for clearly general conversation.

Extract any key entities (concepts, names, topics) mentioned in the query.

{format_instructions}"""


class RouterAgent(BaseAgent):
    """
    Router agent that classifies user intent and decides which agent should handle the query.
    """

    def __init__(self):
        super().__init__(
            name="Router",
            description="Classifies user intent and routes to the appropriate specialized agent.",
            model="gpt-4o-mini",
            temperature=0  # Deterministic for classification
        )
        self.parser = PydanticOutputParser(pydantic_object=RouterDecision)

    def get_system_prompt(self) -> str:
        return ROUTER_SYSTEM_PROMPT.format(
            format_instructions=self.parser.get_format_instructions()
        )

    @staticmethod
    def _format_selection_context(context: Optional[Dict[str, Any]]) -> str:
        """Format selected resource hints for the router prompt."""
        if not context:
            return "No selected resource."

        lines = []
        if context.get("document_id"):
            lines.append(f"selected_document_id: {context['document_id']}")
        if context.get("dataset_id"):
            lines.append(f"selected_dataset_id: {context['dataset_id']}")

        return "\n".join(lines) if lines else "No selected resource."

    async def invoke(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Classify the user's query intent.

        :param query: The user's query
        :param db: Database session (not used by router but required by interface)
        :param context: Optional context (e.g., conversation history)
        :return: AgentResponse containing RouterDecision
        """
        try:
            selection_context = self._format_selection_context(context)

            # Use partial_variables to avoid escaping issues with format_instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_SYSTEM_PROMPT),
                ("user", "Selected resource context:\n{selection_context}\n\nUser query:\n{query}")
            ]).partial(format_instructions=self.parser.get_format_instructions())

            chain = prompt | self.llm | self.parser

            decision: RouterDecision = await chain.ainvoke({
                "query": query,
                "selection_context": selection_context
            })

            return AgentResponse(
                content=decision.intent.value,
                success=True,
                metadata={
                    "agent": self.name,
                    "intent": decision.intent.value,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "entities": decision.entities,
                    "decision": decision.model_dump()
                }
            )

        except Exception as e:
            return await self.handle_error(e)

    async def classify(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> RouterDecision:
        """
        Convenience method that returns the RouterDecision directly.
        """
        response = await self.invoke(query, db, context=context)

        if not response.success:
            # Return a default decision on error
            return RouterDecision(
                intent=Intent.GENERAL,
                confidence=0.0,
                reasoning=f"Error during classification: {response.error}",
                entities=[]
            )

        return RouterDecision(**response.metadata["decision"])
