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
    DATA_ANALYSIS = "data_analysis"         # Data/analytics questions (future)
    GENERAL = "general"                      # General conversation


class RouterDecision(BaseModel):
    """Structured output from the router."""
    intent: Intent = Field(description="The classified intent of the user query")
    confidence: float = Field(description="Confidence score from 0 to 1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")
    entities: list[str] = Field(default_factory=list, description="Key entities extracted from the query")


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a document Q&A system.

Analyze the user's query and classify it into one of these intents:

1. **document_query**: Questions about document content, asking for information from uploaded documents.
   Examples: "What does the document say about X?", "Summarize the key points", "Find information about Y"

2. **knowledge_graph**: Questions about relationships between concepts/entities.
   Examples: "How is X related to Y?", "What concepts are connected to Z?", "Show me related topics"

3. **data_analysis**: Questions about analyzing data, statistics, or datasets.
   Examples: "Analyze this data", "What's the correlation?", "Show me trends" (Note: This is for future use)

4. **general**: General conversation, greetings, or questions not related to documents/data.
   Examples: "Hello", "What can you do?", "Help me understand how to use this"

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
            # Use partial_variables to avoid escaping issues with format_instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_SYSTEM_PROMPT),
                ("user", "{query}")
            ]).partial(format_instructions=self.parser.get_format_instructions())

            chain = prompt | self.llm | self.parser

            decision: RouterDecision = await chain.ainvoke({"query": query})

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
        db: AsyncSession
    ) -> RouterDecision:
        """
        Convenience method that returns the RouterDecision directly.
        """
        response = await self.invoke(query, db)

        if not response.success:
            # Return a default decision on error
            return RouterDecision(
                intent=Intent.GENERAL,
                confidence=0.0,
                reasoning=f"Error during classification: {response.error}",
                entities=[]
            )

        return RouterDecision(**response.metadata["decision"])
