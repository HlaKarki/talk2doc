"""Base agent class with common functionality for all agents."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config


@dataclass
class AgentResponse:
    """Standard response from an agent."""
    content: str
    metadata: Dict[str, Any] = None
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    Provides common functionality:
    - LLM access
    - Database session management
    - Standard response format
    - Error handling
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-load the LLM instance."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=config.openai_api_key
            )
        return self._llm

    @abstractmethod
    async def invoke(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a query and return a response.

        :param query: The user's query
        :param db: Database session for data access
        :param context: Optional context from previous steps
        :return: AgentResponse with the result
        """
        pass

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        Override in subclasses for specialized behavior.
        """
        return f"""You are {self.name}, an AI assistant.
{self.description}

Be helpful, accurate, and concise in your responses."""

    async def handle_error(self, error: Exception) -> AgentResponse:
        """Handle errors gracefully."""
        return AgentResponse(
            content="",
            success=False,
            error=str(error),
            metadata={"agent": self.name, "error_type": type(error).__name__}
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
