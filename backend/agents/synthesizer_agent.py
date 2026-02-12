"""Synthesizer agent for combining and enhancing multi-source responses."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base_agent import BaseAgent, AgentResponse
from core.config import config


@dataclass
class SynthesisInput:
    """Input for synthesis from a single agent."""
    agent_name: str
    response: str
    sources: List[Dict[str, Any]]
    confidence: float = 1.0


class SynthesizerAgent(BaseAgent):
    """
    Agent that synthesizes outputs from multiple sources into a coherent response.

    Capabilities:
    - Combine outputs from multiple agents
    - Identify common themes across sources
    - Resolve conflicts or contradictions
    - Enhance responses with additional context
    - Generate comprehensive summaries
    """

    def __init__(self):
        super().__init__(
            name="Synthesizer Agent",
            description="Combines and enhances responses from multiple sources",
            model="gpt-4o-mini",
            temperature=0.5  # Lower temperature for more consistent synthesis
        )

    async def invoke(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Synthesize a response from the provided context.

        Context should contain:
        - response: The primary response from the agent
        - sources: List of sources used
        - agent_used: Which agent generated the response
        - memory_context: Memory context (optional)
        """
        context = context or {}

        primary_response = context.get("response", "")
        sources = context.get("sources", [])
        agent_used = context.get("agent_used", "unknown")
        memory_context = context.get("memory_context", {})

        # If no response to synthesize, return empty
        if not primary_response:
            return AgentResponse(
                content="I don't have enough information to provide a response.",
                metadata={"synthesized": False, "reason": "no_input"}
            )

        # Check if synthesis is needed
        needs_synthesis = self._needs_synthesis(
            response=primary_response,
            sources=sources,
            memory_context=memory_context
        )

        if not needs_synthesis:
            # Pass through with minimal enhancement
            return AgentResponse(
                content=primary_response,
                metadata={
                    "synthesized": False,
                    "agent_used": agent_used,
                    "source_count": len(sources)
                }
            )

        # Perform synthesis
        synthesized = await self._synthesize(
            query=query,
            primary_response=primary_response,
            sources=sources,
            agent_used=agent_used,
            memory_context=memory_context
        )

        return AgentResponse(
            content=synthesized,
            metadata={
                "synthesized": True,
                "agent_used": agent_used,
                "source_count": len(sources),
                "enhancement_applied": True
            }
        )

    def _needs_synthesis(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        memory_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if synthesis is needed.

        Synthesis is recommended when:
        - Multiple sources are present
        - Memory context contains relevant information
        - Response is very short or very long
        """
        # Multiple sources benefit from synthesis
        if len(sources) > 2:
            return True

        # Memory context with long-term memories
        long_term = memory_context.get("long_term_memories", [])
        if long_term:
            return True

        # Graph memories present
        graph = memory_context.get("graph_memories", [])
        if graph:
            return True

        # Very long responses might need summarization
        if len(response) > 2000:
            return True

        return False

    async def _synthesize(
        self,
        query: str,
        primary_response: str,
        sources: List[Dict[str, Any]],
        agent_used: str,
        memory_context: Dict[str, Any]
    ) -> str:
        """
        Perform the actual synthesis.
        """
        # Build context string
        context_parts = []

        # Add source information
        if sources:
            source_summary = self._summarize_sources(sources)
            context_parts.append(f"Sources used:\n{source_summary}")

        # Add memory context
        memory_summary = self._format_memory_context(memory_context)
        if memory_summary:
            context_parts.append(f"User context:\n{memory_summary}")

        context_str = "\n\n".join(context_parts) if context_parts else "No additional context available."

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_synthesis_prompt()),
            ("user", """Original query: {query}

Agent used: {agent_used}

Primary response:
{response}

Additional context:
{context}

Please synthesize a comprehensive response.""")
        ])

        chain = prompt | self.llm
        result = await chain.ainvoke({
            "query": query,
            "agent_used": agent_used,
            "response": primary_response,
            "context": context_str
        })

        return result.content

    def _get_synthesis_prompt(self) -> str:
        """Get the system prompt for synthesis."""
        return """You are a Synthesizer Agent that enhances and refines responses.

Your role is to:
1. Take the primary response from another agent
2. Incorporate any additional context (sources, user preferences, related concepts)
3. Produce a comprehensive, well-structured response

Guidelines:
- Maintain the accuracy of the original response
- Add relevant context when it enhances understanding
- If user preferences are known, tailor the response accordingly
- Keep the response concise but complete
- If sources are cited, maintain proper attribution
- Resolve any apparent conflicts between sources
- Use markdown formatting for readability when appropriate
- Do not assume the user's profession or role unless explicitly provided by the user
- Mention stored user preferences/profile details only when directly relevant to the user's query

DO NOT:
- Contradict the original response without good reason
- Add information that isn't supported by the context
- Make the response unnecessarily longer
- Remove important details from the original response"""

    def _summarize_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Summarize source information."""
        if not sources:
            return "No sources available."

        summaries = []
        for i, source in enumerate(sources[:5], 1):
            content = source.get("content", "")[:200]
            score = source.get("score", 0)
            summaries.append(f"[{i}] (relevance: {score:.2f}) {content}...")

        return "\n".join(summaries)

    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """Format memory context for the synthesis prompt."""
        parts = []

        # Long-term memories (preferences, facts)
        long_term = memory_context.get("long_term_memories", [])
        if long_term:
            prefs = []
            for mem in long_term[:3]:
                key = mem.get("key", "")
                value = mem.get("value", "")
                prefs.append(f"- {key}: {value}")
            if prefs:
                parts.append("User preferences:\n" + "\n".join(prefs))

        # Graph memories (related concepts)
        graph = memory_context.get("graph_memories", [])
        if graph:
            concepts = []
            for g in graph[:3]:
                concept = g.get("concept", {}).get("concept", "")
                if concept:
                    concepts.append(concept)
            if concepts:
                parts.append(f"Related concepts: {', '.join(concepts)}")

        # Semantic memories (similar past interactions)
        semantic = memory_context.get("semantic_memories", [])
        if semantic:
            past = semantic[0] if semantic else None
            if past:
                past_q = past.get("query", "")[:100]
                parts.append(f"Similar past query: {past_q}...")

        return "\n".join(parts) if parts else ""

    async def synthesize_multiple(
        self,
        query: str,
        inputs: List[SynthesisInput],
        db: AsyncSession
    ) -> AgentResponse:
        """
        Synthesize responses from multiple agents.

        Used for multi-modal queries where both document and data agents
        contribute to the response.
        """
        if not inputs:
            return AgentResponse(
                content="No inputs to synthesize.",
                metadata={"synthesized": False}
            )

        if len(inputs) == 1:
            # Single input, delegate to regular invoke
            return await self.invoke(
                query=query,
                db=db,
                context={
                    "response": inputs[0].response,
                    "sources": inputs[0].sources,
                    "agent_used": inputs[0].agent_name
                }
            )

        # Multiple inputs - combine them
        combined_response = await self._combine_multiple_responses(query, inputs)

        return AgentResponse(
            content=combined_response,
            metadata={
                "synthesized": True,
                "input_count": len(inputs),
                "agents_used": [inp.agent_name for inp in inputs]
            }
        )

    async def _combine_multiple_responses(
        self,
        query: str,
        inputs: List[SynthesisInput]
    ) -> str:
        """Combine responses from multiple agents."""
        # Build input summary
        input_summaries = []
        for inp in inputs:
            input_summaries.append(f"""
From {inp.agent_name} (confidence: {inp.confidence:.2f}):
{inp.response}

Sources: {len(inp.sources)} items
""")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Synthesizer Agent that combines insights from multiple AI agents.

Your task is to:
1. Identify common themes across all agent responses
2. Resolve any conflicts or contradictions
3. Produce a unified, comprehensive response
4. Highlight when different agents provide complementary information

Be concise but complete. Use the strengths of each agent's perspective."""),
            ("user", """Query: {query}

Agent Responses:
{inputs}

Please synthesize a unified response that combines the best insights from all agents.""")
        ])

        chain = prompt | self.llm
        result = await chain.ainvoke({
            "query": query,
            "inputs": "\n---\n".join(input_summaries)
        })

        return result.content
