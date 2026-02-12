import json
from dataclasses import dataclass
from typing import List, Optional, AsyncGenerator, Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from services import vector_store


@dataclass
class Source:
    """Source citation from a document chunk."""
    chunk_index: int
    content: str
    score: float
    document_id: UUID


@dataclass
class QueryResponse:
    """Response from RAG query."""
    answer: str
    sources: List[Source]

RERANK_PROMPT = """Score how relevant this text passage is to answering the question.
Return ONLY a number from 0 to 10, where:
- 0: Completely irrelevant
- 5: Somewhat relevant
- 10: Highly relevant and directly answers the question

Question: {query}

Passage: {passage}

Relevance score (0-10):"""

QUERY_REWRITE_PROMPT = """Rewrite the following user question to be more specific and search-friendly.
Keep the core intent but expand abbreviations, add relevant synonyms, and make it clearer.
Only output the rewritten query, nothing else.

Original question: {query}

Rewritten query:"""

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from documents.

  Instructions:
  - Answer based ONLY on the provided context
  - If the context doesn't contain enough information to answer, say "I don't have enough information in the documents to 
  answer this question."
  - Be concise and direct
  - Cite your sources using [Source 1], [Source 2], etc. when referencing specific information
  - Do not make up information that isn't in the context"""


def _extract_text(content: Any) -> str:
    """Best-effort extraction of text from model content payloads."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _is_empty_answer(answer: str) -> bool:
    """Return True if answer is effectively empty."""
    return not (answer or "").strip()

def get_llm(streaming: bool = False) -> ChatOpenAI:
    """Get a configured LLM instance."""
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.6,
        max_tokens=1000,
        api_key=config.openai_api_key,
        streaming=streaming
    )


def get_fallback_llm(streaming: bool = False) -> ChatOpenAI:
    """Fallback model when primary model returns no visible text output."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
        api_key=config.openai_api_key,
        streaming=streaming
    )


async def rerank_chunks(
    query: str,
    chunks: List[vector_store.SearchResult],
    top_k: int = 5
) -> List[vector_store.SearchResult]:
    """
    Re-rank chunks using LLM scoring for better relevance.
    Returns top_k chunks sorted by relevance score.
    """
    if not chunks:
        return chunks

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        max_tokens=10,
        api_key=config.openai_api_key
    )

    prompt = ChatPromptTemplate.from_template(RERANK_PROMPT)
    chain = prompt | llm

    scored_chunks = []
    for chunk in chunks:
        try:
            response = await chain.ainvoke({
                "query": query,
                "passage": chunk.content[:500]  # Limit passage length
            })
            score_text = response.content.strip() if response.content else "0"
            # Extract number from response
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.') or '0')
            score = min(10, max(0, score))  # Clamp to 0-10
        except Exception:
            score = chunk.score * 10  # Fallback to original score

        scored_chunks.append((chunk, score))

    # Sort by re-rank score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_k with updated scores (normalized to 0-1)
    return [
        vector_store.SearchResult(
            content=chunk.content,
            score=score / 10,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            chunk_metadata=chunk.chunk_metadata
        )
        for chunk, score in scored_chunks[:top_k]
    ]


async def preprocess_query(query: str) -> str:
    """
    Rewrite query to improve retrieval.
    Expands abbreviations, adds context, makes query more specific.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.3,
        max_tokens=200,
        api_key=config.openai_api_key
    )

    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain = prompt | llm

    response = await chain.ainvoke({"query": query})
    rewritten = response.content.strip() if response.content else query

    return rewritten


def build_context(chunks: List[vector_store.SearchResult]) -> str:
    """Build context string from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}]: {chunk.content}")
    return "\n\n".join(context_parts)


async def query_documents(
        query: str,
        db: AsyncSession,
        document_id: Optional[UUID] = None,
        k: int = 5,
        preprocess: bool = False,
        rerank: bool = False,
) -> QueryResponse:
    """
    Query documents using RAG pipeline with LangChain.
    :param query: The user's question
    :param db: Database session
    :param document_id: Optional specific document to query
    :param k: Number of chunks to retrieve
    :param preprocess: Whether to preprocess/rewrite the query
    :param rerank: Whether to re-rank chunks using LLM scoring
    :return: QueryResponse with answer and sources
    """
    # Step 0: Optionally preprocess query
    search_query = await preprocess_query(query) if preprocess else query

    # Step 1: Retrieve relevant chunks (fetch more if re-ranking)
    fetch_k = k * 2 if rerank else k
    chunks = await vector_store.similarity_search(search_query, db, k=fetch_k, document_id=document_id)

    # Step 1.5: Optionally re-rank chunks
    if rerank and chunks:
        chunks = await rerank_chunks(query, chunks, top_k=k)

    if not chunks:
        return QueryResponse(answer="No relevant information found in the documents.", sources=[])

    # Step 2: Build context from chunks
    context = build_context(chunks)

    # Step 3: Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", """Context from documents:
{context}

Question: {question}

Answer:""")
    ])

    # Step 4: Initialize LLM with LangChain
    llm = get_llm()

    # Step 5: Create chain and invoke
    chain = prompt | llm
    response = await chain.ainvoke({
        "context": context,
        "question": query
    })

    answer = _extract_text(response.content)

    # Fallback when reasoning-only output yields empty visible content.
    if _is_empty_answer(answer):
        try:
            fallback_llm = get_fallback_llm()
            fallback_chain = prompt | fallback_llm
            fallback_response = await fallback_chain.ainvoke({
                "context": context,
                "question": query
            })
            answer = _extract_text(fallback_response.content)
        except Exception:
            answer = ""

    if _is_empty_answer(answer):
        answer = "I don't have enough information in the documents to answer this question."

    # Step 6: Build sources list
    sources = [
        Source(
            chunk_index=chunk.chunk_index,
            content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            score=chunk.score,
            document_id=chunk.document_id
        )
        for chunk in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)


async def stream_query_documents(
        query: str,
        db: AsyncSession,
        document_id: Optional[UUID] = None,
        k: int = 5,
        preprocess: bool = False,
        rerank: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream query response token-by-token using SSE format.
    Yields SSE-formatted events: sources first, then token chunks, then done.
    """
    # Step 0: Optionally preprocess query
    search_query = await preprocess_query(query) if preprocess else query

    # Step 1: Retrieve relevant chunks (fetch more if re-ranking)
    fetch_k = k * 2 if rerank else k
    chunks = await vector_store.similarity_search(search_query, db, k=fetch_k, document_id=document_id)

    # Step 1.5: Optionally re-rank chunks
    if rerank and chunks:
        chunks = await rerank_chunks(query, chunks, top_k=k)

    if not chunks:
        yield f"data: {json.dumps({'type': 'error', 'content': 'No relevant information found in the documents.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Step 2: Build sources and send them first
    sources = [
        {
            "chunk_index": chunk.chunk_index,
            "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "score": chunk.score,
            "document_id": str(chunk.document_id)
        }
        for chunk in chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

    # Step 3: Build context
    context = build_context(chunks)

    # Step 4: Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", """Context from documents:
{context}

Question: {question}

Answer:""")
    ])

    # Step 5: Initialize LLM with streaming enabled
    llm = get_llm(streaming=True)

    # Step 6: Create chain and stream
    chain = prompt | llm
    emitted_tokens = False
    async for chunk in chain.astream({"context": context, "question": query}):
        if chunk.content:
            emitted_tokens = True
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

    # Fallback when stream emits nothing (e.g., reasoning-only output).
    if not emitted_tokens:
        try:
            fallback_llm = get_fallback_llm(streaming=False)
            fallback_chain = prompt | fallback_llm
            fallback_response = await fallback_chain.ainvoke({
                "context": context,
                "question": query
            })
            fallback_text = _extract_text(fallback_response.content)
        except Exception:
            fallback_text = ""

        if _is_empty_answer(fallback_text):
            fallback_text = "I don't have enough information in the documents to answer this question."

        yield f"data: {json.dumps({'type': 'token', 'content': fallback_text})}\n\n"

    yield "data: [DONE]\n\n"
