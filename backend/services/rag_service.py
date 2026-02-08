from dataclasses import dataclass
from typing import List, Optional
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

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from documents.

  Instructions:
  - Answer based ONLY on the provided context
  - If the context doesn't contain enough information to answer, say "I don't have enough information in the documents to 
  answer this question."
  - Be concise and direct
  - Cite your sources using [Source 1], [Source 2], etc. when referencing specific information
  - Do not make up information that isn't in the context"""

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
) -> QueryResponse:
    """
    Query documents using RAG pipeline with LangChain.
    :param query: The user's question
    :param db: Database session
    :param document_id: Optional specific document to query
    :param k: Number of chunks to retrieve
    :return: QueryResponse with answer and sources
    """
    # Step 1: Retrieve relevant chunks
    chunks = await vector_store.similarity_search(query, db, k=k, document_id=document_id)

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
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.6,
        max_tokens=1000,
        api_key=config.openai_api_key
    )

    # Step 5: Create chain and invoke
    chain = prompt | llm
    response = await chain.ainvoke({
        "context": context,
        "question": query
    })

    answer = response.content or ""

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