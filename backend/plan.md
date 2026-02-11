Project Iterations (Backend-First Approach)
Each iteration is self-contained with clear inputs, tasks, and verifiable outcomes. Focus on backend implementation first; frontend will come later.

Iteration 1: Database Foundation + Configuration
Duration: 1-2 days
Learning Focus: PostgreSQL with vector extensions, SQLAlchemy setup, environment configuration
What You'll Build

PostgreSQL database with pgvector extension
SQLAlchemy models for core tables
Database connection management
Environment configuration system

Files to Create

backend/core/config.py

Load environment variables using pydantic-settings
Database URL configuration
OpenAI API key configuration
Vector store settings (collection name, dimensions)


backend/models/database.py

SQLAlchemy base model
Document model (id, filename, file_type, upload_date, metadata, status)
DocumentChunk model (id, document_id, chunk_index, content, embedding vector, metadata)
Database engine and session management


backend/database/init.sql

Enable pgvector extension
Enable uuid-ossp extension
Create initial tables
Create vector indexes (ivfflat for fast similarity search)


backend/alembic.ini + backend/alembic/

Alembic configuration for database migrations
Initial migration creating all tables



Dependencies to Install
bashcd backend
uv add sqlalchemy asyncpg psycopg[binary] alembic pydantic-settings pgvector
Verification Steps

Create PostgreSQL database: createdb talk2doc
Run backend/database/init.sql to enable extensions
Run Alembic migration: alembic upgrade head
Connect to database and verify tables exist
Test inserting a document record with embedding vector
Query vector similarity: SELECT * FROM document_chunks ORDER BY embedding <-> '[0.1, 0.2, ...]' LIMIT 5

Outcome
✅ Database is ready with vector support
✅ SQLAlchemy models can interact with database
✅ Configuration system loads environment variables
✅ Migration system is set up for future schema changes

Iteration 2: Document Upload + Text Extraction
Duration: 1-2 days
Learning Focus: File handling, PDF/DOCX parsing, async FastAPI endpoints
What You'll Build

File upload endpoint accepting PDFs and DOCX files
Text extraction from uploaded documents
Document metadata storage
Basic file validation

Files to Create

backend/services/document_service.py

upload_document(file, user_id) - Save file, extract text, store metadata
get_documents(user_id) - List all documents for user
get_document_by_id(document_id) - Get specific document
delete_document(document_id) - Delete document and chunks
Use PyPDF for PDF extraction
Use docx2txt for Word extraction


backend/routers/documents.py

POST /api/documents/upload - Upload document endpoint
GET /api/documents - List documents endpoint
GET /api/documents/{id} - Get document details endpoint
DELETE /api/documents/{id} - Delete document endpoint


backend/main.py (modify)

Include documents router
Add CORS middleware
Add file upload size limits



Dependencies Already Installed

python-multipart (file uploads)
pypdf (PDF extraction)
docx2txt (Word extraction)

Verification Steps

Start FastAPI: uvicorn main:app --reload
Go to http://localhost:8000/docs (Swagger UI)
Test POST /api/documents/upload with a PDF file
Verify document record created in database
Test GET /api/documents returns the uploaded document
Test GET /api/documents/{id} returns document details
Test DELETE /api/documents/{id} removes document

Outcome
✅ Can upload PDF and DOCX files
✅ Text is extracted and stored
✅ Document metadata saved in database
✅ Can list, retrieve, and delete documents via API

Iteration 3: Text Chunking + Embeddings
Duration: 1-2 days
Learning Focus: LangChain text splitters, OpenAI embeddings, vector storage
What You'll Build

Text chunking using LangChain RecursiveCharacterTextSplitter
OpenAI embedding generation
Vector storage in PostgreSQL
Chunk metadata tracking

Files to Create/Modify

backend/services/embedding_service.py

chunk_text(text, chunk_size=1000, overlap=200) - Split text into chunks
generate_embeddings(texts) - Generate embeddings for list of texts
store_chunks(document_id, chunks, embeddings) - Store in database
Uses LangChain RecursiveCharacterTextSplitter
Uses OpenAI Embeddings API


backend/services/document_service.py (modify)

After text extraction, call embedding_service to chunk and embed
Store chunks with embeddings in document_chunks table
Update document status to "processed" when complete



Dependencies to Install
bashuv add langchain langchain-openai langchain-text-splitters openai
Environment Variables
bash# Add to backend/.env
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
Verification Steps

Upload a document via API
Check database: SELECT COUNT(*) FROM document_chunks WHERE document_id = '...'
Verify chunks have content and embeddings (1536 dimensions)
Check chunk metadata includes chunk_index and character positions
Test different chunk sizes (500, 1000, 2000) and observe differences

Outcome
✅ Documents are automatically chunked after upload
✅ Each chunk has vector embedding
✅ Chunks stored in database with metadata
✅ Can experiment with different chunking strategies

Iteration 4: Vector Similarity Search
Duration: 1 day
Learning Focus: Vector similarity, LangChain PGVector, retrieval strategies
What You'll Build

Initialize LangChain PGVector store
Semantic similarity search
MMR (Maximum Marginal Relevance) for diverse results
Metadata filtering

Files to Create

backend/services/vector_store.py

initialize_vector_store() - Create PGVector store connection
similarity_search(query, k=5, document_id=None) - Find similar chunks
mmr_search(query, k=5, fetch_k=20) - Diverse retrieval
search_with_score(query, k=5) - Return similarity scores
Uses LangChain PGVector integration


backend/routers/documents.py (modify)

POST /api/documents/search - Search across all documents
POST /api/documents/{id}/search - Search within specific document



Dependencies to Install
bashuv add langchain-postgres
```

### Verification Steps
1. Upload 2-3 documents on different topics
2. Test search: POST `/api/documents/search` with query "machine learning"
3. Verify returns relevant chunks with scores
4. Test document-specific search
5. Compare similarity search vs MMR search results
6. Test metadata filtering (e.g., only search document_id=X)

### Outcome
✅ Can search across all documents semantically
✅ Can search within specific documents
✅ Retrieval returns most relevant chunks
✅ Understand difference between similarity and MMR search

---

## Iteration 5: RAG Pipeline (Basic)
**Duration**: 2-3 days
**Learning Focus**: RAG fundamentals, prompt engineering, context assembly, response generation

### What You'll Build
- RAG query pipeline
- Context assembly from retrieved chunks
- LLM response generation with source citations
- Conversation context management

### Files to Create
1. **`backend/services/rag_service.py`**
   - `query_documents(query, document_id=None, conversation_history=[])` - Main RAG pipeline
   - Retrieve relevant chunks using vector_store
   - Assemble context with chunk content and metadata
   - Generate LLM response with OpenAI
   - Include source citations in response
   - Uses LangChain RetrievalQA or custom RAG chain

2. **`backend/routers/documents.py`** (modify)
   - POST `/api/documents/{id}/query` - Ask questions about document
   - POST `/api/documents/query` - Ask questions across all documents
   - Request body: `{query: str, conversation_history?: Message[]}`
   - Response: `{answer: str, sources: Source[]}`

3. **`backend/models/database.py`** (modify)
   - Add Conversation model (id, user_id, created_at, title)
   - Add Message model (id, conversation_id, role, content, timestamp, metadata)

### Prompt Template Example
```
You are a helpful AI assistant that answers questions based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain the answer, say "I don't have enough information"
- Cite your sources using [Source 1], [Source 2], etc.

Answer:
```

### Verification Steps
1. Upload a document (e.g., research paper PDF)
2. Ask question: "What is the main conclusion?"
3. Verify response is accurate and cites sources
4. Ask follow-up question with conversation history
5. Test question that document doesn't answer - should say "I don't have enough information"
6. Test asking same question with different chunk sizes - observe quality differences

### Outcome
✅ Can ask questions about uploaded documents
✅ Responses are grounded in document content
✅ Source citations link back to specific chunks
✅ Basic conversation context is maintained
✅ System handles out-of-scope questions gracefully

---

## Iteration 6: RAG Streaming + Advanced Features
**Duration**: 1-2 days
**Learning Focus**: Streaming responses, async patterns, advanced RAG techniques

### What You'll Build
- Streaming LLM responses (token-by-token)
- Re-ranking retrieved chunks for better quality
- Conversation history management
- Query preprocessing

### Files to Create/Modify
1. **`backend/services/rag_service.py`** (modify)
   - Add `query_documents_stream(query, ...)` - Async generator for streaming
   - Add `rerank_chunks(query, chunks)` - Re-rank by relevance
   - Add `preprocess_query(query)` - Query expansion, spell correction
   - Implement conversation summarization for long histories

2. **`backend/routers/documents.py`** (modify)
   - Add WebSocket `/ws/documents/{id}/query` - Streaming endpoint
   - Or use Server-Sent Events (SSE) if WebSocket is complex

### Advanced Techniques
- **Re-ranking**: After vector retrieval, use LLM to score chunks by relevance
- **Query expansion**: Expand query with synonyms or related terms
- **Conversation summarization**: Summarize long conversation history to fit context window
- **Hybrid search**: Combine vector search with keyword search (BM25)

### Verification Steps
1. Test streaming endpoint - tokens should appear progressively
2. Compare re-ranked results vs raw retrieval - should be more relevant
3. Test with long conversation history - should handle gracefully
4. Measure response latency - streaming should feel faster

### Outcome
✅ Responses stream token-by-token for better UX
✅ Retrieved chunks are re-ranked for higher quality
✅ Long conversations handled efficiently
✅ Queries are preprocessed for better retrieval

---

## Iteration 7: Knowledge Graph - Entity Extraction
**Duration**: 2-3 days
**Learning Focus**: LLM-based entity extraction, relationship detection, structured output

### What You'll Build
- Extract entities from document chunks using LLM
- Extract relationships between entities
- Store entities and relationships in database
- Build graph structure using NetworkX

### Files to Create
1. **`backend/models/database.py`** (modify)
   - Add KnowledgeGraphNode model (id, entity_name, entity_type, properties, embedding, source_document_id)
   - Add KnowledgeGraphEdge model (id, source_node_id, target_node_id, relationship_type, properties, confidence)

2. **`backend/services/entity_extraction_service.py`**
   - `extract_entities(text)` - Use LLM to extract entities
   - `extract_relationships(text, entities)` - Use LLM to find relationships
   - Uses structured output (JSON) from OpenAI
   - Confidence scoring for each entity/relationship

3. **`backend/services/knowledge_graph_service.py`**
   - `build_graph_from_document(document_id)` - Process all chunks
   - `store_graph(nodes, edges)` - Save to database
   - `merge_duplicate_entities(nodes)` - Entity resolution
   - Uses NetworkX for graph manipulation

4. **`backend/routers/knowledge_graph.py`**
   - POST `/api/kg/extract/{document_id}` - Extract graph from document
   - GET `/api/kg/documents/{id}/graph` - Get graph data
   - GET `/api/kg/nodes/{id}` - Get node details
   - GET `/api/kg/edges/{id}` - Get edge details

### LLM Prompt Example for Entity Extraction
```
Extract entities from the following text. Return a JSON array of entities.

Text: {text}

Format:
[
  {
    "name": "Entity Name",
    "type": "Person|Organization|Location|Concept|Date|Other",
    "confidence": 0.95
  }
]

Entities:
Dependencies to Install
bashuv add networkx
Verification Steps

Upload a document
Call POST /api/kg/extract/{document_id}
Check database: SELECT COUNT(*) FROM kg_nodes WHERE source_document_id = '...'
Verify entities make sense for document content
Check relationships are meaningful
Test entity merging - duplicate entities should be merged

Outcome
✅ Entities extracted from documents automatically
✅ Relationships detected between entities
✅ Graph stored in database
✅ Entities have types and confidence scores
✅ Duplicate entities are merged

Iteration 8: Knowledge Graph - Querying + Traversal
Duration: 1-2 days
Learning Focus: Graph algorithms, graph-based retrieval, graph memory
What You'll Build

Query knowledge graph using graph algorithms
Find shortest paths between entities
Get entity neighbors and subgraphs
Semantic search over graph entities

Files to Create/Modify

backend/services/knowledge_graph_service.py (modify)

get_entity_by_name(name) - Find entity in graph
get_entity_neighbors(entity_id, depth=1) - Get connected entities
find_path(entity_id_1, entity_id_2) - Shortest path between entities
get_subgraph(entity_ids) - Extract subgraph around entities
semantic_search_entities(query, k=10) - Search entities by embedding similarity


backend/services/graph_memory.py

Store document relationships as graph
Retrieve related documents via graph traversal
Concept clustering across documents


backend/routers/knowledge_graph.py (modify)

GET /api/kg/search?query=... - Semantic entity search
GET /api/kg/nodes/{id}/neighbors - Get neighbors
GET /api/kg/path?from={id1}&to={id2} - Find path
GET /api/kg/visualize/{document_id} - Graph data for visualization



Verification Steps

Extract graph from 2-3 related documents
Search for an entity: GET /api/kg/search?query=machine learning
Get neighbors of that entity
Find path between two entities
Test semantic search returns relevant entities even with different wording

Outcome
✅ Can query knowledge graph with various methods
✅ Can traverse graph to find related concepts
✅ Can search entities semantically
✅ Graph memory connects related documents
✅ Ready for graph visualization in frontend

Iteration 9: Multi-Agent System - Router + Base Agent
Duration: 2-3 days
Learning Focus: Agent design patterns, intent classification, LangGraph basics
What You'll Build

Base agent class with common functionality
Router agent for intent classification
LangGraph state management setup
Basic workflow with single agent

Files to Create

backend/agents/base_agent.py

BaseAgent class with __init__, invoke, stream methods
Memory access methods
Tool invocation interface
State management utilities


backend/agents/router_agent.py

Classify user intent: DOCUMENT_QUERY, DATA_ANALYSIS, GENERAL, MULTI_MODAL
Extract entities/context from query
Determine routing decision
Confidence scoring


backend/workflows/state.py

AgentState TypedDict definition
Fields: messages, query, intent, route_decision, context, results, metadata


backend/workflows/agent_workflow.py

Simple LangGraph workflow with START -> router -> END
State graph definition
Conditional edges based on intent


backend/services/agent_executor.py

execute_workflow(query, user_id) - Run LangGraph workflow
execute_workflow_stream(query, user_id) - Streaming execution
Error handling and retries



Dependencies to Install
bashuv add langgraph langchain-community
LangGraph Workflow Structure (Simple)
pythonfrom langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_edge(START, "router")
workflow.add_edge("router", END)
Verification Steps

Send query: "What does this document say about AI?"
Verify router classifies as DOCUMENT_QUERY
Check AgentState is populated correctly
Test various query types and verify correct classification
Monitor state transitions in logs

Outcome
✅ Router agent classifies user intent accurately
✅ LangGraph state management is working
✅ Basic workflow executes successfully
✅ Foundation for multi-agent system is ready

Iteration 10: Multi-Agent System - Document Agent
Duration: 2 days
Learning Focus: Specialized agents, tool integration, agent composition
What You'll Build

Document agent specializing in RAG queries
Knowledge graph traversal in agent
Source compilation and citation
Integration with existing RAG service

Files to Create

backend/agents/document_agent.py

DocumentAgent class extending BaseAgent
Tools: search_documents, get_document_chunks, search_knowledge_graph
Execute RAG query using rag_service
Traverse knowledge graph for related concepts
Compile sources and citations
Return structured response


backend/workflows/agent_workflow.py (modify)

Add document_node to workflow
Add conditional edge: router -> document_node (if DOCUMENT_QUERY)
Add edge: document_node -> END



LangGraph Workflow Structure (Updated)
pythonworkflow.add_node("router", router_node)
workflow.add_node("document", document_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_query,
    {
        "document": "document",
        "end": END
    }
)
workflow.add_edge("document", END)
Verification Steps

Ask document question: "What are the key findings in this paper?"
Verify router routes to document agent
Check document agent uses RAG service
Verify knowledge graph is queried for related concepts
Confirm response includes proper citations
Test with multiple documents - agent should search across all

Outcome
✅ Document agent handles RAG queries
✅ Router correctly routes document questions
✅ Knowledge graph enhances document retrieval
✅ Workflow executes: router -> document -> response

Iteration 11: Multi-Agent System - Data Scientist Agent (Placeholder)
Duration: 1 day
Learning Focus: Agent stubs, workflow branching
What You'll Build

Data scientist agent stub (placeholder for now)
Workflow routing to data agent
Response that explains data analysis will come later

Files to Create

backend/agents/data_scientist_agent.py

DataScientistAgent class extending BaseAgent
For now, returns placeholder message: "Data analysis features coming in next iterations"
Logs that data query was received


backend/workflows/agent_workflow.py (modify)

Add data_scientist_node to workflow
Add conditional edge: router -> data_scientist_node (if DATA_ANALYSIS)
Add edge: data_scientist_node -> END



Verification Steps

Ask data question: "What's the correlation between columns A and B?"
Verify router routes to data scientist agent
Check placeholder response is returned
Confirm workflow routing works for both document and data intents

Outcome
✅ Router can handle DATA_ANALYSIS intent
✅ Workflow routes to data scientist agent
✅ Foundation for data analysis features is ready
✅ Multi-agent routing is working

Iteration 12: Memory System - Short-Term Memory
Duration: 1-2 days
Learning Focus: Conversation buffers, context windows, memory management
What You'll Build

Short-term memory (conversation buffer)
Rolling window management
Context injection into agents
Message pruning strategies

Files to Create

backend/models/database.py (modify)

Add ShortTermMemory model (id, conversation_id, message_window, last_updated)


backend/services/memory/short_term_memory.py

get_conversation_buffer(conversation_id, window_size=10) - Get recent messages
add_message(conversation_id, message) - Add message to buffer
summarize_context(messages) - Summarize for token efficiency
prune_messages(conversation_id) - Remove old messages


backend/services/memory/memory_manager.py

MemoryManager class coordinating all memory types
get_context_for_query(query, conversation_id) - Assemble context from all memory layers
update_memory(conversation_id, query, response) - Update memories after interaction


backend/agents/base_agent.py (modify)

Inject memory context into agent prompts
Access MemoryManager in agent invocation



Verification Steps

Start a conversation with multiple turns
Check database: conversation_id and messages are stored
Ask follow-up question referencing previous message
Verify agent has context from conversation history
Test with 20+ message conversation - should maintain last 10

Outcome
✅ Conversation context is maintained
✅ Agents have access to recent messages
✅ Long conversations are handled with pruning
✅ Memory manager coordinates memory access

Iteration 13: Memory System - Long-Term + Semantic Memory
Duration: 2 days
Learning Focus: Persistent memory, semantic retrieval, memory importance
What You'll Build

Long-term memory for user preferences
Semantic memory for past interactions
Memory importance scoring
Memory consolidation

Files to Create

backend/models/database.py (modify)

Add LongTermMemory model (id, user_id, memory_type, key, value, created_at, accessed_count, last_accessed)
Add SemanticMemory model (id, user_id, query, response, embedding, context, created_at, relevance_score)


backend/services/memory/long_term_memory.py

store_preference(user_id, key, value) - Store user preference
get_preference(user_id, key) - Retrieve preference
store_insight(user_id, insight) - Store learned insight
get_important_memories(user_id, k=10) - Get by importance score
update_access_count(memory_id) - Track access frequency


backend/services/memory/semantic_memory.py

store_interaction(user_id, query, response, context) - Store and embed
search_similar_interactions(query, k=5) - Semantic search past interactions
get_relevant_memories(query, threshold=0.7) - Filter by relevance


backend/services/memory/memory_manager.py (modify)

Integrate long-term and semantic memory into context assembly
Implement memory prioritization logic
Add consolidate_memories() - Periodic consolidation task



Memory Importance Scoring
pythonimportance = (
    recency_score * 0.3 +
    access_frequency * 0.3 +
    relevance_to_query * 0.4
)
Verification Steps

Set a user preference (e.g., preferred analysis method)
In later conversation, verify agent uses that preference
Ask question similar to past query
Verify semantic memory retrieves similar interaction
Check that important memories rank higher
Test memory consolidation - duplicates should be merged

Outcome
✅ User preferences persist across sessions
✅ Semantic memory retrieves similar past interactions
✅ Memory importance scoring works
✅ All memory layers integrated into agents
✅ Memory inspector API ready for frontend

Iteration 14: Memory System - Graph Memory
Duration: 1-2 days
Learning Focus: Graph-based memory, concept relationships, memory networks
What You'll Build

Store conversation history as graph
Connect concepts across conversations
Graph-based memory retrieval
Memory network visualization data

Files to Create

backend/services/memory/graph_memory.py

add_memory_node(concept, embedding, conversation_id) - Add concept to graph
add_memory_edge(concept1, concept2, relationship) - Connect concepts
traverse_memory_graph(start_concept, depth=2) - Graph traversal
get_related_memories(concept) - Find connected memories
Integrate with knowledge graph for cross-document memory


backend/services/memory/memory_manager.py (modify)

Add graph memory to context assembly
Traverse memory graph for related concepts
Enrich agent context with graph connections


backend/routers/memory.py

GET /api/memory/short-term/{conversation_id} - Conversation buffer
GET /api/memory/long-term - User memories
GET /api/memory/semantic/search?query=... - Search semantic memory
GET /api/memory/graph/{concept} - Graph memory for concept
POST /api/memory/learn - Explicitly store memory
DELETE /api/memory/{id} - Forget memory



Verification Steps

Have conversation mentioning multiple concepts
Check graph memory has nodes for key concepts
Query related memories for a concept
Verify graph traversal finds connected memories
Test memory graph API endpoints

Outcome
✅ Conversation concepts stored as graph
✅ Related memories connected via relationships
✅ Graph traversal enhances memory retrieval
✅ All four memory layers complete (short-term, long-term, semantic, graph)
✅ Memory APIs ready for frontend integration

Iteration 15: Multi-Agent - Synthesizer Agent
Duration: 1-2 days
Learning Focus: Multi-source synthesis, insight aggregation, agent composition
What You'll Build

Synthesizer agent combining outputs from multiple agents
Multi-source insight aggregation
Conflict resolution between sources
Comprehensive response generation

Files to Create

backend/agents/synthesizer_agent.py

SynthesizerAgent class extending BaseAgent
synthesize(inputs) - Combine outputs from document and data agents
Identify common themes across sources
Resolve conflicts or contradictions
Generate comprehensive summary
Include confidence scoring


backend/workflows/agent_workflow.py (modify)

Add synthesizer_node to workflow
Route document_node and data_scientist_node to synthesizer
synthesizer_node -> END



LangGraph Workflow Structure (Updated)
pythonworkflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge("document", "synthesizer")
workflow.add_edge("data_scientist", "synthesizer")
workflow.add_edge("synthesizer", END)
Verification Steps

Ask question requiring document search
Verify document agent response goes to synthesizer
Check synthesizer enhances or summarizes response
Test with multi-modal query (when data features are built)
Verify synthesizer combines insights from multiple agents

Outcome
✅ Synthesizer agent aggregates multi-source results
✅ Workflow routes through synthesizer before END
✅ Comprehensive responses from multiple agents
✅ Foundation for multi-modal queries is ready

Iteration 16: Multi-Agent - Parallel Execution
Duration: 1-2 days
Learning Focus: LangGraph parallel execution, supervisor pattern, concurrency
What You'll Build

Parallel agent execution for multi-modal queries
Supervisor pattern for orchestration
Concurrent document and data agent execution
State merging from parallel branches

Files to Modify

backend/workflows/agent_workflow.py (modify)

Add supervisor_node for orchestration
Implement parallel execution for document + data agents
Merge state from parallel branches
Conditional routing based on intent (single agent or parallel)



LangGraph Workflow Structure (Final)
pythonworkflow.add_node("router", router_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("document", document_node)
workflow.add_node("data_scientist", data_scientist_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_query,
    {
        "document_only": "document",
        "data_only": "data_scientist",
        "multi_modal": "supervisor"
    }
)

# Supervisor routes to parallel execution
workflow.add_edge("supervisor", ["document", "data_scientist"])  # Parallel

workflow.add_edge("document", "synthesizer")
workflow.add_edge("data_scientist", "synthesizer")
workflow.add_edge("synthesizer", END)
Verification Steps

Send document-only query - verify single agent execution
Send multi-modal query (when data features exist)
Verify document and data agents execute in parallel
Check state merging works correctly
Monitor execution time - parallel should be faster
Verify supervisor coordinates execution

Outcome
✅ Agents can execute in parallel when needed
✅ Supervisor pattern orchestrates multi-agent workflows
✅ State merging from parallel branches works
✅ Multi-agent system is complete and optimized

Iteration 17: Data Upload + Profiling
Duration: 2-3 days
Learning Focus: Data handling, pandas, statistical analysis, schema inference
What You'll Build

CSV/Excel file upload
Data validation and cleaning
Automatic schema inference
Data profiling (statistics, distributions, missing values)

Files to Create

backend/models/database.py (modify)

Add Dataset model (id, filename, upload_date, user_id, row_count, column_count, schema, metadata)
Add AnalysisResult model (id, dataset_id, analysis_type, results, visualizations, created_at)


backend/services/data_service.py

upload_dataset(file, user_id) - Parse CSV/Excel, validate, store
get_datasets(user_id) - List datasets
get_dataset_by_id(dataset_id) - Get dataset details
delete_dataset(dataset_id) - Delete dataset
Uses pandas for reading CSV/Excel


backend/services/data_profiling_service.py

profile_dataset(dataset_id) - Generate comprehensive profile
Detect data types (numeric, categorical, datetime, text)
Calculate statistics (mean, median, std, min, max, quartiles)
Count missing values and duplicates
Calculate correlations between numeric columns
Identify potential issues (outliers, high cardinality, imbalance)


backend/routers/datasets.py

POST /api/datasets/upload - Upload dataset
GET /api/datasets - List datasets
GET /api/datasets/{id} - Get dataset details
GET /api/datasets/{id}/profile - Get data profiling
GET /api/datasets/{id}/preview?limit=100 - Preview rows
DELETE /api/datasets/{id} - Delete dataset



Dependencies to Install
bashuv add pandas openpyxl xlrd
Verification Steps

Upload CSV file via API
Check dataset record created in database
GET /api/datasets/{id}/profile - verify profiling data
Check column types are correctly detected
Verify statistics are accurate
Test with Excel file
Test with messy data (missing values, mixed types)

Outcome
✅ Can upload CSV and Excel files
✅ Data is validated and stored
✅ Automatic data profiling works
✅ Dataset APIs ready for analysis features
✅ Data quality issues are identified

Iteration 18: NLP Analysis on Text Data
Duration: 2-3 days
Learning Focus: NLP techniques, sentiment analysis, entity extraction, topic modeling
What You'll Build

Sentiment analysis on text columns
Named entity extraction
Keyword extraction
Topic modeling (LDA)
Text classification

Files to Create

backend/services/nlp_service.py

analyze_sentiment(texts) - Sentiment scores and labels
extract_entities(texts) - Named entity recognition
extract_keywords(texts, n=10) - Top keywords/phrases
topic_modeling(texts, n_topics=5) - LDA topic modeling
classify_text(texts, categories) - Zero-shot classification
Uses OpenAI or local models (TextBlob, spaCy)


backend/routers/datasets.py (modify)

POST /api/datasets/{id}/analyze/nlp - Run NLP analysis
Request body: {column: str, analysis_type: 'sentiment'|'entities'|'keywords'|'topics'}
Response: {results: any[], insights: str[]}



Dependencies to Install
bashuv add nltk spacy textblob gensim
NLP Pipeline

Text preprocessing (lowercase, remove stopwords, tokenization)
Apply selected NLP technique
Store results in AnalysisResult table
Generate insights summary

Verification Steps

Upload dataset with text column (e.g., product reviews)
Run sentiment analysis: POST /api/datasets/{id}/analyze/nlp
Verify sentiment scores are accurate
Test entity extraction - should find names, locations, organizations
Run topic modeling - topics should make sense
Test keyword extraction

Outcome
✅ Sentiment analysis on text columns works
✅ Entity extraction identifies key entities
✅ Topic modeling discovers themes in text
✅ Keyword extraction finds important terms
✅ NLP results stored for later retrieval

Iteration 19: Supervised Learning (Classification)
Duration: 2-3 days
Learning Focus: Scikit-learn, classification algorithms, model evaluation, hyperparameter tuning
What You'll Build

Classification pipeline (Random Forest, Logistic Regression, SVM)
Train/test split and cross-validation
Feature engineering and selection
Model evaluation (accuracy, precision, recall, F1, confusion matrix)
Hyperparameter tuning

Files to Create

backend/services/ml_service.py

train_classifier(dataset_id, target_column, feature_columns, algorithm='random_forest') - Train model
Preprocess data (handle missing, encode categorical, scale numeric)
Train/test split (80/20)
K-fold cross-validation
Calculate evaluation metrics
Save trained model (pickle or joblib)
Return performance report


backend/models/database.py (modify)

Add MLModel model (id, dataset_id, model_type, parameters, metrics, artifact_path, created_at)


backend/routers/datasets.py (modify)

POST /api/datasets/{id}/analyze/classify - Train classification model
Request body: {target_column: str, feature_columns: str[], algorithm: str, parameters?: any}
Response: {model_id: str, metrics: any, feature_importance: any[]}



Dependencies to Install
bashuv add scikit-learn joblib
```

### Classification Pipeline
1. Load dataset from database/storage
2. Preprocess features (encoding, scaling, imputation)
3. Split data into train/test
4. Train model with cross-validation
5. Evaluate on test set
6. Calculate feature importance
7. Save model artifact
8. Store metrics in database

### Verification Steps
1. Upload dataset with target variable (e.g., Titanic survival)
2. POST `/api/datasets/{id}/analyze/classify` with target and features
3. Verify model trains successfully
4. Check evaluation metrics are reasonable
5. Test feature importance - important features should make sense
6. Try different algorithms and compare performance

### Outcome
✅ Can train classification models
✅ Multiple algorithms supported
✅ Model evaluation metrics calculated
✅ Feature importance shows which features matter
✅ Trained models saved for later use

---

## Iteration 20: Unsupervised Learning (Clustering)
**Duration**: 2 days
**Learning Focus**: Clustering algorithms, dimensionality reduction, pattern discovery

### What You'll Build
- Clustering pipeline (K-means, DBSCAN, Hierarchical)
- Optimal cluster number detection (elbow method, silhouette score)
- Dimensionality reduction (PCA, t-SNE) for visualization
- Cluster profiling and interpretation

### Files to Create/Modify
1. **`backend/services/ml_service.py`** (modify)
   - `cluster_data(dataset_id, feature_columns, algorithm='kmeans', n_clusters=None)` - Perform clustering
   - Auto-detect optimal K for K-means (elbow method)
   - Apply dimensionality reduction for visualization
   - Calculate cluster statistics (size, centroids, characteristics)
   - Profile each cluster (feature means, distributions)
   - Assign cluster labels to data points

2. **`backend/routers/datasets.py`** (modify)
   - POST `/api/datasets/{id}/analyze/cluster` - Run clustering analysis
   - Request body: `{feature_columns: str[], algorithm: str, n_clusters?: int}`
   - Response: `{cluster_labels: int[], n_clusters: int, cluster_profiles: any[], visualization_data: any}`

### Verification Steps
1. Upload dataset suitable for clustering (e.g., customer segments)
2. POST `/api/datasets/{id}/analyze/cluster`
3. Verify clusters are found
4. Check silhouette score is reasonable (> 0.3)
5. Test PCA visualization data has 2D coordinates
6. Verify cluster profiles show distinct characteristics
7. Try different algorithms and compare results

### Outcome
✅ Can perform clustering on datasets
✅ Optimal cluster number detected automatically
✅ Dimensionality reduction for visualization
✅ Cluster profiles help interpret results
✅ Multiple clustering algorithms supported

---

## Iteration 21: Data Visualization Generation
**Duration**: 2 days
**Learning Focus**: Plotly, data visualization, chart selection, JSON serialization

### What You'll Build
- Generate Plotly visualizations from data
- Automatic chart type selection based on data
- Interactive visualizations (zoom, pan, hover)
- Visualization for ML results (clusters, feature importance)

### Files to Create
1. **`backend/services/visualization_service.py`**
   - `generate_distribution_plot(data, column)` - Histogram or box plot
   - `generate_scatter_plot(data, x_col, y_col, color_by=None)` - Scatter plot
   - `generate_correlation_matrix(data, columns)` - Heatmap
   - `generate_cluster_visualization(data, cluster_labels, coords)` - Cluster plot
   - `generate_feature_importance_chart(feature_names, importance_scores)` - Bar chart
   - `generate_time_series_plot(data, date_col, value_col)` - Line chart
   - Returns Plotly JSON for frontend rendering

2. **`backend/routers/datasets.py`** (modify)
   - GET `/api/datasets/{id}/visualize?type=distribution&column=age` - Get specific visualization
   - GET `/api/datasets/{id}/visualize/auto` - Auto-generate multiple visualizations

### Dependencies Already Installed
- `plotly`

### Verification Steps
1. Get dataset profile data
2. Request distribution plot for numeric column
3. Verify Plotly JSON is valid
4. Request correlation matrix
5. Get cluster visualization after clustering
6. Test auto-generate creates multiple relevant charts

### Outcome
✅ Visualizations generated automatically
✅ Plotly JSON ready for frontend rendering
✅ Multiple chart types supported
✅ ML results visualized (clusters, feature importance)
✅ Interactive visualizations with hover and zoom

---

## Iteration 22: Data Scientist Agent - Implementation
**Duration**: 2-3 days
**Learning Focus**: Agent tools, data analysis workflows, ML orchestration

### What You'll Build
- Complete data scientist agent implementation
- Tools for profiling, NLP, ML, visualization
- Intelligent analysis selection
- Result interpretation and insights

### Files to Modify
1. **`backend/agents/data_scientist_agent.py`** (complete implementation)
   - Tools: profile_dataset, run_nlp_analysis, train_classifier, perform_clustering, generate_visualizations
   - Intelligent tool selection based on user query
   - Multi-step analysis workflows
   - Interpret ML results and generate insights
   - Return structured response with visualizations

2. **`backend/workflows/agent_workflow.py`** (modify)
   - Update data_scientist_node to use fully implemented agent
   - Handle data analysis queries end-to-end

### Example Agent Workflow
```
User: "Analyze the sentiment in the reviews column and show me the distribution"

Data Scientist Agent:
1. Calls profile_dataset tool -> identifies 'reviews' is text column
2. Calls run_nlp_analysis(column='reviews', type='sentiment') -> sentiment scores
3. Calls generate_visualizations(type='distribution', data=sentiment_scores)
4. Interprets results: "The reviews are predominantly positive (68% positive, 22% neutral, 10% negative)"
5. Returns response with insights + visualization JSON
```

### Verification Steps
1. Upload dataset with text and numeric columns
2. Ask: "What insights can you find in this data?"
3. Verify agent profiles data first
4. Check agent selects appropriate analyses
5. Test specific requests: "Classify the target column"
6. Verify agent returns insights + visualizations
7. Test multi-step analysis: "Cluster the data and show me the characteristics of each cluster"

### Outcome
✅ Data scientist agent fully functional
✅ Can handle diverse data analysis queries
✅ Intelligent tool selection based on query
✅ Returns insights with visualizations
✅ Multi-step analysis workflows work
✅ Router correctly routes data queries to data agent

---

## Iteration 23: Multi-Modal Agent + Cross-Modal Queries
**Duration**: 2-3 days
**Learning Focus**: Cross-modal reasoning, document-data integration, comprehensive insights

### What You'll Build
- Multi-modal agent coordinating document and data analysis
- Cross-reference documents with datasets
- Combined insights from multiple sources
- Handle complex queries spanning documents and data

### Files to Create
1. **`backend/agents/multi_modal_agent.py`**
   - MultiModalAgent class extending BaseAgent
   - Coordinate document_agent and data_scientist_agent
   - Cross-reference findings between modalities
   - Identify patterns across documents and data
   - Generate comprehensive multi-source insights

2. **`backend/workflows/advanced_workflow.py`**
   - Enhanced workflow with multi-modal support
   - Conditional routing for single vs multi-modal queries
   - Memory integration at each step
   - Knowledge graph enrichment from data analysis

3. **`backend/services/insight_generator.py`**
   - `generate_cross_modal_insights(document_results, data_results)` - Synthesize insights
   - Pattern detection across modalities
   - Trend identification
   - Anomaly highlighting
   - Confidence scoring for insights

### Example Multi-Modal Query
```
User: "Compare the sentiment expressed in the research paper with the sentiment distribution in the customer reviews dataset"

Workflow:
1. Router classifies as MULTI_MODAL
2. Supervisor triggers parallel execution:
   - Document agent: Extract sentiment mentions from paper
   - Data scientist agent: Analyze sentiment in reviews dataset
3. Synthesizer combines results:
   - "The paper discusses predominantly positive outcomes (85% positive mentions),
     which aligns with customer reviews (72% positive sentiment).
     However, reviews show more nuance with 18% neutral sentiment not discussed in the paper."
Verification Steps

Upload document and dataset on related topics
Ask cross-modal query
Verify both document and data agents are invoked
Check synthesizer combines insights meaningfully
Test knowledge graph is enriched with data insights
Verify memory captures cross-modal patterns

Outcome
✅ Multi-modal agent handles complex queries
✅ Document and data insights are cross-referenced
✅ Comprehensive multi-source insights generated
✅ Knowledge graph enriched with data patterns
✅ Memory captures cross-modal relationships

Iteration 24: Streaming + WebSocket Support
Duration: 1-2 days
Learning Focus: Async Python, WebSocket, streaming patterns, real-time communication
What You'll Build

WebSocket endpoint for real-time chat
Streaming agent responses token-by-token
Progress updates during long operations
Connection management and error handling

Files to Create/Modify

backend/routers/chat.py

WebSocket /ws/chat - Real-time chat endpoint
Handle connection lifecycle (connect, disconnect)
Stream agent responses as they're generated
Send progress updates for long operations
Error handling and reconnection logic


backend/services/agent_executor.py (modify)

Enhance streaming support for all agents
Emit intermediate results during execution
Progress indicators for long operations (ML training, clustering)



WebSocket Message Protocol
json{
  "type": "message" | "progress" | "error" | "complete",
  "content": "...",
  "metadata": {
    "agent": "document",
    "step": "retrieval",
    "progress": 0.6
  }
}
Dependencies to Install
bashuv add websockets
Verification Steps

Connect to WebSocket endpoint
Send query and verify streaming response
Check progress updates during ML training
Test error handling - disconnect and reconnect
Verify multiple concurrent connections work
Monitor memory usage with many connections

Outcome
✅ Real-time chat via WebSocket
✅ Agent responses stream token-by-token
✅ Progress updates for long operations
✅ Connection management is robust
✅ Better UX with streaming responses

Iteration 25: Monitoring + Logging
Duration: 1 day
Learning Focus: Observability, performance monitoring, error tracking
What You'll Build

Agent performance metrics
Response time tracking
Error logging and alerting
Usage statistics
Performance dashboard data

Files to Create

backend/utils/monitoring.py

track_agent_execution(agent_name, query, duration, success) - Log execution
track_error(error, context) - Log errors with context
get_performance_metrics() - Aggregate metrics
get_usage_statistics(user_id, date_range) - Usage stats


backend/models/database.py (modify)

Add AgentExecution model (id, agent_name, query, duration, success, error_message, timestamp)
Add UsageStatistics model (id, user_id, action_type, timestamp, metadata)


backend/routers/monitoring.py

GET /api/monitoring/performance - Performance metrics
GET /api/monitoring/usage - Usage statistics
GET /api/monitoring/errors - Recent errors



Verification Steps

Execute various queries
Check AgentExecution records are created
GET /api/monitoring/performance - verify metrics
Trigger an error - verify error logging works
Check usage statistics are tracked

Outcome
✅ Agent performance is tracked
✅ Errors are logged with context
✅ Usage statistics available
✅ Monitoring APIs ready for dashboard
✅ Can identify performance bottlenecks

Summary: All Backend Iterations

Progress & Execution Order:
Note: Iteration 16 (Parallel Execution) deferred until after data features (17-22)
are implemented, since parallel execution is most useful with multiple functional agents.

Actual execution order: 1-15, 17-22, 16, 23-25

Completed Iterations:
✅ 1. Database Foundation + Configuration
✅ 2. Document Upload + Text Extraction
✅ 3. Text Chunking + Embeddings
✅ 4. Vector Similarity Search
✅ 5. RAG Pipeline (Basic)
✅ 6. RAG Streaming + Advanced Features
✅ 7. Knowledge Graph - Entity Extraction
✅ 8. Knowledge Graph - Querying + Traversal
✅ 9. Multi-Agent System - Router + Base Agent
✅ 10. Multi-Agent System - Document Agent
✅ 11. Multi-Agent System - Data Scientist Agent (Placeholder)
✅ 12. Memory System - Short-Term Memory
✅ 13. Memory System - Long-Term + Semantic Memory
✅ 14. Memory System - Graph Memory
✅ 15. Multi-Agent - Synthesizer Agent

Next Up (Data Features):
✅ 17. Data Upload + Profiling
✅ 18. NLP Analysis on Text Data (sentiment + keywords only)
✅ 19. Supervised Learning (Classification)
⬜ 20. Unsupervised Learning (Clustering)
⬜ 21. Data Visualization Generation
⬜ 22. Data Scientist Agent - Full Implementation

Deferred (After Data Features):
⬜ 16. Multi-Agent - Parallel Execution

Remaining:
⬜ 23. Multi-Modal Agent + Cross-Modal Queries
⬜ 24. Streaming + WebSocket Support
⬜ 25. Monitoring + Logging

What You'll Have After All Iterations

✅ Full RAG system with document Q&A
✅ Knowledge graph extraction and querying
✅ Multi-agent system with LangGraph orchestration
✅ All memory layers (short-term, long-term, semantic, graph)
✅ Data analysis with NLP, ML, visualizations
✅ Multi-modal agent handling documents + data
✅ Streaming responses and real-time communication
✅ Monitoring and performance tracking
✅ Complete backend API ready for frontend

Frontend Development (After Backend Complete)
Once all backend iterations are complete, you can build the frontend:

Dashboard with document/dataset library
Chat interface with streaming
Knowledge graph visualization
Memory inspector
Data visualization panels
Analysis configuration UI