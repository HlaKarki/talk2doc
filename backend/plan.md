# Talk2Doc - Backend Development Plan

A document Q&A system with RAG, knowledge graphs, multi-agent orchestration, and data analysis capabilities.

---

## Completed Iterations

### Core Infrastructure (1-6)
- [x] **1. Database Foundation** - PostgreSQL + pgvector, SQLAlchemy models, Alembic migrations
- [x] **2. Document Upload** - PDF/DOCX parsing, text extraction, file validation
- [x] **3. Text Chunking + Embeddings** - LangChain splitters, OpenAI embeddings, vector storage
- [x] **4. Vector Similarity Search** - PGVector integration, MMR search, metadata filtering
- [x] **5. RAG Pipeline** - Query pipeline, context assembly, source citations
- [x] **6. RAG Streaming** - Token streaming, re-ranking, conversation summarization

### Knowledge Graph (7-8)
- [x] **7. Entity Extraction** - LLM-based extraction, relationship detection, NetworkX graphs
- [x] **8. Graph Querying** - Traversal, shortest paths, semantic entity search

### Multi-Agent System (9-11, 15)
- [x] **9. Router + Base Agent** - Intent classification, LangGraph state management
- [x] **10. Document Agent** - RAG queries, knowledge graph integration
- [x] **11. Data Scientist Agent (Placeholder)** - Workflow routing setup
- [x] **15. Synthesizer Agent** - Multi-source aggregation, conflict resolution

### Memory System (12-14)
- [x] **12. Short-Term Memory** - Conversation buffers, rolling windows
- [x] **13. Long-Term + Semantic Memory** - User preferences, semantic retrieval
- [x] **14. Graph Memory** - Concept relationships, memory networks

### Data Analysis (17-22)
- [x] **17. Data Upload + Profiling** - CSV/Excel upload, statistics, schema inference
- [x] **18. NLP Analysis** - Sentiment analysis (TextBlob), keyword extraction (TF-IDF)
- [x] **19. Classification** - Random Forest, Logistic Regression, SVM with evaluation metrics
- [x] **20. Clustering** - K-Means, DBSCAN, Hierarchical with auto-K detection
- [x] **21. Data Visualization** - Plotly charts (histograms, scatter, correlation, time series)
- [x] **22. Data Scientist Agent** - Full implementation with LLM-powered tool selection

---

## Current Capabilities

### Document Processing
- Upload PDF/DOCX files
- Automatic chunking and embedding
- Semantic search across documents
- RAG-based Q&A with citations

### Knowledge Graph
- Entity and relationship extraction
- Graph traversal and querying
- Semantic entity search

### Data Analysis
- Dataset profiling and statistics
- Sentiment analysis and keyword extraction
- Classification and clustering models
- Auto-generated visualizations

### Multi-Agent System
- Intent-based routing (document, data, general)
- Specialized agents for each domain
- Response synthesis and enhancement
- Multi-layer memory system

---

## Future Plans

### 16. Parallel Agent Execution
- Concurrent document + data agent execution
- Supervisor pattern for orchestration
- State merging from parallel branches

### 23. Multi-Modal Agent
- Cross-reference documents with datasets
- Combined insights from multiple sources
- Pattern detection across modalities

### 24. Streaming + WebSocket
- Real-time chat via WebSocket
- Token-by-token streaming
- Progress updates for long operations

### 25. Monitoring + Logging
- Agent performance metrics
- Error tracking and alerting
- Usage statistics dashboard

---

## Tech Stack

- **Framework**: FastAPI (async)
- **Database**: PostgreSQL + pgvector
- **ORM**: SQLAlchemy (async)
- **LLM**: OpenAI (GPT-4o-mini)
- **Embeddings**: OpenAI text-embedding-3-small
- **Orchestration**: LangGraph
- **ML**: scikit-learn
- **NLP**: TextBlob, TF-IDF
- **Visualization**: Plotly
- **Storage**: Cloudflare R2

---

## API Endpoints

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents` - List documents
- `POST /documents/search` - Semantic search
- `POST /documents/{id}/query` - RAG query

### Datasets
- `POST /datasets/upload` - Upload CSV/Excel
- `GET /datasets/{id}/profile` - Get statistics
- `POST /datasets/{id}/analyze/sentiment` - Sentiment analysis
- `POST /datasets/{id}/analyze/keywords` - Keyword extraction
- `POST /datasets/{id}/visualize/*` - Generate charts

### Models
- `POST /models/train/{dataset_id}` - Train classifier
- `POST /models/cluster/{dataset_id}` - Run clustering
- `POST /models/{id}/predict` - Make predictions

### Knowledge Graph
- `POST /kg/extract/{document_id}` - Extract entities
- `GET /kg/search` - Search entities
- `GET /kg/nodes/{id}/neighbors` - Get relationships

### Chat
- `POST /chat/` - Send message to agents

### Memory
- `GET /memory/short-term/{conversation_id}` - Conversation buffer
- `GET /memory/semantic/search` - Search past interactions
