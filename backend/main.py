from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import documents, knowledge_graph, chat, conversations, memory, datasets, models
from database.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Initialize database
    await init_db()
    print("✅ Database initialized")

    yield

    # Shutdown: cleanup if needed
    print("👋 Shutting down...")


app = FastAPI(
    title="Talk2Doc API",
    description="Document processing and Q&A API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assign routers
app.include_router(documents.router)
app.include_router(knowledge_graph.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(memory.router)
app.include_router(datasets.router)
app.include_router(models.router)


@app.get("/")
async def root():
    return {"message": "Talk2Doc API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
