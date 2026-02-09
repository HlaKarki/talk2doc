import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from core.config import config
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Document(Base):
    """Document model for storing uploaded files"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    doc_metadata = Column(JSON, default=dict)
    status = Column(String(50), default="pending", nullable=False) # pending, processing, processed, failed

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document id={self.id} filename={self.filename}>"


class DocumentChunk(Base):
    """Document chunk model for storing text chunks with embeddings."""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(config.embedding_dimensions))
    chunk_metadata = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk id={self.id} document_id={self.document_id} chunk_index={self.chunk_index}>"


class KnowledgeGraphNode(Base):
    """Node in the knowledge graph representing an entity."""
    __tablename__ = "kg_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_name = Column(String(255), nullable=False)
    entity_type = Column(String(100), nullable=False)  # Person, Organization, Concept, Location, etc.
    properties = Column(JSON, default=dict)
    embedding = Column(Vector(config.embedding_dimensions))  # For semantic search over entities
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    source_chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document")
    chunk = relationship("DocumentChunk")

    # Edges where this node is the source
    outgoing_edges = relationship("KnowledgeGraphEdge", foreign_keys="KnowledgeGraphEdge.source_node_id", back_populates="source_node", cascade="all, delete-orphan")
    # Edges where this node is the target
    incoming_edges = relationship("KnowledgeGraphEdge", foreign_keys="KnowledgeGraphEdge.target_node_id", back_populates="target_node", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KGNode id={self.id} name={self.entity_name} type={self.entity_type}>"


class KnowledgeGraphEdge(Base):
    """Edge in the knowledge graph representing a relationship between entities."""
    __tablename__ = "kg_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(100), nullable=False)  # e.g., "works_at", "located_in", "related_to"
    properties = Column(JSON, default=dict)
    confidence = Column(Float, default=1.0)  # Confidence score for the relationship
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    source_node = relationship("KnowledgeGraphNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("KnowledgeGraphNode", foreign_keys=[target_node_id], back_populates="incoming_edges")
    document = relationship("Document")

    def __repr__(self):
        return f"<KGEdge id={self.id} {self.source_node_id} --[{self.relationship_type}]--> {self.target_node_id}>"


# ============================================================================
# Memory System Models
# ============================================================================

class Conversation(Base):
    """Conversation model for tracking chat sessions."""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)  # Auto-generated or user-set title
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    conv_metadata = Column(JSON, default=dict)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation id={self.id} title={self.title}>"


class Message(Base):
    """Message model for storing conversation messages."""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    msg_metadata = Column(JSON, default=dict)  # Store sources, agent used, etc.

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message id={self.id} role={self.role} conversation_id={self.conversation_id}>"


class ConversationSummary(Base):
    """Summarized context for long conversations to maintain within token limits."""
    __tablename__ = "conversation_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, unique=True)
    summary = Column(Text, nullable=False)  # Condensed summary of older messages
    messages_summarized = Column(Integer, default=0)  # Number of messages included in summary
    last_message_id = Column(UUID(as_uuid=True), nullable=True)  # Last message included in summary
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation")

    def __repr__(self):
        return f"<ConversationSummary id={self.id} conversation_id={self.conversation_id}>"

