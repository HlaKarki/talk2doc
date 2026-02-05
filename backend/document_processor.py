"""Document processing module for Talk2Doc.

Handles document loading, text splitting, embedding, and storage.
"""
import dotenv
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class DocumentProcessor:
    """Process and store documents for semantic search."""

    def __init__(
        self,
    ):
        """Initialize document processor.
        """