import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.routes import router
from app.documents.repository import DocumentRepository
from app.ingestion.ingest import DocumentIngestionManager
from app.llm import get_llm_client
from app.llm.chains import RAGChain
from app.retrieval.retriever import RetrievalManager
from app.vectordb.chroma_client import ChromaDBManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI, initializing RAG components."""
    logger.info("Initializing RAG Agent resources...")
    
    # 1. Initialize DB and Repositories
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    repo_dir = os.getenv("DOCUMENT_REPO_DIR", "data/documents")
    collection_name = os.getenv("DEFAULT_COLLECTION_NAME", "documents")

    chroma = ChromaDBManager(persist_directory=persist_dir)
    repository = DocumentRepository(repo_dir=repo_dir)

    # 2. Initialize Ingestion and Retrieval Managers
    ingest_manager = DocumentIngestionManager(
        chroma=chroma,
        document_repository=repository,
        collection_name=collection_name,
    )
    
    retriever = RetrievalManager(
        chroma=chroma,
        collection_name=collection_name,
    )

    # 3. Initialize LLM Client
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3")
    
    # Check for Gemini credentials if provider is Gemini
    client_kwargs = {}
    if provider.lower() == "gemini":
        client_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")
    if model:
        client_kwargs["model"] = model

    llm_client = get_llm_client(provider, **client_kwargs)

    # 4. Initialize RAG Chain
    rag_chain = RAGChain(
        retriever=retriever,
        llm_client=llm_client,
        default_collection_name=collection_name,
    )

    # 5. Share instances via FastAPI state
    app.state.chroma = chroma
    app.state.document_repository = repository
    app.state.ingest_manager = ingest_manager
    app.state.retriever = retriever
    app.state.llm_client = llm_client
    app.state.rag_chain = rag_chain

    logger.info("RAG Agent resources successfully initialized.")
    yield
    logger.info("Shutting down RAG Agent server...")


app = FastAPI(
    title="RAG Agent REST API Server",
    description="REST API interface exposing ingestion, retrieval, and LLM query chains for the RAG Agent.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up CORS middleware to allow requests from Streamlit frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes router
app.include_router(router)
