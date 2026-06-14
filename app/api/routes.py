from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

router = APIRouter()

# --- Request / Response Schemas ---

class QueryRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    query: str = Field(..., min_length=1, description="The user query string.")
    collection_name: Optional[str] = Field(None, description="Vector DB collection name.")
    top_k: Optional[int] = Field(5, gt=0, description="Number of results to retrieve.")
    min_score_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score.")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply.")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Optional conversation history list.")

class RAGSourceNode(BaseModel):
    model_config = {"protected_namespaces": ()}
    chunk_id: str
    document_text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None
    score: Optional[float] = None
    collection: str

class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    answer: str
    source_nodes: List[RAGSourceNode]
    prompt: str
    web_search_triggered: bool = False
    latency_ms: float = 0.0
    faithfulness: int = 5
    answer_relevance: int = 5
    agent_loop_logs: List[str] = []

class IngestTextRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    doc_id: str = Field(..., min_length=1, description="Unique document ID.")
    text: str = Field(..., min_length=1, description="Full text content.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata dictionary.")

class IngestResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    doc_id: str
    chunk_ids: List[str]

class DocumentMetadataResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    doc_id: str
    title: str
    source: str
    created_at: str

# --- Dependency Resolvers ---

def get_rag_chain(request: Request):
    return request.app.state.rag_chain

def get_ingest_manager(request: Request):
    return request.app.state.ingest_manager

def get_document_repository(request: Request):
    return request.app.state.document_repository

# --- Route Handlers ---

@router.post("/chat", response_model=QueryResponse, status_code=status.HTTP_200_OK)
@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_rag_chain(
    req: QueryRequest,
    rag_chain=Depends(get_rag_chain)
):
    """Query the RAG Chain with context-augmented prompt assembly and LLM response generation."""
    try:
        response = rag_chain.query(
            query=req.query,
            collection_name=req.collection_name,
            top_k=req.top_k,
            min_score_threshold=req.min_score_threshold,
            metadata_filter=req.metadata_filter,
            chat_history=req.chat_history,
        )
        return QueryResponse(
            answer=response.answer,
            source_nodes=[
                RAGSourceNode(
                    chunk_id=r.chunk_id,
                    document_text=r.document_text,
                    metadata=r.metadata,
                    distance=r.distance,
                    score=r.score,
                    collection=r.collection,
                )
                for r in response.source_nodes
            ],
            prompt=response.prompt,
            web_search_triggered=response.web_search_triggered,
            latency_ms=response.latency_ms,
            faithfulness=response.faithfulness,
            answer_relevance=response.answer_relevance,
            agent_loop_logs=response.agent_loop_logs,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {e}",
        )

@router.post("/ingest/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_raw_text(
    req: IngestTextRequest,
    ingest_manager=Depends(get_ingest_manager)
):
    """Ingest a single raw text document directly into the system."""
    try:
        chunk_ids = ingest_manager.ingest_text(
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata,
        )
        return IngestResponse(doc_id=req.doc_id, chunk_ids=chunk_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text ingestion failed: {e}",
        )

@router.post("/ingest/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    ingest_manager=Depends(get_ingest_manager)
):
    """Upload a file (e.g. text/markdown) and ingest it into the vector store."""
    try:
        content_bytes = await file.read()
        content_text = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {e}",
        )

    resolved_doc_id = doc_id or file.filename or "uploaded_file"
    metadata = {
        "title": file.filename or "",
        "source": file.filename or "file_upload",
    }

    try:
        chunk_ids = ingest_manager.ingest_text(
            doc_id=resolved_doc_id,
            text=content_text,
            metadata=metadata,
        )
        return IngestResponse(doc_id=resolved_doc_id, chunk_ids=chunk_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File ingestion failed: {e}",
        )

@router.get("/documents", response_model=List[DocumentMetadataResponse], status_code=status.HTTP_200_OK)
async def list_documents(
    doc_repo=Depends(get_document_repository)
):
    """Retrieve lists of all ingested original documents with basic metadata."""
    try:
        docs = doc_repo.list_documents()
        return [
            DocumentMetadataResponse(
                doc_id=d["doc_id"],
                title=d.get("title") or "",
                source=d.get("source") or "",
                created_at=d.get("created_at") or "",
            )
            for d in docs
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {e}",
        )

@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    doc_id: str,
    collection_name: Optional[str] = None,
    ingest_manager=Depends(get_ingest_manager)
):
    """Delete a document by its ID from the file store and vector store."""
    # Check if doc exists first
    if not ingest_manager.document_repository.exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    
    try:
        deleted = ingest_manager.delete_document(doc_id, collection_name=collection_name)
        if deleted:
            return {"detail": f"Document '{doc_id}' deleted successfully."}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document '{doc_id}'.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {e}",
        )
