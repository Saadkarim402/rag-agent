import os
import time
import requests
from typing import List, Optional
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="Antigravity RAG Agent Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")

# --- Custom Premium CSS Injector ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Apply font to the entire app */
html, body, [class*="css"], .stMarkdown, p {
    font-family: 'Outfit', sans-serif !important;
}

/* Gradient Header */
.main-header {
    background: linear-gradient(135deg, #818cf8, #3b82f6, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
}

.sub-header {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Glassmorphism sidebar elements */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Document List Card Container */
.doc-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0.85rem;
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.25s ease;
}
.doc-card:hover {
    background: rgba(51, 65, 85, 0.4);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-1px);
}
.doc-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 170px;
}
.doc-source {
    font-size: 0.75rem;
    color: #64748b;
}

/* Source citations card */
.source-card {
    background: rgba(30, 41, 59, 0.35);
    border-left: 4px solid #3b82f6;
    padding: 0.85rem 1.1rem;
    border-radius: 4px 10px 10px 4px;
    margin-bottom: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    border-right: 1px solid rgba(255, 255, 255, 0.03);
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}
.source-meta {
    font-size: 0.8rem;
    color: #34d399;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.source-score {
    float: right;
    font-weight: 800;
    color: #60a5fa;
}
.source-text {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.45;
}
</style>
""", unsafe_allow_html=True)


# --- REST API Service Client Layer ---

def is_backend_online() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def get_documents_list() -> List[dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

def delete_document_by_id(doc_id: str) -> bool:
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def upload_document_file(file_bytes: bytes, filename: str, doc_id: Optional[str] = None) -> bool:
    try:
        files = {"file": (filename, file_bytes, "text/plain")}
        data = {}
        if doc_id:
            data["doc_id"] = doc_id
        response = requests.post(f"{API_BASE_URL}/ingest/file", files=files, data=data, timeout=15)
        return response.status_code == 201
    except Exception:
        return False

def query_rag_engine(query: str, collection: str, top_k: int, threshold: float) -> Optional[dict]:
    try:
        payload = {
            "query": query,
            "collection_name": collection,
            "top_k": top_k,
            "min_score_threshold": threshold
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Communication error with the server: {e}")
    return None


# --- Main Application Logic ---

def main():
    st.markdown('<div class="main-header">🌌 ANTIGRAVITY RAG AGENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Local Offline Knowledge Augmentation Engine</div>', unsafe_allow_html=True)

    # Check backend connectivity
    backend_active = is_backend_online()

    if not backend_active:
        st.error("⚠️ REST API Server is Offline!")
        st.info(
            "Please start the FastAPI backend server on port 8000 using your terminal before chatting:\n\n"
            "```powershell\n"
            "$env:PYTHONPATH=\".\"\n"
            ".\\venv\\Scripts\\python -m uvicorn app.api.server:app --reload --port 8000\n"
            "```"
        )
        return

    # Initialize chat session history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR: Controls & Ingestion ---
    with st.sidebar:
        st.subheader("⚙️ Config Parameters")
        collection = st.text_input("Active Collection", value="documents", help="Vector DB collection name to search.")
        top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=4)
        threshold = st.slider("Min Relevance Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="Filter out chunks below this similarity score.")

        st.markdown("---")
        st.subheader("📥 Upload & Ingestion")
        
        custom_id = st.text_input("Document ID Override", placeholder="e.g. guide_v1 (optional)")
        uploaded_file = st.file_uploader("Choose a text/markdown file", type=["txt", "md"])

        if uploaded_file is not None:
            if st.button("🚀 Ingest Document"):
                with st.spinner("Analyzing and embedding document..."):
                    file_bytes = uploaded_file.read()
                    success = upload_document_file(file_bytes, uploaded_file.name, doc_id=custom_id)
                    if success:
                        st.success("✓ Ingested and stored document!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to ingest document.")

        st.markdown("---")
        st.subheader("📂 Ingested Documents")
        
        # Load and render original documents list
        docs = get_documents_list()
        if not docs:
            st.info("No documents indexed yet.")
        else:
            for d in docs:
                doc_id = d["doc_id"]
                title = d.get("title") or doc_id
                
                # Render document details and delete option
                col_text, col_btn = st.columns([0.75, 0.25])
                with col_text:
                    st.markdown(
                        f"<div class='doc-card'>"
                        f"  <div>"
                        f"    <div class='doc-title' title='{title}'>{title}</div>"
                        f"    <div class='doc-source'>ID: {doc_id}</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_btn:
                    # Render delete button aligned with list item
                    if st.button("🗑️", key=f"del_{doc_id}", help=f"Delete {doc_id}"):
                        if delete_document_by_id(doc_id):
                            st.success(f"Deleted {doc_id}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc_id}.")

    # --- MAIN CHAT INTERFACE ---
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "source_nodes" in msg:
                # Expandable details underneath response
                with st.expander("🔍 View Retrieved Context & Similarity Scores"):
                    for i, source in enumerate(msg["source_nodes"], 1):
                        meta = source.get("metadata") or {}
                        source_id = meta.get("source_id") or meta.get("source") or "unknown"
                        idx = meta.get("chunk_index", 0)
                        score = source.get("score")
                        score_text = f"{score:.4f}" if score is not None else "N/A"
                        
                        st.markdown(
                            f"<div class='source-card'>"
                            f"  <div class='source-meta'>"
                            f"    <span class='source-score'>Score: {score_text}</span>"
                            f"    [{i}] (Source: {source_id} | Chunk: {idx})"
                            f"  </div>"
                            f"  <div class='source-text'>\"{source['document_text']}\"</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                if "prompt" in msg:
                    with st.expander("⚙️ View Compiled Prompt Template"):
                        st.code(msg["prompt"], language="text")

    # Chat input and execution
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate agent answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating answer..."):
                response = query_rag_engine(prompt, collection, top_k, threshold)
                
                if response:
                    answer = response.get("answer", "I cannot answer this based on the provided context.")
                    st.write(answer)
                    
                    # Store response details in state
                    msg_data = {
                        "role": "assistant",
                        "content": answer,
                        "source_nodes": response.get("source_nodes", []),
                        "prompt": response.get("prompt", "")
                    }
                    st.session_state.messages.append(msg_data)
                    st.rerun()
                else:
                    st.error("Failed to generate answer from RAG server.")

if __name__ == "__main__":
    main()
