---
title: RAG Agent
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RAG Agent

Small Python workspace for building a retrieval-augmented generation agent.

## Current State

- `test_embedding.py` loads `BAAI/bge-small-en-v1.5` with Sentence Transformers.
- The embedding smoke test returns a 384-dimensional vector for `"hello world"`.
- The local environment already has RAG-oriented packages installed, including Sentence Transformers, ChromaDB, Transformers, Torch, and Uvicorn.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Smoke Test

```powershell
python test_embedding.py
```

Expected output starts with:

```text
384
```

## Next Steps

1. Initialize Git once Git is installed locally.
2. Add document loading and chunking.
3. Store embeddings in ChromaDB.
4. Add retrieval over stored chunks.
5. Add an LLM answer-generation layer.
