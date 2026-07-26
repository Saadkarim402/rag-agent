#!/bin/bash
# Start the unified FastAPI backend and static frontend server on the exact port provided by the environment
python -m uvicorn app.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
