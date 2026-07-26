#!/bin/bash
# Start the FastAPI backend server in the background (force port 8000 to avoid conflicting with Render's public PORT)
PORT=8000 python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 &

# Start the Streamlit UI frontend on the environment port (Render) or 7860 (Hugging Face)
python -m streamlit run app/ui/streamlit_app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --browser.gatherUsageStats false
