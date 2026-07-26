#!/bin/bash
# Start the FastAPI backend server in the background (bind to localhost only for security)
python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 &

# Start the Streamlit UI frontend on port 7860 (Hugging Face default exposed port)
python -m streamlit run app/ui/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --browser.gatherUsageStats false
