#!/bin/bash
# Start MediRAG FastAPI backend

# Activate virtual environment
source /home/chezy/rag_medical/venv/bin/activate

# Set environment variables
export PYTHONPATH="/home/chezy/rag_medical/src:$PYTHONPATH"
export RAG_DATA_DIR="/home/chezy/rag_medical/data"

# Start the server
cd /home/chezy/rag_medical/backend
python main.py
