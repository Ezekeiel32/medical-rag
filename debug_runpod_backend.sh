#!/bin/bash

echo "=== RunPod Backend Debug Script ==="
echo ""
echo "Run this on your RunPod to debug why the backend won't start"
echo ""

# Check the log file for errors
echo "1. Checking uvicorn.log for errors..."
if [ -f /workspace/uvicorn.log ]; then
    echo "Last 30 lines of uvicorn.log:"
    tail -30 /workspace/uvicorn.log
else
    echo "No log file found"
fi

echo ""
echo "2. Checking if port 8000 is already in use..."
lsof -i :8000

echo ""
echo "3. Killing any existing processes..."
pkill -f 'uvicorn main:app'
sleep 2

echo ""
echo "4. Testing Python imports directly..."
cd /workspace/rag_medical_gpu/backend
python -c "
import sys
try:
    import main
    print('✓ main.py imports OK')
    app = main.app
    print('✓ FastAPI app created OK')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "5. Starting uvicorn with verbose output (no background)..."
echo "This will show the actual error:"
echo ""
cd /workspace/rag_medical_gpu/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug 2>&1 | head -50