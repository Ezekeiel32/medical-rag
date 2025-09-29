#!/bin/bash

# IMMEDIATE FIX SCRIPT - Copy and paste this directly into your RunPod SSH session
# This script will diagnose and fix the uvicorn startup issue

echo "=== RunPod Backend Immediate Fix ==="
echo "Starting diagnostic and fix process..."

# Step 1: Ensure we're in the medrag environment
echo ""
echo "Step 1: Checking environment..."
if [[ "$CONDA_DEFAULT_ENV" != "medrag" ]]; then
    echo "Activating medrag environment..."
    micromamba activate medrag
fi
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Step 2: Find and navigate to backend directory
echo ""
echo "Step 2: Finding backend directory..."
if [ -f /workspace/rag_medical_gpu/backend/main.py ]; then
    cd /workspace/rag_medical_gpu/backend
    echo "Found backend at: /workspace/rag_medical_gpu/backend"
elif [ -f /workspace/rag_medical/backend/main.py ]; then
    cd /workspace/rag_medical/backend
    echo "Found backend at: /workspace/rag_medical/backend"
else
    echo "ERROR: Cannot find main.py in expected locations!"
    echo "Searching for main.py..."
    find /workspace -name "main.py" -path "*/backend/*" 2>/dev/null
    exit 1
fi

# Step 3: Kill any existing processes
echo ""
echo "Step 3: Cleaning up existing processes..."
pkill -f 'uvicorn main:app' 2>/dev/null && echo "Killed existing uvicorn processes"
lsof -Pi :8000 -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null && echo "Freed port 8000"
sleep 2

# Step 4: Check critical dependencies
echo ""
echo "Step 4: Checking dependencies..."
python -c "
import sys
missing = []
try:
    import fastapi
    print('✓ FastAPI installed')
except ImportError:
    missing.append('fastapi')
    print('✗ FastAPI missing')
try:
    import uvicorn
    print('✓ Uvicorn installed')
except ImportError:
    missing.append('uvicorn')
    print('✗ Uvicorn missing')
try:
    import pydantic
    print('✓ Pydantic installed')
except ImportError:
    missing.append('pydantic')
    print('✗ Pydantic missing')

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Installing missing packages..."
    pip install uvicorn[standard] fastapi pydantic httpx python-multipart python-dotenv
fi

# Step 5: Set environment variables for GPU safety
echo ""
echo "Step 5: Setting environment variables for GPU optimization..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Step 6: Set Python path
echo ""
echo "Step 6: Setting Python path..."
BACKEND_DIR=$(pwd)
PARENT_DIR=$(dirname $BACKEND_DIR)
export PYTHONPATH="$BACKEND_DIR:$PARENT_DIR:$PARENT_DIR/src:$PYTHONPATH"
echo "PYTHONPATH configured"

# Step 7: Test main.py import
echo ""
echo "Step 7: Testing main.py imports..."
python -c "
import sys
import os

try:
    # Test if we can import main
    import main
    print('✓ main.py imports successfully')
    
    # Test if FastAPI app is accessible
    from main import app
    print('✓ FastAPI app object accessible')
    
except ImportError as e:
    print(f'✗ Import error: {e}')
    print('')
    print('Attempting to diagnose the issue...')
    
    # Check for common missing dependencies
    missing_deps = []
    try:
        import chromadb
    except ImportError:
        missing_deps.append('chromadb')
        print('  Missing: chromadb')
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append('sentence-transformers')
        print('  Missing: sentence-transformers')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
        print('  Missing: pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
        print('  Missing: numpy')
    
    if missing_deps:
        print('')
        print('To install missing dependencies, run:')
        print(f\"  pip install {' '.join(missing_deps)}\")
    
    sys.exit(1)
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Failed to import main.py. Attempting to install common dependencies..."
    pip install chromadb sentence-transformers pandas numpy python-multipart
    
    # Try import again
    python -c "import main; print('✓ Import successful after dependency installation')" 2>&1
    if [ $? -ne 0 ]; then
        echo "Still failing. Please check the error messages above."
        exit 1
    fi
fi

# Step 8: Start uvicorn
echo ""
echo "Step 8: Starting uvicorn backend..."

# Clear old log
> /workspace/uvicorn.log

# First, try to run in foreground briefly to see any immediate errors
echo "Testing startup (5 seconds)..."
timeout 5 uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info 2>&1 | tee /tmp/test_startup.log

# Check if there were any critical errors
if grep -q "ERROR\|Exception\|Failed" /tmp/test_startup.log; then
    echo ""
    echo "⚠️  Warning: Detected errors during startup test"
    echo "Attempting to start anyway..."
fi

# Start in background
echo ""
echo "Starting background server..."
nohup uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log \
    --proxy-headers \
    --no-server-header \
    > /workspace/uvicorn.log 2>&1 &

PID=$!
echo "Started uvicorn with PID: $PID"

# Step 9: Wait for server to start
echo ""
echo "Step 9: Waiting for server to start..."
SUCCESS=false
for i in {1..20}; do
    if curl -sS http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo " ✓ Server is responding!"
        SUCCESS=true
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Step 10: Test endpoints
if [ "$SUCCESS" = true ]; then
    echo ""
    echo "Step 10: Testing endpoints..."
    echo "Root endpoint (/):"
    curl -sS http://127.0.0.1:8000/ 2>&1 | python -m json.tool 2>/dev/null | head -5 || curl -sS http://127.0.0.1:8000/ 2>&1 | head -5
    echo ""
    echo "Patients API (/api/patients):"
    curl -sS http://127.0.0.1:8000/api/patients 2>&1 | python -m json.tool 2>/dev/null | head -5 || curl -sS http://127.0.0.1:8000/api/patients 2>&1 | head -5
    echo ""
    echo "========================================="
    echo "✅ SUCCESS! Backend is running on port 8000"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. On your LAPTOP, create SSH tunnel (in a new terminal):"
    echo ""
    echo "   ssh -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes g1sz4cjvu4dfdc-64411397@ssh.runpod.io"
    echo ""
    echo "2. Test on laptop:"
    echo "   curl http://127.0.0.1:8000/"
    echo "   curl http://127.0.0.1:8000/api/patients"
    echo ""
    echo "3. Access frontend at http://localhost:5173"
    echo ""
    echo "4. To keep the backend running, leave this SSH session open"
else
    echo ""
    echo "========================================="
    echo "❌ FAILED: Backend did not start properly"
    echo "========================================="
    echo ""
    echo "Checking uvicorn.log for errors:"
    echo "--------------------------------"
    tail -30 /workspace/uvicorn.log
    echo "--------------------------------"
    echo ""
    echo "Checking if process is still running:"
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process $PID is still running but not responding"
        echo ""
        echo "Process details:"
        ps aux | grep $PID | grep -v grep
    else
        echo "Process $PID has died"
    fi
    echo ""
    echo "Port 8000 status:"
    lsof -i :8000 2>/dev/null || echo "Port 8000 is not in use"
    echo ""
    echo "Troubleshooting suggestions:"
    echo "1. Check if all dependencies are installed:"
    echo "   pip list | grep -E 'fastapi|uvicorn|pydantic|chromadb|sentence-transformers'"
    echo ""
    echo "2. Try running manually to see errors:"
    echo "   cd $(pwd)"
    echo "   python main.py"
    echo ""
    echo "3. Check Python import paths:"
    echo "   python -c 'import sys; print(sys.path)'"
fi
