#!/bin/bash

# RunPod GPU Backend Diagnostic and Fix Script
# This script diagnoses and fixes issues with the uvicorn backend on RunPod

echo "========================================="
echo "RunPod Backend Diagnostic & Fix Script"
echo "========================================="
echo ""

# Function to print status
print_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if port is in use
check_port() {
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

print_status "Step 1: Checking existing uvicorn processes..."
if pgrep -f 'uvicorn main:app' > /dev/null; then
    print_status "Found existing uvicorn processes. Killing them..."
    pkill -f 'uvicorn main:app'
    sleep 2
else
    print_status "No existing uvicorn processes found."
fi

print_status "Step 2: Checking if port 8000 is in use..."
if check_port; then
    print_status "Port 8000 is in use. Checking what's using it..."
    lsof -i :8000
    print_status "Killing process on port 8000..."
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
else
    print_status "Port 8000 is free."
fi

print_status "Step 3: Checking uvicorn log for previous errors..."
if [ -f /workspace/uvicorn.log ]; then
    echo "Last 20 lines of uvicorn.log:"
    echo "--------------------------------"
    tail -20 /workspace/uvicorn.log
    echo "--------------------------------"
    echo ""
fi

print_status "Step 4: Verifying Python environment..."
echo "Python location: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo ""

print_status "Step 5: Checking required packages..."
echo "Checking uvicorn installation..."
if python -c "import uvicorn" 2>/dev/null; then
    echo "✓ uvicorn is installed"
    uvicorn --version
else
    echo "✗ uvicorn is NOT installed!"
    echo "Installing uvicorn..."
    pip install uvicorn[standard]
fi

echo "Checking fastapi installation..."
if python -c "import fastapi" 2>/dev/null; then
    echo "✓ fastapi is installed"
else
    echo "✗ fastapi is NOT installed!"
    echo "Installing fastapi..."
    pip install fastapi
fi

echo "Checking other critical dependencies..."
for package in "pydantic" "httpx" "python-multipart" "python-dotenv" "surya_ocr"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is NOT installed!"
        case $package in
            "surya_ocr")
                echo "Installing surya-ocr..."
                pip install surya-ocr
                ;;
            *)
                echo "Installing $package..."
                pip install $package
                ;;
        esac
    fi
done

print_status "Step 6: Checking backend directory and main.py..."
cd /workspace/rag_medical_gpu/backend

if [ ! -f main.py ]; then
    print_status "ERROR: main.py not found in /workspace/rag_medical_gpu/backend"
    print_status "Current directory contents:"
    ls -la
    exit 1
fi

print_status "Step 7: Testing Python import of main.py..."
echo "Testing if main.py can be imported..."
python -c "import main; print('✓ main.py imported successfully')" 2>&1 | head -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_status "ERROR: Failed to import main.py. Check the error above."
    echo ""
    print_status "Attempting to diagnose import issues..."
    python -c "
import sys
import os
sys.path.insert(0, '/workspace/rag_medical_gpu/backend')
sys.path.insert(0, '/workspace/rag_medical_gpu')
print('Python path:', sys.path[:3])
try:
    import main
    print('✓ Import successful after path adjustment')
except ImportError as e:
    print('Import error:', e)
except Exception as e:
    print('Other error:', e)
"
fi

print_status "Step 8: Starting uvicorn with verbose output..."
echo "Starting uvicorn in foreground for 5 seconds to check for errors..."
timeout 5 python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug 2>&1 | head -50
echo ""

print_status "Step 9: Starting uvicorn in background..."
# Clear old log
> /workspace/uvicorn.log

# Start uvicorn with proper environment
export PYTHONPATH="/workspace/rag_medical_gpu/backend:$PYTHONPATH"
nohup python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log \
    --proxy-headers \
    --no-server-header \
    --log-level info \
    > /workspace/uvicorn.log 2>&1 &

UVICORN_PID=$!
echo "Started uvicorn with PID: $UVICORN_PID"

print_status "Step 10: Waiting for server to start..."
for i in {1..30}; do
    if curl -sS http://127.0.0.1:8000/ >/dev/null 2>&1; then
        print_status "✓ Server is responding!"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

print_status "Step 11: Testing endpoints..."
echo "Testing root endpoint (/)..."
curl -sS http://127.0.0.1:8000/ 2>&1 | head -5
echo ""
echo "Testing /api/patients endpoint..."
curl -sS http://127.0.0.1:8000/api/patients 2>&1 | head -5
echo ""

print_status "Step 12: Checking server status..."
if check_port; then
    print_status "✓ Server is running on port 8000"
    print_status "Process details:"
    ps aux | grep "[u]vicorn main:app"
else
    print_status "✗ Server is NOT running on port 8000"
    print_status "Checking uvicorn.log for errors:"
    tail -20 /workspace/uvicorn.log
fi

echo ""
echo "========================================="
echo "Diagnostic Complete"
echo "========================================="
echo ""
echo "If the server is running, you can now:"
echo "1. On your laptop, create SSH tunnel:"
echo "   ssh -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes g1sz4cjvu4dfdc-64411397@ssh.runpod.io"
echo ""
echo "2. Test on laptop:"
echo "   curl http://127.0.0.1:8000/"
echo "   curl http://127.0.0.1:8000/api/patients"
echo ""
echo "3. Access frontend at http://localhost:5173"