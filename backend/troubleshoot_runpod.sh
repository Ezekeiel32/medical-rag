#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RunPod Backend Troubleshooting Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Check uvicorn log
echo -e "${YELLOW}Step 1: Checking uvicorn log for errors...${NC}"
if [ -f /workspace/uvicorn.log ]; then
    echo -e "Last 20 lines of uvicorn.log:"
    tail -20 /workspace/uvicorn.log
    echo ""
else
    echo -e "${RED}No uvicorn.log found at /workspace/${NC}"
fi

# Step 2: Kill any existing uvicorn processes
echo -e "${YELLOW}Step 2: Killing any existing uvicorn processes...${NC}"
pkill -f 'uvicorn main:app' || echo "No uvicorn processes found"
sleep 1

# Step 3: Check Python environment
echo -e "${YELLOW}Step 3: Checking Python environment...${NC}"
which python
python --version
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 4: Check if we're in the correct directory
echo -e "${YELLOW}Step 4: Checking current directory...${NC}"
pwd
echo ""

# Step 5: Check if main.py exists
echo -e "${YELLOW}Step 5: Checking if main.py exists...${NC}"
if [ -f main.py ]; then
    echo -e "${GREEN}main.py found${NC}"
else
    echo -e "${RED}main.py not found in current directory${NC}"
    echo "Checking /workspace/rag_medical_gpu/backend/..."
    if [ -f /workspace/rag_medical_gpu/backend/main.py ]; then
        echo -e "${GREEN}Found at /workspace/rag_medical_gpu/backend/main.py${NC}"
        cd /workspace/rag_medical_gpu/backend/
    else
        echo -e "${RED}main.py not found anywhere!${NC}"
        exit 1
    fi
fi
echo ""

# Step 6: Check Python dependencies
echo -e "${YELLOW}Step 6: Checking critical Python dependencies...${NC}"
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import fastapi
    print(f'✓ FastAPI installed: {fastapi.__version__}')
except ImportError as e:
    print(f'✗ FastAPI not installed: {e}')
try:
    import uvicorn
    print(f'✓ Uvicorn installed')
except ImportError as e:
    print(f'✗ Uvicorn not installed: {e}')
try:
    import pydantic
    print(f'✓ Pydantic installed: {pydantic.__version__}')
except ImportError as e:
    print(f'✗ Pydantic not installed: {e}')
try:
    import torch
    print(f'✓ PyTorch installed: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA devices: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'✗ PyTorch not installed: {e}')
"
echo ""

# Step 7: Try to import main.py and check for errors
echo -e "${YELLOW}Step 7: Testing main.py import...${NC}"
python -c "
try:
    import main
    print('✓ main.py imports successfully')
except Exception as e:
    print(f'✗ Error importing main.py: {e}')
    import traceback
    traceback.print_exc()
"
echo ""

# Step 8: Check for port availability
echo -e "${YELLOW}Step 8: Checking if port 8000 is available...${NC}"
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}Port 8000 is already in use by:${NC}"
    lsof -Pi :8000
    echo "Attempting to kill the process..."
    lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 1
else
    echo -e "${GREEN}Port 8000 is available${NC}"
fi
echo ""

# Step 9: Start uvicorn with better error handling
echo -e "${YELLOW}Step 9: Starting uvicorn with verbose output...${NC}"
echo "Running: uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info"
echo ""

# Start uvicorn in the background but capture initial output
timeout 5 uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info 2>&1 | tee /tmp/uvicorn_start.log &

# Wait a bit for server to start
sleep 3

# Check if server started successfully
echo -e "\n${YELLOW}Step 10: Checking if server started...${NC}"
if curl -sS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is running!${NC}"
    echo ""
    echo "Testing endpoints:"
    echo -e "${BLUE}GET /${NC}"
    curl -sS http://127.0.0.1:8000/ | head -50
    echo -e "\n${BLUE}GET /api/patients${NC}"
    curl -sS http://127.0.0.1:8000/api/patients | head -50
else
    echo -e "${RED}✗ Server failed to start${NC}"
    echo "Startup log:"
    cat /tmp/uvicorn_start.log
    echo ""
    echo -e "${YELLOW}Trying alternative startup method...${NC}"
    
    # Try with Python directly
    echo "Running: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug"
    timeout 5 python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug 2>&1 | tee /tmp/uvicorn_debug.log &
    sleep 3
    
    if curl -sS http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Server started with python -m uvicorn${NC}"
    else
        echo -e "${RED}✗ Server still not starting${NC}"
        echo "Debug log:"
        cat /tmp/uvicorn_debug.log
    fi
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Troubleshooting Complete${NC}"
echo -e "${BLUE}========================================${NC}"