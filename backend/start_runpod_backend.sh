#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting RunPod Backend Server${NC}"

# Ensure we're in the right directory
cd /workspace/rag_medical_gpu/backend || {
    echo -e "${RED}Error: Could not change to backend directory${NC}"
    echo "Trying alternative paths..."
    if [ -d /workspace/rag_medical/backend ]; then
        cd /workspace/rag_medical/backend
    else
        echo -e "${RED}Backend directory not found${NC}"
        exit 1
    fi
}

# Kill any existing uvicorn processes
echo -e "${YELLOW}Stopping any existing uvicorn processes...${NC}"
pkill -f 'uvicorn main:app' || true
sleep 2

# Check if port is free
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Port 8000 is in use, killing process...${NC}"
    lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Create log directory if it doesn't exist
mkdir -p /workspace/logs

# Start uvicorn with proper settings for stability
echo -e "${GREEN}Starting uvicorn server...${NC}"
echo "Command: uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers --no-server-header"

# Start uvicorn and log output
nohup uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log \
    --proxy-headers \
    --no-server-header \
    --log-level info \
    > /workspace/logs/uvicorn.log 2>&1 &

UVICORN_PID=$!
echo "Uvicorn PID: $UVICORN_PID"

# Wait for server to start
echo -e "${YELLOW}Waiting for server to start...${NC}"
for i in {1..30}; do
    if curl -sS http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running!${NC}"
        
        # Test endpoints
        echo -e "\n${GREEN}Testing endpoints:${NC}"
        echo -e "${YELLOW}GET /${NC}"
        curl -sS http://127.0.0.1:8000/ 2>/dev/null | python -m json.tool 2>/dev/null || curl -sS http://127.0.0.1:8000/
        
        echo -e "\n${YELLOW}GET /api/patients${NC}"
        curl -sS http://127.0.0.1:8000/api/patients 2>/dev/null | python -m json.tool 2>/dev/null || curl -sS http://127.0.0.1:8000/api/patients
        
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Backend is running successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Server PID: $UVICORN_PID"
        echo "Log file: /workspace/logs/uvicorn.log"
        echo ""
        echo "To check logs: tail -f /workspace/logs/uvicorn.log"
        echo "To stop server: kill $UVICORN_PID"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo -e "\n${RED}✗ Server failed to start after 30 seconds${NC}"
echo "Checking logs..."
tail -20 /workspace/logs/uvicorn.log
exit 1