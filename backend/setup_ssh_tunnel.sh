#!/bin/bash

# SSH Tunnel Setup Script for Local Laptop
# Run this on your local laptop, NOT on the RunPod pod

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SSH Tunnel Setup for RunPod Backend${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Configuration - Update these values
RUNPOD_SSH_USER="g1sz4cjvu4dfdc-64411397"  # Update with your actual RunPod SSH user
RUNPOD_HOST="ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_PORT=8000
REMOTE_PORT=8000

# Kill any existing SSH tunnels on port 8000
echo -e "${YELLOW}Checking for existing SSH tunnels...${NC}"
existing_tunnels=$(ps aux | grep "ssh.*8000:127.0.0.1:8000" | grep -v grep)
if [ ! -z "$existing_tunnels" ]; then
    echo -e "${YELLOW}Found existing tunnel, killing it...${NC}"
    ps aux | grep "ssh.*8000:127.0.0.1:8000" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Check if local port is in use
if lsof -Pi :${LOCAL_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Local port ${LOCAL_PORT} is already in use${NC}"
    echo "Process using port ${LOCAL_PORT}:"
    lsof -Pi :${LOCAL_PORT}
    echo ""
    read -p "Do you want to kill this process and continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -Pi :${LOCAL_PORT} -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 1
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Create SSH tunnel
echo -e "${GREEN}Creating SSH tunnel...${NC}"
echo "Command: ssh -N -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} -i ${SSH_KEY} -o IdentitiesOnly=yes ${RUNPOD_SSH_USER}@${RUNPOD_HOST}"

# Start SSH tunnel in background
ssh -N -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} \
    -i ${SSH_KEY} \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    ${RUNPOD_SSH_USER}@${RUNPOD_HOST} &

SSH_PID=$!
echo "SSH tunnel PID: $SSH_PID"

# Wait and test the tunnel
echo -e "${YELLOW}Waiting for tunnel to establish...${NC}"
sleep 3

# Test the tunnel
echo -e "${YELLOW}Testing tunnel connection...${NC}"
if curl -sS http://127.0.0.1:${LOCAL_PORT}/ >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Tunnel is working!${NC}"
    
    echo -e "\n${GREEN}Testing backend endpoints:${NC}"
    echo -e "${YELLOW}GET http://127.0.0.1:${LOCAL_PORT}/${NC}"
    curl -sS http://127.0.0.1:${LOCAL_PORT}/ 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -sS http://127.0.0.1:${LOCAL_PORT}/
    
    echo -e "\n${YELLOW}GET http://127.0.0.1:${LOCAL_PORT}/api/patients${NC}"
    curl -sS http://127.0.0.1:${LOCAL_PORT}/api/patients 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -sS http://127.0.0.1:${LOCAL_PORT}/api/patients
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}SSH Tunnel Established Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Tunnel PID: $SSH_PID"
    echo "Backend URL: http://127.0.0.1:${LOCAL_PORT}"
    echo "Frontend should connect to: http://127.0.0.1:${LOCAL_PORT}"
    echo ""
    echo "To keep tunnel running: Leave this terminal open"
    echo "To stop tunnel: Press Ctrl+C or kill $SSH_PID"
    echo ""
    echo -e "${BLUE}You can now access your frontend at http://localhost:5173${NC}"
    
    # Keep the script running to maintain the tunnel
    echo -e "${YELLOW}Tunnel is running. Press Ctrl+C to stop.${NC}"
    wait $SSH_PID
else
    echo -e "${RED}✗ Tunnel established but backend is not responding${NC}"
    echo "This could mean:"
    echo "1. The backend is not running on the RunPod pod"
    echo "2. The backend is running on a different port"
    echo ""
    echo "Please SSH into your RunPod and run:"
    echo "  cd /workspace/rag_medical_gpu/backend"
    echo "  bash troubleshoot_runpod.sh"
    echo ""
    echo "SSH tunnel is still running (PID: $SSH_PID)"
    echo "Press Ctrl+C to stop the tunnel"
    wait $SSH_PID
fi