#!/bin/bash

# SSH Tunnel Setup Script for Laptop
# This script establishes a stable SSH tunnel to your RunPod GPU pod

echo "========================================="
echo "RunPod SSH Tunnel Manager"
echo "========================================="

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
RUNPOD_USER="g1sz4cjvu4dfdc-64411397"
RUNPOD_HOST="ssh.runpod.io"
LOCAL_PORT=8000
REMOTE_PORT=8000

# Function to check if tunnel is already running
check_tunnel() {
    if pgrep -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}.*${RUNPOD_USER}" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to kill existing tunnels
kill_tunnel() {
    echo "Killing existing SSH tunnels..."
    pkill -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}.*${RUNPOD_USER}" 2>/dev/null || true
    sleep 2
}

# Function to test local endpoint
test_local() {
    echo "Testing local endpoints..."
    echo -n "  Root endpoint (/)... "
    if curl -sS http://127.0.0.1:${LOCAL_PORT}/ >/dev/null 2>&1; then
        echo "✓ OK"
        return 0
    else
        echo "✗ Failed"
        return 1
    fi
}

# Main script
case "${1:-start}" in
    start)
        if check_tunnel; then
            echo "Tunnel already running!"
            echo "Testing connection..."
            if test_local; then
                echo "✓ Tunnel is working"
            else
                echo "✗ Tunnel exists but not responding"
                echo "Run '$0 restart' to fix"
            fi
        else
            echo "Starting SSH tunnel..."
            ssh -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} \
                -i ${SSH_KEY} \
                -o IdentitiesOnly=yes \
                -o ServerAliveInterval=30 \
                -o ServerAliveCountMax=3 \
                -o ExitOnForwardFailure=yes \
                -N -f \
                ${RUNPOD_USER}@${RUNPOD_HOST}
            
            sleep 3
            
            if check_tunnel; then
                echo "✓ Tunnel started"
                echo "Testing connection..."
                if test_local; then
                    echo "✓ Backend accessible at http://localhost:${LOCAL_PORT}"
                    echo ""
                    echo "You can now:"
                    echo "  1. Test API: curl http://localhost:${LOCAL_PORT}/api/patients"
                    echo "  2. Open frontend: http://localhost:5173"
                else
                    echo "✗ Tunnel created but backend not responding"
                    echo "Check if backend is running on the pod"
                fi
            else
                echo "✗ Failed to start tunnel"
            fi
        fi
        ;;
        
    stop)
        kill_tunnel
        echo "Tunnel stopped"
        ;;
        
    restart)
        kill_tunnel
        echo "Restarting tunnel..."
        $0 start
        ;;
        
    status)
        if check_tunnel; then
            echo "✓ Tunnel is running"
            ps aux | grep -v grep | grep "ssh.*-L ${LOCAL_PORT}"
            echo ""
            test_local
        else
            echo "✗ No tunnel running"
        fi
        ;;
        
    test)
        echo "Testing backend endpoints through tunnel..."
        echo ""
        echo "1. Root endpoint:"
        curl -sS http://localhost:${LOCAL_PORT}/ | python -m json.tool 2>/dev/null | head -10 || echo "Failed"
        echo ""
        echo "2. Patients API:"
        curl -sS http://localhost:${LOCAL_PORT}/api/patients | python -m json.tool 2>/dev/null | head -10 || echo "Failed"
        echo ""
        echo "3. Frontend (if running):"
        curl -sS -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:5173/ 2>/dev/null || echo "Frontend not running"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start SSH tunnel (default)"
        echo "  stop    - Stop SSH tunnel"
        echo "  restart - Restart SSH tunnel"
        echo "  status  - Check tunnel status"
        echo "  test    - Test all endpoints"
        exit 1
        ;;
esac