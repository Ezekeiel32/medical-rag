#!/bin/bash

echo "=== Testing RunPod Connection ==="
echo ""

# Step 1: Kill any existing SSH tunnels
echo "1. Cleaning up existing SSH tunnels..."
pkill -f "ssh.*8000" 2>/dev/null
sleep 2

# Step 2: Create new SSH tunnel
echo "2. Creating SSH tunnel to RunPod..."
ssh -N -L 8000:127.0.0.1:8000 \
    -i ~/.ssh/id_ed25519 \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=10 \
    g1sz4cjvu4dfdc-64411397@ssh.runpod.io &

SSH_PID=$!
echo "SSH tunnel PID: $SSH_PID"

# Step 3: Wait for tunnel to establish
echo "3. Waiting for tunnel to establish..."
sleep 5

# Step 4: Test the connection
echo "4. Testing backend connection..."
echo ""

# Try multiple times
for i in {1..3}; do
    echo "Attempt $i:"
    
    # Test with curl
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/ 2>/dev/null)
    
    if [ "$RESPONSE" = "200" ]; then
        echo "✓ Success! Backend is accessible through tunnel"
        echo ""
        echo "Testing endpoints:"
        echo "GET /:"
        curl -s http://127.0.0.1:8000/ 2>/dev/null | head -50
        echo ""
        echo "GET /api/patients:"
        curl -s http://127.0.0.1:8000/api/patients 2>/dev/null | head -50
        echo ""
        echo "==================================="
        echo "✓ SSH Tunnel is working!"
        echo "==================================="
        echo ""
        echo "Backend URL: http://127.0.0.1:8000"
        echo "SSH Tunnel PID: $SSH_PID"
        echo ""
        echo "Keep this terminal open to maintain the tunnel."
        echo "Access your frontend at: http://localhost:5173"
        echo ""
        echo "To stop the tunnel: kill $SSH_PID"
        
        # Keep the script running to maintain tunnel
        wait $SSH_PID
        exit 0
    else
        echo "✗ Failed (HTTP code: $RESPONSE)"
        sleep 2
    fi
done

echo ""
echo "==================================="
echo "✗ Connection Failed"
echo "==================================="
echo ""
echo "The SSH tunnel was created but the backend is not responding."
echo ""
echo "Possible issues:"
echo "1. The backend on RunPod may have stopped"
echo "2. The SSH connection may be blocked"
echo ""
echo "To fix:"
echo "1. SSH into RunPod manually:"
echo "   ssh -i ~/.ssh/id_ed25519 g1sz4cjvu4dfdc-64411397@ssh.runpod.io"
echo ""
echo "2. Restart the backend:"
echo "   cd /workspace/rag_medical_gpu"
echo "   bash runpod_immediate_fix.sh"
echo ""
echo "3. Then run this script again."

# Kill the non-working tunnel
kill $SSH_PID 2>/dev/null

exit 1