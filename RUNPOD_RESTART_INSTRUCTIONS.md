# RunPod Backend Restart Instructions

## The Problem
The backend on RunPod has stopped running. The SSH tunnel is connecting but can't forward traffic because there's no backend listening on port 8000.

## Quick Fix Instructions

### Step 1: SSH into RunPod (in a new terminal)
```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes g1sz4cjvu4dfdc-64411397@ssh.runpod.io
```

### Step 2: Once connected to RunPod, restart the backend
```bash
# Activate environment
micromamba activate medrag

# Go to backend directory
cd /workspace/rag_medical_gpu/backend

# Kill any existing processes
pkill -f 'uvicorn main:app'

# Start the backend
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers --no-server-header > /workspace/uvicorn.log 2>&1 &

# Test it's running
sleep 3
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/patients
```

### Step 3: Keep the RunPod SSH session open

Don't close this terminal! The backend needs to keep running.

### Step 4: On your laptop (different terminal), create the SSH tunnel
```bash
# Kill any existing tunnels
pkill -f "ssh.*8000"

# Create new tunnel
ssh -N -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes g1sz4cjvu4dfdc-64411397@ssh.runpod.io
```

Keep this terminal open too!

### Step 5: Test on your laptop (another terminal)
```bash
# Test backend through tunnel
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/patients
```

### Step 6: Access your frontend
Open http://localhost:5173 in your browser

## Alternative: Use tmux or screen on RunPod

For a more stable setup, use tmux on RunPod to keep the backend running:

```bash
# On RunPod
tmux new -s backend

# Inside tmux
micromamba activate medrag
cd /workspace/rag_medical_gpu/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers --no-server-header

# Detach from tmux: Press Ctrl+B, then D

# To reattach later:
tmux attach -t backend
```

## Troubleshooting

If the backend keeps stopping, check the logs:
```bash
# On RunPod
tail -100 /workspace/uvicorn.log
```

Common issues:
- **Out of memory**: Reduce batch sizes or use CPU mode
- **Import errors**: Install missing packages with pip
- **Port conflict**: Kill processes using port 8000

## Keep It Running

For long-term stability:
1. Use tmux/screen on RunPod
2. Monitor resource usage with `nvidia-smi` and `htop`
3. Consider using RunPod's public HTTP port instead of SSH tunnel