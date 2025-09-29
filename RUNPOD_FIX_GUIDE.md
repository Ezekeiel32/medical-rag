# RunPod Backend Fix Guide

## Problem Summary
Your uvicorn backend server on RunPod GPU pod is failing to start, showing "Connection refused" errors when trying to access endpoints.

## Root Causes & Solutions

### Common Issues:
1. **Wrong directory** - The backend might not be in the expected location
2. **Missing dependencies** - FastAPI/Uvicorn might not be installed in the conda environment
3. **Import errors** - The main.py might have import issues
4. **Port conflicts** - Port 8000 might already be in use
5. **Environment not activated** - The medrag conda environment might not be properly activated

## Step-by-Step Fix Instructions

### On the RunPod Pod (via SSH)

#### Step 1: Run the Troubleshooting Script
```bash
# SSH into your RunPod
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes g1sz4cjvu4dfdc-64411397@ssh.runpod.io

# Activate the environment
micromamba activate medrag

# Navigate to backend directory
cd /workspace/rag_medical_gpu/backend

# Make the troubleshooting script executable and run it
chmod +x troubleshoot_runpod.sh
bash troubleshoot_runpod.sh
```

This script will:
- Check uvicorn logs for errors
- Verify Python environment and dependencies
- Test if main.py imports correctly
- Check port availability
- Attempt to start the server with verbose output

#### Step 2: Fix Any Issues Found

Based on the troubleshooting output:

**If dependencies are missing:**
```bash
# Install missing dependencies
pip install fastapi uvicorn pydantic python-multipart
```

**If main.py has import errors:**
```bash
# Check the specific import error and install missing packages
# Common missing packages:
pip install torch torchvision surya-ocr
pip install chromadb sentence-transformers
pip install pandas numpy pillow
```

**If the directory is wrong:**
```bash
# Find the correct location
find /workspace -name "main.py" -path "*/backend/*" 2>/dev/null
# Then cd to the correct directory
```

#### Step 3: Start the Backend Properly
```bash
# Make sure you're in the right directory
cd /workspace/rag_medical_gpu/backend

# Make the startup script executable and run it
chmod +x start_runpod_backend.sh
bash start_runpod_backend.sh
```

If the script succeeds, you'll see:
```
✓ Server is running!
Testing endpoints:
GET /
GET /api/patients
```

#### Step 4: Alternative Manual Start (if scripts fail)
```bash
# Ensure environment is activated
micromamba activate medrag

# Navigate to backend
cd /workspace/rag_medical_gpu/backend

# Kill any existing processes
pkill -f 'uvicorn main:app'

# Start with explicit Python path
python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level debug \
    --reload-dir . \
    2>&1 | tee /workspace/uvicorn_debug.log
```

### On Your Local Laptop

#### Step 1: Set Up Stable SSH Tunnel
```bash
# Navigate to your local project
cd ~/rag_medical\ \(Copy\)/backend

# Make the tunnel script executable
chmod +x setup_ssh_tunnel.sh

# Run the tunnel setup
bash setup_ssh_tunnel.sh
```

The script will:
- Kill any existing SSH tunnels on port 8000
- Create a new stable tunnel with keep-alive settings
- Test the connection
- Keep running to maintain the tunnel

#### Step 2: Alternative Manual Tunnel (if script fails)
```bash
# Kill any existing tunnels
ps aux | grep "ssh.*8000:127.0.0.1:8000" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

# Create new tunnel with keep-alive
ssh -N -L 8000:127.0.0.1:8000 \
    -i ~/.ssh/id_ed25519 \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    g1sz4cjvu4dfdc-64411397@ssh.runpod.io
```

#### Step 3: Verify Tunnel Works
In a new terminal on your laptop:
```bash
# Test backend via tunnel
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/patients
```

#### Step 4: Access Frontend
```bash
# Make sure your frontend is configured correctly
# In your frontend directory, check vite.config.js has:
# proxy: {
#   '/api': {
#     target: 'http://127.0.0.1:8000',
#     changeOrigin: true
#   }
# }

# Start frontend (if not already running)
npm run dev

# Access at http://localhost:5173
```

## Quick Diagnostic Commands

### On RunPod Pod:
```bash
# Check if server is running
curl http://127.0.0.1:8000/

# Check running processes
ps aux | grep uvicorn

# Check port usage
lsof -i :8000

# Check logs
tail -f /workspace/logs/uvicorn.log
```

### On Laptop:
```bash
# Check tunnel
ps aux | grep "ssh.*8000"

# Test backend through tunnel
curl http://127.0.0.1:8000/api/patients
```

## Stable Configuration Summary

### Backend (RunPod):
```bash
# Start command (no auto-reload for stability)
uvicorn main:app --host 0.0.0.0 --port 8000 \
    --no-access-log --proxy-headers --no-server-header
```

### SSH Tunnel (Laptop):
```bash
# Stable tunnel with keep-alive
ssh -N -L 8000:127.0.0.1:8000 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -i ~/.ssh/id_ed25519 \
    g1sz4cjvu4dfdc-64411397@ssh.runpod.io
```

### Frontend (Laptop):
```javascript
// vite.config.js proxy configuration
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    secure: false
  }
}
```

## Alternative: RunPod Public URL

If SSH tunnel is unstable, use RunPod's public URL:

1. In RunPod dashboard, go to your pod
2. Click "Networking" → "Add HTTP Port"
3. Add port 8000
4. Copy the provided public URL
5. Update frontend environment:
   ```bash
   export VITE_BACKEND_URL=https://your-pod-id-8000.proxy.runpod.net
   npm run dev
   ```

## Troubleshooting Tips

1. **Always activate conda environment first**: `micromamba activate medrag`
2. **Check you're in the right directory**: Backend files should be in `/workspace/rag_medical_gpu/backend`
3. **Monitor logs**: Keep `tail -f /workspace/logs/uvicorn.log` running in a separate terminal
4. **Keep tunnel alive**: Don't close the terminal running the SSH tunnel
5. **No auto-reload**: Don't use `--reload` flag in production for stability

## Success Checklist
- [ ] Backend starts without errors on RunPod
- [ ] `curl http://127.0.0.1:8000/` works on RunPod
- [ ] SSH tunnel established from laptop
- [ ] `curl http://127.0.0.1:8000/api/patients` works on laptop
- [ ] Frontend loads at http://localhost:5173
- [ ] Frontend can fetch data from backend

If all checks pass, your system is properly configured and stable!