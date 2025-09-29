# RunPod GPU Backend Troubleshooting Guide

## Current Issue
The uvicorn backend is not starting on your RunPod GPU pod, showing "Connection refused" errors.

## Quick Fix Steps (Run on RunPod Pod)

### Step 1: Check and Fix Python Environment

```bash
# Ensure you're in the correct environment
micromamba activate medrag

# Check Python location
which python
# Should show something like /opt/micromamba/envs/medrag/bin/python

# Navigate to backend
cd /workspace/rag_medical_gpu/backend
```

### Step 2: Install/Verify Dependencies

```bash
# Check and install missing dependencies
pip install --upgrade pip
pip install uvicorn[standard] fastapi pydantic httpx python-multipart python-dotenv

# Verify Surya OCR is installed
pip list | grep surya
# If not installed:
# pip install surya-ocr
```

### Step 3: Debug the Main Application

```bash
# Test if main.py can be imported
cd /workspace/rag_medical_gpu/backend
python -c "
try:
    from main import app
    print('✓ main.py imports successfully')
except Exception as e:
    print(f'✗ Import error: {e}')
    import traceback
    traceback.print_exc()
"
```

### Step 4: Common Fixes for Import Errors

If you see import errors, try these fixes:

```bash
# Fix 1: CUDA/Surya memory issues
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="0"

# Fix 2: Path issues
export PYTHONPATH="/workspace/rag_medical_gpu/backend:/workspace/rag_medical_gpu:$PYTHONPATH"

# Fix 3: Missing src directory imports
cd /workspace/rag_medical_gpu
if [ -d "src" ]; then
    export PYTHONPATH="/workspace/rag_medical_gpu/src:$PYTHONPATH"
fi
cd backend
```

### Step 5: Start Backend with Error Visibility

```bash
# Kill any stuck processes first
pkill -f 'uvicorn main:app' || true
fuser -k 8000/tcp 2>/dev/null || true

# Start with full logging to see errors
cd /workspace/rag_medical_gpu/backend
python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level debug \
    --reload-exclude '*'
```

If this shows errors, address them before proceeding.

### Step 6: Start Backend in Production Mode

Once errors are fixed:

```bash
# Start in background without auto-reload
cd /workspace/rag_medical_gpu/backend
nohup uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log \
    --proxy-headers \
    --no-server-header \
    --workers 1 \
    > /workspace/uvicorn.log 2>&1 &

# Wait for startup
sleep 5

# Test endpoints
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/patients
```

## Common Error Solutions

### Error: "No module named 'src'"
```bash
# Fix Python path
cd /workspace/rag_medical_gpu
export PYTHONPATH="$(pwd):$(pwd)/src:$(pwd)/backend:$PYTHONPATH"
```

### Error: "CUDA out of memory"
```bash
# Limit GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=1

# Or use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

### Error: "Address already in use"
```bash
# Find and kill process using port 8000
lsof -i :8000
fuser -k 8000/tcp
```

### Error: Missing environment variables
```bash
# Create .env file if needed
cd /workspace/rag_medical_gpu/backend
cat > .env << 'EOF'
# Add your environment variables here
ENVIRONMENT=production
LOG_LEVEL=info
EOF
```

## SSH Tunnel Setup (Run on Laptop)

### One-time setup:
```bash
# Make the tunnel script executable
chmod +x laptop_ssh_tunnel.sh

# Start the tunnel
./laptop_ssh_tunnel.sh start
```

### Manual tunnel command:
```bash
# Kill any existing tunnels
pkill -f "ssh.*-L 8000:127.0.0.1:8000.*g1sz4cjvu4dfdc" || true

# Create new tunnel
ssh -L 8000:127.0.0.1:8000 \
    -i ~/.ssh/id_ed25519 \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=30 \
    -N -f \
    g1sz4cjvu4dfdc-64411397@ssh.runpod.io

# Test connection
curl http://localhost:8000/
curl http://localhost:8000/api/patients
```

## Frontend Setup (Run on Laptop)

```bash
# Navigate to frontend directory
cd ~/rag_medical_gpu/frontend  # Adjust path as needed

# Ensure correct backend URL
export VITE_BACKEND_URL="http://127.0.0.1:8000"

# Or update vite.config.js proxy settings
# proxy: {
#   '/api': {
#     target: 'http://127.0.0.1:8000',
#     changeOrigin: true
#   }
# }

# Start frontend
npm run dev

# Access at http://localhost:5173
```

## Verification Checklist

- [ ] Backend running on pod (port 8000)
- [ ] SSH tunnel active on laptop
- [ ] Backend accessible via tunnel (http://localhost:8000)
- [ ] Frontend running on laptop (port 5173)
- [ ] Frontend can reach backend API

## Alternative: Public URL (if SSH tunnel fails)

On RunPod dashboard:
1. Go to your pod settings
2. Click "Networking" → "Add HTTP Port"
3. Add port 8000
4. Copy the provided public URL
5. Update frontend to use public URL:
   ```bash
   export VITE_BACKEND_URL="https://your-pod-id-8000.proxy.runpod.net"
   ```

## Debug Commands Reference

```bash
# Check backend logs
tail -f /workspace/uvicorn.log

# Check running processes
ps aux | grep uvicorn

# Check port usage
netstat -tulpn | grep 8000

# Check Python packages
pip list | grep -E "uvicorn|fastapi|surya"

# Monitor GPU usage (if applicable)
nvidia-smi

# Check disk space
df -h /workspace
```

## Contact Support If:
- Backend crashes immediately after starting
- Persistent CUDA/GPU errors
- Network connectivity issues between pod and laptop
- Frontend shows CORS errors despite correct setup