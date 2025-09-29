# RunPod Fix Commands - Copy & Paste These

## On your RunPod SSH session, run these commands:

```bash
# 1. Pull the latest changes from git
cd /workspace/rag_medical_gpu
git pull

# 2. Navigate to backend and run the immediate fix script
cd backend
chmod +x runpod_immediate_fix.sh
bash runpod_immediate_fix.sh
```

## If the automatic fix doesn't work, try manual steps:

```bash
# Ensure you're in the medrag environment
micromamba activate medrag

# Navigate to backend
cd /workspace/rag_medical_gpu/backend

# Kill any existing processes
pkill -f 'uvicorn main:app'
lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null

# Install dependencies
pip install fastapi uvicorn python-multipart chromadb sentence-transformers

# Start the server
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers --no-server-header > /workspace/uvicorn.log 2>&1 &

# Wait and test
sleep 3
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/patients
```

## Expected successful output:
```
✓ SUCCESS! Server is running!
Testing endpoints:
GET /:
{"status":"healthy","message":"Medical RAG API is running"}
GET /api/patients:
[list of patients]
```

## After backend is running, on your laptop:

```bash
# Create SSH tunnel (keep this terminal open!)
ssh -N -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519 -o ServerAliveInterval=60 g1sz4cjvu4dfdc-64411397@ssh.runpod.io
```

Then in a new terminal:
```bash
# Test the tunnel
curl http://127.0.0.1:8000/api/patients

# Your frontend at http://localhost:5173 should now work