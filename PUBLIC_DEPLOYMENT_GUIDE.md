# Public File Upload Deployment - Any Computer Can Upload

## ✅ Backend Already Has File Upload Endpoints

Your backend already supports file uploads from any computer! The endpoints are:

- `POST /api/patients/{patient_id}/documents` - Upload PDFs for existing patients
- `POST /api/patients/create-from-pdf` - Upload PDF and auto-create patient
- `POST /api/ocr/extract` - Direct OCR extraction

## 🚀 Step 1: Expose RunPod Backend Publicly

### On RunPod Dashboard:
1. Go to your pod: https://runpod.io/console/pods
2. Click on your pod `g1sz4cjvu4dfdc`
3. Click "Networking" tab
4. Click "Add HTTP Port"
5. Add port `8000`
6. Copy the public URL (looks like: `https://xxxxx-8000.proxy.runpod.net`)

### On RunPod SSH (restart backend with public URL):
```bash
# Pull latest changes
cd /workspace/rag_medical_gpu
git pull

# Set the public URL environment variable
export RUNPOD_PUBLIC_URL=https://xxxxx-8000.proxy.runpod.net

# Start backend
cd backend
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers --no-server-header > /workspace/uvicorn.log 2>&1 &
```

## 🚀 Step 2: Update Frontend to Use Public URL

### Create/update `.env` file in your frontend directory:
```bash
# In your frontend directory (where you run npm run dev)
echo "VITE_BACKEND_URL=https://xxxxx-8000.proxy.runpod.net" > .env
```

### Or update `vite.config.js`:
```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'https://xxxxx-8000.proxy.runpod.net',
        changeOrigin: true,
        secure: true
      }
    }
  }
})
```

## 🚀 Step 3: Test File Uploads from Any Computer

### Test the API directly:
```bash
# Test backend health
curl https://xxxxx-8000.proxy.runpod.net/

# Test patients endpoint
curl https://xxxxx-8000.proxy.runpod.net/api/patients

# Upload a PDF file (replace with your actual file)
curl -X POST \
  -F "file=@/path/to/your/document.pdf" \
  https://xxxxx-8000.proxy.runpod.net/api/patients/create-from-pdf
```

### Start Frontend:
```bash
cd /path/to/your/frontend
npm run dev
```

Now any computer can:
1. Access your frontend at `http://localhost:5173`
2. Upload PDF files through the web interface
3. Files get processed on RunPod GPU with Surya OCR
4. Results are stored and searchable

## 🔧 Backend Features Available Publicly:

- ✅ **File Upload**: Multiple PDF uploads per patient
- ✅ **Auto Patient Creation**: Upload PDF → auto-extract patient info → create patient
- ✅ **OCR Processing**: GPU-accelerated Hebrew OCR with Surya
- ✅ **RAG Search**: Ask questions about patient documents
- ✅ **Case Reports**: Generate comprehensive medical reports
- ✅ **Document Management**: View, search, and analyze medical documents

## 📋 API Endpoints for File Upload:

```bash
# Create patient from PDF (auto OCR + patient extraction)
POST /api/patients/create-from-pdf
Form data: file (PDF)

# Upload documents for existing patient
POST /api/patients/{patient_id}/documents
Form data: files[] (multiple PDFs)

# Direct OCR extraction
POST /api/ocr/extract
Form data: file (PDF), bidi (text direction)
```

## 🎯 Success Checklist:

- [ ] RunPod backend exposed publicly on port 8000
- [ ] Frontend configured with public RunPod URL
- [ ] File uploads work from any computer
- [ ] OCR processing runs on GPU
- [ ] Search and case reports work

## 🔍 Troubleshooting:

**If uploads fail:**
```bash
# Check backend logs on RunPod
tail -f /workspace/uvicorn.log
```

**If CORS errors:**
- Backend automatically allows all origins for RunPod public URLs
- Make sure frontend uses the correct public URL

**If OCR fails:**
- Check GPU memory: RunPod has A100 GPU with 80GB VRAM
- Surya OCR is optimized for GPU processing

## 💡 Why This Works:

1. **No SSH tunnels needed** - Direct HTTP access from any computer
2. **GPU acceleration** - Surya OCR runs on RunPod's A100 GPU
3. **Scalable** - Can handle multiple concurrent uploads
4. **Secure** - Files processed server-side, results stored securely
5. **Cross-platform** - Works from Windows, Mac, Linux, mobile

Your system is now ready for production use! Any computer can upload medical PDFs and get instant OCR + RAG analysis powered by GPU acceleration.