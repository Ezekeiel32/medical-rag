# MediRAG with Google Gemini API

This version of MediRAG uses Google's Gemini API for high-quality Hebrew medical responses.

## Quick Setup

### 1. Get Your Free Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Configure the System
```bash
# Set your API key
export GEMINI_API_KEY="your_actual_api_key_here"

# Or use the setup script
./setup_gemini.sh your_actual_api_key_here
```

### 3. Start the System
```bash
# Start backend
cd backend
python main.py

# In another terminal, start frontend (if needed)
cd MediRAG_UI/frontend
npm run dev
```

## API Key Safety
- Never commit your API key to version control
- Keep your API key secure and private
- The system includes fallback to local Ollama if Gemini fails

## Benefits of Gemini API
- ✅ **Superior Hebrew language understanding**
- ✅ **No Chinese text contamination**
- ✅ **Better medical terminology handling**
- ✅ **More coherent and complete responses**
- ✅ **Professional medical analysis**

## Troubleshooting

### "GEMINI_API_KEY not set" error
- Make sure you've exported the API key: `export GEMINI_API_KEY="your_key"`
- Check the key is valid at [AI Studio](https://aistudio.google.com/app/apikey)

### API quota exceeded
- Gemini has generous free limits
- Monitor usage at [Google Cloud Console](https://console.cloud.google.com/)

### System falls back to Ollama
- This happens when Gemini is unavailable
- Ensure Ollama is running: `ollama serve`
- Make sure `gemma2:2b-instruct` model is installed: `ollama pull gemma2:2b-instruct`

## System Architecture
- **Primary**: Google Gemini API (high quality)
- **Fallback**: Local Ollama (offline capability)
- **Language**: Hebrew medical terminology optimized
- **Output**: Clean, professional medical responses
