#!/bin/bash

echo "=== MediRAG Gemini API Setup ==="
echo ""

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: ./setup_gemini.sh <your_gemini_api_key>"
    echo ""
    echo "Get your free Gemini API key at: https://aistudio.google.com/app/apikey"
    echo ""
    echo "Example:"
    echo "./setup_gemini.sh AIzaSyBx_abc123def456..."
    exit 1
fi

API_KEY=$1

# Set environment variable
export GEMINI_API_KEY=$API_KEY

# Add to .env file
echo "GEMINI_API_KEY=$API_KEY" > .env

echo "✅ Gemini API key configured successfully!"
echo ""

# Test the API
echo "🧪 Testing Gemini API connection..."
cd $(dirname $0)

# Activate virtual environment
source venv/bin/activate

# Test the API
python3 -c "
from src.llm_gemini import test_gemini_api
if test_gemini_api():
    print('✅ Gemini API is working correctly!')
    print('🚀 You can now start the MediRAG system')
else:
    print('❌ Gemini API test failed. Please check your API key.')
    exit(1)
"

echo ""
echo "To start the system:"
echo "1. cd backend"
echo "2. python main.py"
echo ""
echo "Frontend will be available at: http://localhost:5173"
