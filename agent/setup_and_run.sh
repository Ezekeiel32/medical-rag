#!/bin/bash

# RAG System Research Session Setup and Runner
# This script checks the environment and starts the research session

echo "=========================================="
echo "RAG System Research Session Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "../src/cli.py" ]; then
    echo "❌ Error: Must be run from the agent/ directory within the RAG project"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "🔄 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if .env file exists and has API key
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "Creating template .env file..."
    cat > .env << EOF
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
EOF
    echo "✅ Created .env file template"
    echo "⚠️  Please edit .env and add your Google AI Studio API key"
    echo "   Get your key from: https://aistudio.google.com/apikey"
    read -p "Press Enter after you've configured your API key..."
fi

# Check API key configuration
if grep -q "YOUR_ACTUAL_API_KEY_HERE" .env; then
    echo "⚠️  Google API key not configured - ADK features will be disabled"
    echo "   The system will run in basic research mode"
    MODE="basic"
else
    echo "✅ Google API key appears to be configured"
    MODE="auto"
fi

# Check if Google ADK is installed
if ! python -c "import google.adk" 2>/dev/null; then
    echo "⚠️  Google ADK not available - installing..."
    pip install google-adk
fi

# Check if OCR data exists
OCR_DIR="/home/chezy/rag_medical/ocr_out"
if [ ! -d "$OCR_DIR" ]; then
    echo "❌ Error: OCR data directory not found: $OCR_DIR"
    echo "   Please run the OCR pipeline first"
    exit 1
fi

# Check for required files
MISSING_FILES=""
for file in "structured_documents.jsonl" "documents_index.json"; do
    if [ ! -f "$OCR_DIR/$file" ]; then
        MISSING_FILES="$MISSING_FILES $file"
    fi
done

if [ -n "$MISSING_FILES" ]; then
    echo "⚠️  Missing OCR files:$MISSING_FILES"
    echo "   These will be generated during the research session if needed"
fi

echo ""
echo "=========================================="
echo "Environment Check Complete"
echo "=========================================="
echo "Mode: $MODE"
echo "OCR Directory: $OCR_DIR"
echo ""

# Ask user if they want to proceed
read -p "Start research session? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Research session cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting RAG System Research Session..."
echo "   This may take 30-60 minutes depending on your hardware"
echo "   Progress will be logged to research_session.log"
echo ""

# Make script executable and run
chmod +x run_research.py
python run_research.py --mode $MODE --verbose

echo ""
echo "=========================================="
echo "Research Session Complete!"
echo "=========================================="
echo "Check the research_results/ directory for reports"
echo "Review research_session.log for detailed execution logs"

