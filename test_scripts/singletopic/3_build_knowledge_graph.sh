#!/bin/bash

echo "========================================"
echo "  SingleTopic Knowledge Graph Builder"
echo "========================================"
echo ""

# Check if corpus exists
if [ ! -f "datasets/SingleTopic/raw/corpus.jsonl" ]; then
    echo "[FAIL] corpus.jsonl not found at datasets/SingleTopic/raw/corpus.jsonl"
    echo "       Please ensure the corpus file exists before building the knowledge graph."
    exit 1
fi

echo "[OK] Found corpus.jsonl"

# Check OpenAI API key
if [ ! -f "openai_api_key.txt" ]; then
    echo "[FAIL] openai_api_key.txt not found in project root."
    echo "       Please create it with your OpenAI API key:"
    echo "       echo \"sk-your-key-here\" > openai_api_key.txt"
    exit 1
fi

echo "[OK] Found OpenAI API key"

# Check if KG already exists
if [ -d "expr/SingleTopic" ]; then
    echo ""
    echo "[WARNING] Knowledge graph already exists at expr/SingleTopic/"
    echo "          Delete it first if you want to rebuild:"
    echo "          rm -rf expr/SingleTopic"
    echo ""
    read -p "Do you want to delete and rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[SKIP] Using existing knowledge graph."
        exit 0
    fi
    echo "[OK] Deleting existing knowledge graph..."
    rm -rf expr/SingleTopic
fi

# Build knowledge graph
echo ""
echo "========================================"
echo "  Building Knowledge Graph"
echo "========================================"
echo ""
echo "  This will take approximately 2-4 hours for ~9K documents."
echo "  Progress will be saved periodically."
echo ""
echo "  Dataset: SingleTopic"
echo "  Corpus: datasets/SingleTopic/raw/corpus.jsonl"
echo "  Output: expr/SingleTopic/"
echo ""
echo "  Started at: $(date)"
echo ""

# Run graph builder
python script_build.py --data_source SingleTopic

# Check if build succeeded
if [ $? -eq 0 ] && [ -d "expr/SingleTopic" ]; then
    echo ""
    echo "========================================"
    echo "  [OK] Knowledge Graph Built Successfully!"
    echo "========================================"
    echo ""
    echo "  Location: expr/SingleTopic/"
    echo "  Completed at: $(date)"
    echo ""
    echo "  Files created:"
    ls -lh expr/SingleTopic/ | grep -E "\.json|\.graphml"
    echo ""
    echo "  Next step: Start the server"
    echo "    cd backend && python server.py --data_source SingleTopic"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  [FAIL] Knowledge Graph Build Failed"
    echo "========================================"
    echo ""
    echo "  Check logs above for error details."
    echo "  Common issues:"
    echo "    - OpenAI API rate limits (wait and retry)"
    echo "    - Insufficient disk space"
    echo "    - Invalid corpus format"
    echo ""
    exit 1
fi
