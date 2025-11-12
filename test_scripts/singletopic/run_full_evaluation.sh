#!/bin/bash

echo "========================================"
echo "  SingleTopic Full Evaluation Pipeline"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Build knowledge graph (if not exists)"
echo "  2. Check if server is running"
echo "  3. Generate BiG-RAG answers"
echo "  4. Evaluate results and calculate metrics"
echo ""
echo "Estimated time: 2-4 hours (mostly KG building)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# ============================================================================
# Step 1: Build Knowledge Graph
# ============================================================================

if [ ! -d "expr/SingleTopic" ]; then
    echo ""
    echo "========================================"
    echo "  Step 1/4: Building Knowledge Graph"
    echo "========================================"
    echo ""

    bash test_scripts/singletopic/3_build_knowledge_graph.sh

    if [ $? -ne 0 ]; then
        echo ""
        echo "[FAIL] Knowledge graph build failed. Exiting."
        exit 1
    fi
else
    echo ""
    echo "========================================"
    echo "  Step 1/4: Knowledge Graph (SKIP)"
    echo "========================================"
    echo ""
    echo "[OK] Knowledge graph already exists at expr/SingleTopic/"
    echo ""
fi

# ============================================================================
# Step 2: Check if Server is Running
# ============================================================================

echo ""
echo "========================================"
echo "  Step 2/4: Checking BiG-RAG Server"
echo "========================================"
echo ""

if curl -s http://localhost:8001/ > /dev/null 2>&1; then
    SERVER_DATASET=$(curl -s http://localhost:8001/ | python -c "import json, sys; print(json.load(sys.stdin).get('dataset', 'unknown'))" 2>/dev/null)

    echo "[OK] Server is running (dataset: $SERVER_DATASET)"

    if [ "$SERVER_DATASET" != "SingleTopic" ]; then
        echo ""
        echo "[WARNING] Server is using dataset '$SERVER_DATASET' but we expect 'SingleTopic'"
        echo "          Results may be incorrect!"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "[ABORT] Please restart server with correct dataset:"
            echo "  cd backend && python server.py --data_source SingleTopic"
            exit 1
        fi
    fi
else
    echo "[FAIL] Server is not running at http://localhost:8001/"
    echo ""
    echo "Please start the server first:"
    echo "  cd backend && python server.py --data_source SingleTopic"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# ============================================================================
# Step 3: Generate Answers
# ============================================================================

echo ""
echo "========================================"
echo "  Step 3/4: Generating BiG-RAG Answers"
echo "========================================"
echo ""

python test_scripts/singletopic/4_generate_answers.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Answer generation failed. Exiting."
    exit 1
fi

# ============================================================================
# Step 4: Evaluate Results
# ============================================================================

echo ""
echo "========================================"
echo "  Step 4/4: Evaluating Results"
echo "========================================"
echo ""

python test_scripts/singletopic/5_evaluate_results.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Evaluation failed. Exiting."
    exit 1
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "========================================"
echo "  [OK] Full Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: datasets/SingleTopic/results/"
echo ""
echo "View evaluation report:"
echo "  cat datasets/SingleTopic/results/evaluation_report.md"
echo ""
echo "Or open in browser:"
echo "  # Windows:"
echo "  start datasets/SingleTopic/results/evaluation_report.md"
echo "  # Linux:"
echo "  xdg-open datasets/SingleTopic/results/evaluation_report.md"
echo "  # macOS:"
echo "  open datasets/SingleTopic/results/evaluation_report.md"
echo ""
