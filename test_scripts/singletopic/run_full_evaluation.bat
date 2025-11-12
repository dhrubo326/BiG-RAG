@echo off
REM SingleTopic Full Evaluation Pipeline (Windows Batch Version)

echo ========================================
echo   SingleTopic Full Evaluation Pipeline
echo ========================================
echo.
echo This script will:
echo   1. Build knowledge graph (if not exists)
echo   2. Check if server is running
echo   3. Generate BiG-RAG answers
echo   4. Evaluate results and calculate metrics
echo.
echo Estimated time: 2-4 hours (mostly KG building)
echo.
pause

REM ============================================================================
REM Step 1: Build Knowledge Graph
REM ============================================================================

if not exist "expr\SingleTopic" (
    echo.
    echo ========================================
    echo   Step 1/4: Building Knowledge Graph
    echo ========================================
    echo.

    REM Check if corpus exists
    if not exist "datasets\SingleTopic\raw\corpus.jsonl" (
        echo [FAIL] corpus.jsonl not found
        echo        Please ensure datasets\SingleTopic\raw\corpus.jsonl exists
        pause
        exit /b 1
    )

    REM Check OpenAI API key
    if not exist "openai_api_key.txt" (
        echo [FAIL] openai_api_key.txt not found
        echo        Create it with: echo sk-your-key-here ^> openai_api_key.txt
        pause
        exit /b 1
    )

    echo [OK] Starting knowledge graph build...
    python script_build.py --data_source SingleTopic

    if errorlevel 1 (
        echo.
        echo [FAIL] Knowledge graph build failed
        pause
        exit /b 1
    )
) else (
    echo.
    echo ========================================
    echo   Step 1/4: Knowledge Graph (SKIP)
    echo ========================================
    echo.
    echo [OK] Knowledge graph already exists at expr\SingleTopic\
    echo.
)

REM ============================================================================
REM Step 2: Check if Server is Running
REM ============================================================================

echo.
echo ========================================
echo   Step 2/4: Checking BiG-RAG Server
echo ========================================
echo.

curl -s http://localhost:8001/ >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Server is not running at http://localhost:8001/
    echo.
    echo Please start the server first in a separate Command Prompt:
    echo   cd backend
    echo   python server.py --data_source SingleTopic
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [OK] Server is running

REM Check dataset
for /f "delims=" %%i in ('curl -s http://localhost:8001/ ^| python -c "import json, sys; print(json.load(sys.stdin).get('dataset', 'unknown'))" 2^>nul') do set SERVER_DATASET=%%i

if not "%SERVER_DATASET%"=="SingleTopic" (
    echo.
    echo [WARNING] Server is using dataset '%SERVER_DATASET%' but we expect 'SingleTopic'
    echo           Results may be incorrect!
    echo.
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "%CONTINUE%"=="y" (
        echo [ABORT] Please restart server with correct dataset:
        echo   cd backend ^&^& python server.py --data_source SingleTopic
        pause
        exit /b 1
    )
)

REM ============================================================================
REM Step 3: Generate Answers
REM ============================================================================

echo.
echo ========================================
echo   Step 3/4: Generating BiG-RAG Answers
echo ========================================
echo.

python test_scripts\singletopic\4_generate_answers.py

if errorlevel 1 (
    echo.
    echo [FAIL] Answer generation failed
    pause
    exit /b 1
)

REM ============================================================================
REM Step 4: Evaluate Results
REM ============================================================================

echo.
echo ========================================
echo   Step 4/4: Evaluating Results
echo ========================================
echo.

python test_scripts\singletopic\5_evaluate_results.py

if errorlevel 1 (
    echo.
    echo [FAIL] Evaluation failed
    pause
    exit /b 1
)

REM ============================================================================
REM Done
REM ============================================================================

echo.
echo ========================================
echo   [OK] Full Evaluation Complete!
echo ========================================
echo.
echo Results saved to: datasets\SingleTopic\results\
echo.
echo View evaluation report:
echo   type datasets\SingleTopic\results\evaluation_report.md
echo.
echo Or open in default text editor:
echo   start datasets\SingleTopic\results\evaluation_report.md
echo.
pause
