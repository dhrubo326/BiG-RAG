@echo off
echo ================================================================================
echo BiG-RAG Server Startup Script
echo ================================================================================
echo.

REM Check for existing server on port 8001
echo Checking for existing server on port 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo Found existing process PID: %%a
    echo Killing process...
    taskkill /PID %%a /F 2>nul
    timeout /t 2 /nobreak >nul
)

REM Parse command line arguments
set DATA_SOURCE=%1
set PORT=%2

REM Set defaults if not provided
if "%DATA_SOURCE%"=="" set DATA_SOURCE=SingleTopic
if "%PORT%"=="" set PORT=8001

echo.
echo Starting BiG-RAG server with:
echo   Dataset: %DATA_SOURCE%
echo   Port: %PORT%
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

REM Start the server
python server.py --data_source %DATA_SOURCE% --port %PORT%