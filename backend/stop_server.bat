@echo off
echo ================================================================================
echo BiG-RAG Server Shutdown Script
echo ================================================================================
echo.

REM Parse port argument
set PORT=%1
if "%PORT%"=="" set PORT=8001

echo Looking for server on port %PORT%...

set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
    echo Found server process PID: %%a
    echo Stopping server...
    taskkill /PID %%a /F 2>nul
    if %ERRORLEVEL%==0 (
        echo [OK] Server stopped successfully
        set FOUND=1
    ) else (
        echo [ERROR] Failed to stop server
    )
)

if %FOUND%==0 (
    echo No server found on port %PORT%
)

echo.
echo ================================================================================
pause