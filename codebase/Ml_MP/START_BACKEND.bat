#!/bin/bash
# TRUTHLENS - STARTUP & TEST GUIDE
# ==================================
# This guide shows how to start the system and run comprehensive tests

@echo off
REM ===================================================================
REM PART 1: START THE BACKEND API
REM ===================================================================

echo.
echo ===================================================================
echo TRUTHLENS - BACKEND STARTUP
echo ===================================================================
echo.
echo Starting FastAPI backend server...
echo Server will run on: http://127.0.0.1:8000
echo.
echo Press CTRL+C to stop the server.
echo.

cd /d C:\Users\pksj4\OneDrive\Documents\Ml_MP

echo Starting backend...
.\.venv\Scripts\uvicorn.exe Truth.backend.main:app --host 127.0.0.1 --port 8000 --reload

REM After breaking here (CTRL+C), run the test suite

pause
