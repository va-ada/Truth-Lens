@echo off
REM ===================================================================
REM TRUTHLENS - COMPLETE TEST GUIDE
REM ===================================================================

echo.
echo ===================================================================
echo TRUTHLENS SYSTEM - START & TEST COMMANDS
echo ===================================================================
echo.

echo STEP 1: START THE BACKEND SERVER (Run this in Terminal 1)
echo ===================================================================
echo.
echo Windows Command:
echo   cd C:\Users\pksj4\OneDrive\Documents\Ml_MP
echo   .\.venv\Scripts\uvicorn.exe Truth.backend.main:app --host 127.0.0.1 --port 8000 --reload
echo.
echo OR use the batch file:
echo   START_BACKEND.bat
echo.
echo This will:
echo   ✓ Load TruthLens models (ensemble, feature engine)
echo   ✓ Initialize EasyOCR engine (first time: ~5 seconds)
echo   ✓ Preload GloVe embeddings (first time: ~10 seconds)
echo   ✓ Start API on http://127.0.0.1:8000
echo.
echo Wait for message: "Uvicorn running on http://127.0.0.1:8000"
echo.

echo.
echo STEP 2: VERIFY API IS RUNNING (Run this in Terminal 2)
echo ===================================================================
echo.
echo Quick health check:
echo   curl http://127.0.0.1:8000/
echo.
echo Expected response:
echo   {"status": "ok", "message": "Truth API is running."}
echo.

echo.
echo STEP 3: RUN COMPREHENSIVE TEST SUITE
echo ===================================================================
echo.
echo Command:
echo   cd C:\Users\pksj4\OneDrive\Documents\Ml_MP
echo   .\.venv\Scripts\python.exe TEST_SUITE.py
echo.
echo This will automatically test:
echo   ✓ TEST 1: Calendar + Location + Wikipedia (expects CONFLICT - fake news)
echo   ✓ TEST 2: Web Search + Valid Dates (expects VERIFIED - real news)
echo.
echo Output will show:
echo   - ML Prediction (Real/Fake)
echo   - Confidence score
echo   - Explainability (top keywords)
echo   - Factual verification results
echo   - Conflict detection status
echo.

echo.
echo STEP 4: THE TWO TEST CASES
echo ===================================================================
echo.

echo TEST 1: CALENDAR + CONTRADICTION DETECTION
echo -----------
echo Input Text:
echo "Breaking News: Scientists announced on February 30, 2026 that they 
echo discovered a new species of giant panda in the Amazon rainforest..."
echo.
echo What it tests:
echo   ✓ Calendar Validation: February 30 is INVALID (conflict)
echo   ✓ Location Verification: Amazon exists (verified)
echo   ✓ Wikipedia Check: Pandas don't live in South America (conflict)
echo   ✓ Entity Recognition: Oxford, Amazon masked
echo   ✓ Conflict Detection: Should trigger WARNING
echo.
echo Expected Result:
echo   Prediction: FAKE NEWS
echo   Confidence: High (due to conflicts)
echo   Conflicts: Yes (multiple contradictions)
echo.

echo.
echo TEST 2: WEB SEARCH + VALID DATES + INSTITUTIONS
echo -----------
echo Input Text:
echo "According to a study published in Nature on April 3, 2026, artificial 
echo intelligence has achieved a breakthrough in quantum computing. The 
echo research was conducted at MIT and Stanford University..."
echo.
echo What it tests:
echo   ✓ Date Validation: April 3, 2026 is VALID (recent date)
echo   ✓ Web Search: Searches for quantum computing breakthrough
echo   ✓ Institution Check: MIT, Stanford are real universities
echo   ✓ Publication Validation: Nature is real journal
echo   ✓ ML Confidence: High for real news with verifications
echo.
echo Expected Result:
echo   Prediction: REAL NEWS
echo   Confidence: High
echo   Verifications: Mostly passing
echo.

echo.
echo SYSTEM COMPONENTS BEING TESTED
echo ===================================================================
echo.
echo 1. ML ENGINE (TruthLens)
echo    - Stacking Ensemble (SVM, LR, RF)
echo    - 415D feature vectors (TF-IDF + GloVe + Stylometric)
echo    - Entity masking (PERSON, ORG, LOC tokens)
echo.
echo 2. RAG PIPELINE
echo    - Calendar Validation: Checks valid date ranges
echo    - Web Search: DuckDuckGo cross-reference
echo    - Wikipedia: Entity verification + myths
echo    - Location: OpenStreetMap validation
echo.
echo 3. EXPLAINABILITY
echo    - TF-IDF keyword highlighting
echo    - Confidence scoring
echo    - Factual analysis breakdown
echo.
echo 4. FILE HANDLING
echo    - Text input ✓
echo    - PDF parsing (PyPDF2) ✓
echo    - Image OCR (EasyOCR) ✓
echo    - DOCX parsing (python-docx) ✓
echo.

echo.
echo TESTING MANUALLY WITH CURL
echo ===================================================================
echo.
echo Test with text:
echo   curl -X POST http://127.0.0.1:8000/api/analyze ^
echo     -F "text=This is a test article about breaking news."
echo.
echo Expected response (JSON):
echo   {
echo     "prediction": "Real News" or "Fake News",
echo     "confidence": 0.85,
echo     "processing_time": 3.2,
echo     "explainability": [...],
echo     "factual_analysis": [...],
echo     "conflict_detected": false
echo   }
echo.

echo.
echo TROUBLESHOOTING
echo ===================================================================
echo.
echo If API won't start:
echo   1. Check port 8000 is free: netstat -ano ^| findstr :8000
echo   2. Ensure venv is activated
echo   3. Verify TruthLens models exist:
echo      ls TruthLens\models\
echo.
echo If tests timeout:
echo   1. First request takes longer (GloVe loading)
echo   2. Increase timeout in TEST_SUITE.py to 60 seconds
echo   3. Check backend logs for errors
echo.
echo If API returns errors:
echo   1. Check backend console output
echo   2. Verify all dependencies: pip check
echo   3. Ensure spaCy model: python -m spacy download en_core_web_sm
echo.

echo.
echo ===================================================================
echo READY TO TEST! Follow the steps above.
echo ===================================================================
echo.

pause
