# TRUTHLENS - QUICK START COMMANDS
# ==================================================

## STEP 1: START BACKEND API (Terminal 1)

cd C:\Users\pksj4\OneDrive\Documents\Ml_MP
.\.venv\Scripts\uvicorn.exe Truth.backend.main:app --host 127.0.0.1 --port 8000 --reload

# Wait for message: "Uvicorn running on http://127.0.0.1:8000"
# Leave this terminal running


## STEP 2: VERIFY API IS RUNNING (Terminal 2)

curl http://127.0.0.1:8000/

# Expected response:
# {"status": "ok", "message": "Truth API is running."}


## STEP 3: RUN TEST SUITE (Terminal 2)

cd C:\Users\pksj4\OneDrive\Documents\Ml_MP
.\.venv\Scripts\python.exe TEST_SUITE.py

# This runs BOTH test cases automatically


## STEP 4: OR RUN INDIVIDUAL TESTS WITH CURL

# Test case 1: Calendar conflict
curl -X POST http://127.0.0.1:8000/api/analyze ^
  -F "text=Breaking News: Scientists announced on February 30, 2026 that they discovered a new species of giant panda in the Amazon rainforest. The discovery was made by researchers at Oxford University. According to Wikipedia, this would be the first panda species found in South America."

# Test case 2: Valid dates + institutions  
curl -X POST http://127.0.0.1:8000/api/analyze ^
  -F "text=According to a study published in Nature on April 3, 2026, artificial intelligence has achieved a breakthrough in quantum computing. The research was conducted at MIT and Stanford University and shows that quantum computers can now solve previously unsolvable problems in cryptography."


# ==================================================
# TWO COMPREHENSIVE TEST CASES
# ==================================================

## TEST 1: CALENDAR + CONFLICT DETECTION
## Expects: FAKE NEWS (conflict_detected=True)

Text Input:
"Breaking News: Scientists announced on February 30, 2026 that they discovered 
a new species of giant panda in the Amazon rainforest. The discovery was made 
by researchers at Oxford University who were conducting a 5-year study on 
endangered species. According to Wikipedia, this would be the first panda 
species found in South America."

What it tests:
  ✓ Calendar validation: February 30 doesn't exist → CONFLICT
  ✓ Location: Amazon rainforest is real → VERIFIED
  ✓ Institution: Oxford University is real → VERIFIED
  ✓ Wikipedia check: Pandas don't live in South America → CONFLICT
  ✓ Entity masking: PERSON (Oxford), ORG (University), LOC (Amazon)
  ✓ Multi-layer verification: Should detect contradictions
  ✓ Conflict override: Conflicts override ML confidence

Expected Output:
  {
    "prediction": "Fake News",
    "confidence": 0.99,
    "conflict_detected": true,
    "factual_analysis": [
      {"check_type": "Calendar Validation", "status": "conflict", ...},
      {"check_type": "Wikipedia Link", "status": "conflict", ...},
      ...
    ]
  }


## TEST 2: WEB SEARCH + VALID DATES + INSTITUTIONS
## Expects: REAL NEWS (high confidence, multiple verifications)

Text Input:
"According to a study published in Nature on April 3, 2026, artificial 
intelligence has achieved a breakthrough in quantum computing. The research 
was conducted at MIT and Stanford University and shows that quantum computers 
can now solve previously unsolvable problems in cryptography. The findings 
have been validated by leading tech companies and independent reviewers."

What it tests:
  ✓ Date validation: April 3, 2026 is VALID (within system date)
  ✓ Web search: DuckDuckGo finds corroborating sources
  ✓ Institution: MIT, Stanford are real universities
  ✓ Journal: Nature is real publication
  ✓ ML confidence: Should be HIGH for legitimate-looking news
  ✓ Multiple verifications: Should pass multiple checks
  ✓ Entity recognition: MIT, Stanford, Nature are entities

Expected Output:
  {
    "prediction": "Real News",
    "confidence": 0.92,
    "conflict_detected": false,
    "factual_analysis": [
      {"check_type": "Calendar Validation", "status": "verified", ...},
      {"check_type": "Web Cross-Reference", "status": "verified", ...},
      {"check_type": "Wikipedia Link", "status": "verified", ...}
    ]
  }


# ==================================================
# SYSTEM COMPONENTS TESTED
# ==================================================

1. ML ENGINE (TruthLens)
   - Stacking Ensemble (SVM, Logistic Regression, Random Forest)
   - 415-dimensional feature vectors
   - Entity masking (prevents dataset bias)
   - Confidence scoring

2. RAG PIPELINE (Retrieval Augmented Generation)
   ✓ Calendar Validation - System calendar validation with leap year handling
   ✓ Web Search - DuckDuckGo integration for fact-checking
   ✓ Wikipedia - Entity verification and myth detection
   ✓ Location Validation - OpenStreetMap API for address verification

3. EXPLAINABILITY
   - TF-IDF keyword highlighting
   - Feature importance visualization
   - Confidence breakdown

4. FILE HANDLING (Ready but not tested in simple cases)
   ✓ Text input
   ✓ PDF parsing (PyPDF2)
   ✓ Image OCR (EasyOCR, GPU-accelerated)
   ✓ DOCX parsing (python-docx)

5. PREPROCESSING
   ✓ Text cleaning
   ✓ Entity masking
   ✓ Lemmatization
   ✓ GloVe embeddings


# ==================================================
# TROUBLESHOOTING
# ==================================================

Backend won't start:
  - Check port 8000: netstat -ano | findstr :8000
  - Clear any processes: taskkill /PID <PID> /F
  - Activate venv: .\.venv\Scripts\Activate.ps1

API timeout on first request:
  - Normal: First request loads GloVe (128 MB) and features
  - Solution: Increase timeout to 60 seconds in TEST_SUITE.py

Test fails with ImportError:
  - Run: pip install -r Truth/backend/requirements.txt
  - Verify spaCy: python -m spacy download en_core_web_sm

Cannot connect to API:
  - Verify backend is running
  - Check firewall allows port 8000
  - Try: curl http://127.0.0.1:8000/ to verify
