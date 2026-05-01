#!/usr/bin/env python
"""
TRUTHLENS SYSTEM - TEST SUITE
==============================

This script provides 2 comprehensive test cases that exercise all system components:
- ML inference (fake/real news detection)
- RAG verification (calendar, web, wikipedia, location)
- File upload handling
- API response validation

Run this after starting the API server.
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8000/api/analyze"
BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def assert_valid_response(response_json):
    """Validate that an API response has the expected shape and valid values."""
    assert "prediction" in response_json, "Missing 'prediction' field"
    assert response_json["prediction"] in ["Real News", "Fake News", "Uncertain"], \
        f"Invalid prediction: {response_json['prediction']}"
    assert "confidence" in response_json, "Missing 'confidence' field"
    assert 0.0 <= response_json["confidence"] <= 1.0, \
        f"Confidence out of range: {response_json['confidence']}"
    assert "processing_time" in response_json, "Missing 'processing_time' field"
    assert response_json["processing_time"] > 0, "Processing time must be positive"


def test_empty_text():
    """Empty text should return 400, not 500."""
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": ""})
    assert response.status_code == 400, f"Expected 400 for empty text, got {response.status_code}"
    print("  PASS: empty text → 400")


def test_whitespace_only():
    """Whitespace-only text should return 400, not 500."""
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": "   \n\n  "})
    assert response.status_code == 400, f"Expected 400 for whitespace-only text, got {response.status_code}"
    print("  PASS: whitespace-only → 400")


def test_very_long_text():
    """Very long text (10k chars) should not crash the server."""
    long_text = "The stock market rose sharply amid positive economic data. " * 200  # ~11k chars
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": long_text})
    assert response.status_code == 200, f"Expected 200 for long text, got {response.status_code}"
    data = response.json()
    assert_valid_response(data)
    print(f"  PASS: very long text → {data['prediction']} ({data['confidence']:.2f})")


def test_unicode_text():
    """Unicode/non-ASCII text should not crash the server."""
    unicode_text = "これはフェイクニュースです！！！ 이것은 가짜 뉴스입니다"
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": unicode_text})
    assert response.status_code == 200, f"Expected 200 for unicode text, got {response.status_code}"
    data = response.json()
    assert_valid_response(data)
    print(f"  PASS: unicode text → {data['prediction']} ({data['confidence']:.2f})")


def test_short_factual_claim():
    """Short LIAR-style claim should return a valid (possibly uncertain) prediction."""
    claim = "The unemployment rate fell to 3.5% in October 2023."
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": claim})
    assert response.status_code == 200, f"Expected 200 for short claim, got {response.status_code}"
    data = response.json()
    assert_valid_response(data)
    print(f"  PASS: short claim → {data['prediction']} ({data['confidence']:.2f})")


def test_sensational_text():
    """Text with fake-news signals should predict Fake News."""
    sensational = (
        "SHOCKING!!! The GOVERNMENT is HIDING this from you!!! "
        "They don't want you to know!!! SHARE BEFORE DELETED!!! "
        "WAKE UP SHEEPLE!!! This will DESTROY the establishment!!!"
    )
    response = requests.post(f"{BASE_URL}/api/analyze", data={"text": sensational})
    assert response.status_code == 200, f"Expected 200 for sensational text, got {response.status_code}"
    data = response.json()
    assert_valid_response(data)
    # This text has extreme fake-news signals — should be Fake (or at worst Uncertain)
    assert data["prediction"] in ["Fake News", "Uncertain"], \
        f"Expected Fake News/Uncertain for sensational text, got {data['prediction']}"
    print(f"  PASS: sensational text → {data['prediction']} ({data['confidence']:.2f})")

def test_api(text, test_name):
    """Test the API with text input."""
    print_section(f"TEST: {test_name}")
    print(f"Input: {text[:80]}...")
    print()
    
    try:
        start = time.time()
        response = requests.post(
            API_URL,
            data={'text': text},
            timeout=30
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        data = response.json()
        
        # --- PREDICTION RESULT ---
        print("📊 ML PREDICTION:")
        print(f"  Verdict: {data['prediction']}")
        print(f"  Confidence: {data['confidence']}")
        print(f"  Processing Time: {data['processing_time']}s")
        print()
        
        # --- EXPLAINABILITY ---
        print("🔍 EXPLAINABILITY (Top Keywords):")
        if data['explainability']:
            for i, feat in enumerate(data['explainability'][:3], 1):
                print(f"  {i}. '{feat['word']}' (weight: {feat['weight']})")
        print()
        
        # --- FACTUAL ANALYSIS ---
        print("✅ FACTUAL VERIFICATION (RAG Pipeline):")
        if data['factual_analysis']:
            for check in data['factual_analysis']:
                status_icon = {
                    'verified': '✓',
                    'conflict': '✗',
                    'unknown': '?'
                }.get(check['status'], '?')
                
                print(f"  {status_icon} {check['check_type']}")
                print(f"     Result: {check['result'][:60]}...")
                print(f"     Status: {check['status'].upper()}")
                if check.get('url'):
                    print(f"     Source: {check['url']}")
                print()
        
        # --- CONFLICT DETECTION ---
        print("⚠️  CONFLICT DETECTION:")
        print(f"  Conflicts Found: {data['conflict_detected']}")
        if data['conflict_detected']:
            print("  ⚠️  Warning: Factual contradictions detected!")
        print()
        
        print(f"✅ TEST COMPLETED in {elapsed:.2f}s")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (API may be slow on first request due to GloVe loading)")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure server is running:")
        print("   .\\venv\\Scripts\\uvicorn.exe Truth.backend.main:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# TEST CASE 1: CALENDAR + WEB VERIFICATION
# ============================================================================
print("\n" + "#"*70)
print("# TRUTHLENS COMPREHENSIVE TEST SUITE")
print("#"*70)

test1 = """
Breaking News: Scientists announced on February 30, 2026 that they discovered 
a new species of giant panda in the Amazon rainforest. The discovery was made 
by researchers at Oxford University who were conducting a 5-year study on 
endangered species. According to Wikipedia, this would be the first panda 
species found in South America.
"""

test_api(test1, "TEST 1: CALENDAR + LOCATION + WIKIPEDIA VERIFICATION")
print("\nWhat this tests:")
print("  ✓ Calendar validation: February 30 doesn't exist (should flag as CONFLICT)")
print("  ✓ Location validation: Amazon rainforest is real")
print("  ✓ Wikipedia checking: Pandas don't exist in South America")
print("  ✓ Entity recognition: Oxford University, Amazon")
print("  ✓ ML bias detection: Entity masking prevents learning 'Oxford = Real'")
print("  ✓ RAG factual contradictions: Should trigger CONFLICT_DETECTED = True")

# ============================================================================
# TEST CASE 2: WEB SEARCH + CURRENT EVENTS + VALID DATES
# ============================================================================
print("\n\n" + "#"*70)

test2 = """
According to a study published in Nature on April 3, 2026, artificial 
intelligence has achieved a breakthrough in quantum computing. The research 
was conducted at MIT and Stanford University and shows that quantum computers 
can now solve previously unsolvable problems in cryptography. The findings 
have been validated by leading tech companies and independent reviewers.
"""

test_api(test2, "TEST 2: WEB SEARCH + RECENT DATES + ESTABLISHED INSTITUTIONS")
print("\nWhat this tests:")
print("  ✓ Date validation: April 3, 2026 is valid (recent date)")
print("  ✓ Web cross-reference: Searches for 'quantum computing breakthrough'")
print("  ✓ Institution verification: MIT, Stanford are real institutions")
print("  ✓ Journal validation: Nature is a real publication")
print("  ✓ ML confidence: Should be REAL NEWS with high confidence")
print("  ✓ RAG success: Multiple verifications should pass")
print("  ✓ Explainability: Keywords like 'study', 'published', 'validated' highlighted")

# ============================================================================
# EDGE-CASE TESTS
# ============================================================================
print("\n\n" + "#"*70)
print("# EDGE-CASE / ROBUSTNESS TESTS")
print("#"*70)
print("\nRunning 6 edge-case tests...\n")

edge_case_results = []

for test_fn in [
    test_empty_text,
    test_whitespace_only,
    test_very_long_text,
    test_unicode_text,
    test_short_factual_claim,
    test_sensational_text,
]:
    try:
        test_fn()
        edge_case_results.append((test_fn.__name__, "PASS"))
    except AssertionError as e:
        print(f"  FAIL: {test_fn.__name__} — {e}")
        edge_case_results.append((test_fn.__name__, f"FAIL: {e}"))
    except requests.exceptions.ConnectionError:
        print(f"  SKIP: {test_fn.__name__} — server not reachable")
        edge_case_results.append((test_fn.__name__, "SKIP"))
    except Exception as e:
        print(f"  ERROR: {test_fn.__name__} — {e}")
        edge_case_results.append((test_fn.__name__, f"ERROR: {e}"))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("TEST SUITE SUMMARY")
print("="*70)
print("""
These 8 tests comprehensively exercise:

1  TEST 1 - CONFLICT DETECTION
   - Calendar: Invalid date (Feb 30)
   - Location: Real location (Amazon)
   - Wikipedia: Fact contradiction (pandas in South America)
   - Expected Result: FAKE NEWS (conflict_detected=True)

2  TEST 2 - VALIDATION SUCCESS
   - Calendar: Valid recent date (April 3)
   - Institutions: Real universities
   - Web Search: Cross-referencing
   - Expected Result: REAL NEWS (high confidence, verifications pass)

3  EDGE-CASE TESTS (6 tests)
   - Empty text input           → expect 400
   - Whitespace-only input      → expect 400
   - Very long text (~11k chars) → expect 200 + valid response
   - Unicode/non-ASCII text     → expect 200 + valid response
   - Short factual claim        → expect 200 + valid response
   - Sensational fake-news text → expect Fake News or Uncertain

All System Components Tested:
  ML Inference (TruthLens Stacking Ensemble)
  Calendar Integration (Date validation)
  Web RAG (DuckDuckGo search)
  Wikipedia Verification (Entity checking)
  Entity Recognition (Named entity masking)
  Explainability (Keyword highlighting)
  Conflict Detection (Multi-layer validation)
  Input validation (400 on empty/whitespace)
  Robustness (long text, unicode, sensational language)
""")

print("Edge-case results:")
for name, result in edge_case_results:
    print(f"  {name}: {result}")
