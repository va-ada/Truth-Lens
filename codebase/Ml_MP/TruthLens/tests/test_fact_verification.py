"""
Unit tests for TruthLens fact-verification pipeline.
Tests claim_detector.py and fact_checker.py (temporal verification only —
Wikipedia/web tests are integration tests that require network).
"""

import os
import sys

TRUTHLENS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRUTHLENS_DIR)

import pytest
from src.claim_detector import classify_input
from src.fact_checker import verify_temporal, _word_overlap_similarity, _parse_date, aggregate_evidence


# ===========================================================================
# Claim Detector — classify_input
# ===========================================================================

class TestClassifyInput:
    """Tests for claim_detector.classify_input()."""

    def test_temporal_today(self):
        r = classify_input("Is today April 11th 2026?")
        assert r["type"] == "temporal"
        assert r["has_temporal"] is True

    def test_temporal_yesterday(self):
        r = classify_input("Was yesterday Thursday?")
        assert r["type"] == "temporal"

    def test_question(self):
        r = classify_input("Did NASA find water on Mars?")
        assert r["type"] in ("question", "temporal")  # may detect date words
        assert r["has_question"] is True

    def test_opinion(self):
        r = classify_input("I think the government is hiding something")
        assert r["type"] == "opinion"
        assert r["has_opinion"] is True

    def test_claim_with_numbers(self):
        r = classify_input("The GDP of India is 3.7 trillion dollars")
        assert r["type"] == "claim"
        assert r["has_numbers"] is True
        assert len(r["extracted_numbers"]) >= 1

    def test_claim_with_assertion(self):
        r = classify_input("India landed on the moon in 2023")
        assert r["type"] == "claim"
        assert r["has_assertion"] is True

    def test_article_long_text(self):
        long_text = "The government announced new policy changes. " * 20
        r = classify_input(long_text)
        assert r["type"] == "article"
        assert r["word_count"] > 50

    def test_empty_input(self):
        r = classify_input("")
        assert r["type"] == "claim"
        assert r["word_count"] == 0
        assert r["claim_text"] == ""

    def test_none_input(self):
        r = classify_input(None)
        assert r["type"] == "claim"

    def test_claim_text_strips_question_words(self):
        r = classify_input("Did NASA discover water?")
        assert not r["claim_text"].lower().startswith("did")


# ===========================================================================
# Temporal Verification — verify_temporal
# ===========================================================================

class TestVerifyTemporal:
    """Tests for fact_checker.verify_temporal() — uses real system clock."""

    def test_what_day_today(self):
        r = verify_temporal("What day is it today?")
        assert r["verdict"] == "Informational"
        assert r["confidence"] > 0.9

    def test_year_check_correct(self):
        from datetime import datetime
        year = datetime.now().year
        r = verify_temporal(f"Is it {year}?")
        assert r["verdict"] == "Verified"

    def test_year_check_wrong(self):
        r = verify_temporal("Is it 1999?")
        assert r["verdict"] == "False"

    def test_unverifiable_no_date(self):
        r = verify_temporal("The sky is blue")
        assert r["verdict"] == "Unverifiable"

    def test_extracted_dates_fallback(self):
        r = verify_temporal("Something about dates", extracted_dates=["April 11"])
        assert r["verdict"] == "Informational"
        assert "April 11" in r["evidence"]


# ===========================================================================
# Date Parsing — _parse_date
# ===========================================================================

class TestParseDate:
    """Tests for fact_checker._parse_date()."""

    def test_month_day_year(self):
        from datetime import date
        result = _parse_date("April 11th 2026")
        assert result == date(2026, 4, 11)

    def test_day_month_year(self):
        from datetime import date
        result = _parse_date("11 April 2026")
        assert result == date(2026, 4, 11)

    def test_month_day_no_year(self):
        from datetime import datetime, date
        result = _parse_date("April 11")
        assert result is not None
        assert result.month == 4
        assert result.day == 11

    def test_unparseable(self):
        result = _parse_date("not a date at all")
        assert result is None

    def test_abbreviation(self):
        from datetime import date
        result = _parse_date("Jan 1 2026")
        assert result == date(2026, 1, 1)


# ===========================================================================
# Similarity — _word_overlap_similarity
# ===========================================================================

class TestWordOverlapSimilarity:
    """Tests for fact_checker._word_overlap_similarity()."""

    def test_identical(self):
        sim = _word_overlap_similarity("India moon landing", "India moon landing")
        assert sim > 0.9

    def test_no_overlap(self):
        sim = _word_overlap_similarity("cats dogs", "quantum physics")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = _word_overlap_similarity(
            "India moon landing",
            "India successfully completed a moon landing mission"
        )
        assert 0.0 < sim < 1.0

    def test_empty_string(self):
        assert _word_overlap_similarity("", "hello") == 0.0
        assert _word_overlap_similarity("hello", "") == 0.0

    def test_stopwords_filtered(self):
        # Only stopwords → no meaningful overlap
        sim = _word_overlap_similarity("the is a", "the is a")
        assert sim == 0.0


# ===========================================================================
# Evidence Aggregation — aggregate_evidence
# ===========================================================================

class TestAggregateEvidence:
    """Tests for fact_checker.aggregate_evidence()."""

    def test_no_evidence(self):
        result = aggregate_evidence()
        assert result["final_verdict"] == "Cannot Verify"
        assert result["confidence"] == 0.0

    def test_temporal_verified(self):
        temporal = {"verdict": "Verified", "confidence": 0.95, "source": "Temporal Check", "evidence": "Correct."}
        result = aggregate_evidence(temporal_result=temporal)
        assert result["final_verdict"] == "Verified"
        assert result["confidence"] > 0.8

    def test_temporal_false(self):
        temporal = {"verdict": "False", "confidence": 0.95, "source": "Temporal Check", "evidence": "Wrong."}
        result = aggregate_evidence(temporal_result=temporal)
        assert result["final_verdict"] == "False"

    def test_wikipedia_supported(self):
        wiki = {"verdict": "Supported", "confidence": 0.7, "source": "Wikipedia", "evidence": "Found."}
        result = aggregate_evidence(wikipedia_result=wiki)
        assert result["final_verdict"] in ("Verified", "Likely True")

    def test_mixed_signals(self):
        wiki = {"verdict": "Supported", "confidence": 0.6, "source": "Wikipedia", "evidence": "Found."}
        web = {"verdict": "Contradicted", "confidence": 0.6, "source": "Web Search", "evidence": "Debunked."}
        result = aggregate_evidence(wikipedia_result=wiki, web_result=web)
        assert result["final_verdict"] in ("Uncertain", "Likely True", "Likely False")

    def test_confidence_capped(self):
        temporal = {"verdict": "Verified", "confidence": 0.99, "source": "Temporal Check", "evidence": "Yes."}
        wiki = {"verdict": "Supported", "confidence": 0.99, "source": "Wikipedia", "evidence": "Yes."}
        result = aggregate_evidence(temporal_result=temporal, wikipedia_result=wiki)
        assert result["confidence"] <= 0.95
