"""Unit tests for the small, isolated backend helpers introduced in Phase 1+4.

Covers:
  - cache_get / cache_set TTL semantics
  - extract_source_credibility URL→tier resolution
  - verify_google_factcheck request shape (mocked)

The full /api/analyze endpoint is integration-tested separately because it
requires the trained ensemble to be loaded.
"""

from __future__ import annotations

import os
import sys
import time
from unittest import mock

import pytest

# Backend dir on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND_DIR)


# ────────────────────────────────────────────────────────────────────────────
# We have to import the backend module carefully — at import time it tries to
# load the TruthLens ensemble and the EasyOCR engine. Both are slow / heavy.
# We patch joblib.load and torch.cuda.is_available so import is fast.
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def backend():
    with mock.patch("joblib.load", side_effect=Exception("test bypass")), \
         mock.patch.dict(os.environ, {"TRUTHLENS_TEST": "1"}):
        # Re-import every test session because module-level code runs once.
        if "main" in sys.modules:
            del sys.modules["main"]
        import main  # type: ignore
        yield main


# ── cache_get / cache_set TTL ───────────────────────────────────────────────
def test_cache_set_then_get_returns_value(backend):
    backend.RAG_CACHE.clear()
    backend.cache_set("wikipedia", "key1",
                       {"check_type": "Wikipedia Link", "result": "ok",
                        "status": "verified", "url": None})
    got = backend.cache_get("wikipedia", "key1")
    assert got is not None
    assert got["status"] == "verified"
    assert "_ts" not in got, "cache_get should strip the bookkeeping timestamp"


def test_cache_get_returns_none_when_expired(backend, monkeypatch):
    backend.RAG_CACHE.clear()
    backend.cache_set("wikipedia", "stale",
                       {"check_type": "Wikipedia Link", "result": "old",
                        "status": "unknown", "url": None})
    # Force the entry's timestamp to look ancient
    backend.RAG_CACHE["wikipedia"]["stale"]["_ts"] = time.time() - (
        backend.CACHE_TTL_SECONDS + 100
    )
    assert backend.cache_get("wikipedia", "stale") is None
    # Stale entry should have been pruned
    assert "stale" not in backend.RAG_CACHE.get("wikipedia", {})


def test_cache_get_legacy_entry_without_ts_treated_as_fresh(backend):
    backend.RAG_CACHE.clear()
    backend.RAG_CACHE.setdefault("wikipedia", {})["legacy"] = {
        "check_type": "Wikipedia Link", "result": "x",
        "status": "verified", "url": None,
    }
    got = backend.cache_get("wikipedia", "legacy")
    assert got is not None
    assert got["status"] == "verified"


# ── extract_source_credibility ──────────────────────────────────────────────
def test_source_credibility_no_url_returns_none(backend):
    assert backend.extract_source_credibility("plain text with no link") is None


def test_source_credibility_tier1_domain(backend):
    cred = backend.extract_source_credibility(
        "From the article at https://www.reuters.com/article/foo bar"
    )
    assert cred is not None
    assert cred["tier"] == "tier1"
    assert cred["score"] >= 0.85
    assert "reuters" in cred["domain"]


def test_source_credibility_factchecker_domain(backend):
    cred = backend.extract_source_credibility(
        "see https://www.snopes.com/fact-check/example for review"
    )
    assert cred is not None
    assert cred["tier"] == "factcheck"


def test_source_credibility_unranked_domain(backend):
    cred = backend.extract_source_credibility("https://random-blog.example/post")
    assert cred is not None
    assert cred["tier"] == "unranked"
    assert cred["score"] == pytest.approx(0.4, rel=1e-3)


# ── verify_google_factcheck (mock requests) ─────────────────────────────────
def test_google_factcheck_returns_unknown_on_no_results(backend):
    with mock.patch.object(backend, "requests") as fake_req:
        fake_resp = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"claims": []}
        fake_req.get.return_value = fake_resp
        result = backend.verify_google_factcheck("This is a sample claim that nobody fact-checked.")
        assert result.check_type == "Google Fact Check"
        assert result.status == "unknown"


def test_google_factcheck_flags_false_rating(backend):
    with mock.patch.object(backend, "requests") as fake_req:
        fake_resp = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"claims": [{
            "claimReview": [{
                "textualRating": "False",
                "publisher": {"name": "Snopes"},
                "url": "https://www.snopes.com/example",
            }]
        }]}
        fake_req.get.return_value = fake_resp
        result = backend.verify_google_factcheck("a claim that was rated false")
        assert result.status == "conflict"
        assert "Snopes" in result.result


def test_google_factcheck_handles_rate_limit(backend):
    with mock.patch.object(backend, "requests") as fake_req:
        fake_resp = mock.Mock()
        fake_resp.status_code = 429
        fake_req.get.return_value = fake_resp
        result = backend.verify_google_factcheck("anything")
        assert result.status == "unknown"
        assert "rate" in result.result.lower()
