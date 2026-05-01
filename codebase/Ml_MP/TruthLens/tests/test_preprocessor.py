"""
Unit tests for TruthLens preprocessor module.

Pass 1 (synthetic data):  All tests here use controlled, hardcoded inputs.
Pass 2 (real-world):       Run manually with real dataset via preprocess_dataframe().

Notes:
- spaCy and NLTK are loaded once at module level to avoid per-test overhead.
- If the spaCy model is unavailable, NER/lemmatize tests are skipped automatically.
- ENABLE_GLOVE is forced False so no GloVe download is attempted during testing.
"""

import sys
import os

# ── Path setup: add TruthLens root so `config` and `src` are importable ──────
_TRUTHLENS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TRUTHLENS_ROOT not in sys.path:
    sys.path.insert(0, _TRUTHLENS_ROOT)

# ── Patch config BEFORE importing preprocessor ───────────────────────────────
import config
config.ENABLE_GLOVE = False
config.ENABLE_ENTITY_MASKING = True

# ── Now import the module under test ─────────────────────────────────────────
import importlib
import src.preprocessor as preprocessor_mod

# Re-export the four functions we test
from src.preprocessor import (
    clean_text,
    mask_entities,
    remove_stopwords_and_lemmatize,
    preprocess_dataframe,
)

import pandas as pd
import pytest

# ── Module-level spaCy availability check ─────────────────────────────────────
_SPACY_AVAILABLE = False
_SPACY_SKIP_REASON = "spaCy model not available"

try:
    import spacy
    spacy.load(config.SPACY_MODEL, disable=["parser"])
    _SPACY_AVAILABLE = True
except Exception as e:
    _SPACY_SKIP_REASON = f"spaCy model '{config.SPACY_MODEL}' not loadable: {e}"

needs_spacy = pytest.mark.skipif(not _SPACY_AVAILABLE, reason=_SPACY_SKIP_REASON)


# ═════════════════════════════════════════════════════════════════════════════
# clean_text() tests  (no spaCy — pure regex, always run)
# ═════════════════════════════════════════════════════════════════════════════

def test_clean_text_removes_urls():
    """URLs (http/https/www) must be stripped."""
    result = clean_text("Visit http://example.com today")
    assert "http" not in result, f"Expected no 'http' in output, got: {result!r}"


def test_clean_text_removes_html():
    """HTML tags must be stripped; readable text must survive."""
    result = clean_text("<p>Hello world</p>")
    assert "<p>" not in result, f"Expected no '<p>' in output, got: {result!r}"
    assert "hello" in result.lower() or "world" in result.lower(), (
        f"Expected 'hello' or 'world' to survive cleaning, got: {result!r}"
    )


def test_clean_text_removes_emails():
    """Email addresses must be stripped (no '@' left behind)."""
    result = clean_text("Contact user@domain.com please")
    assert "@" not in result, f"Expected no '@' in output, got: {result!r}"


def test_clean_text_truncates_long_text():
    """
    Text longer than MAX_TEXT_LENGTH (5000) must be truncated.
    We allow a small margin (5100) in case clean_text itself pads slightly,
    but in practice truncation should yield exactly <= MAX_TEXT_LENGTH.
    """
    long_input = "a" * 6000
    result = clean_text(long_input)
    assert len(result) <= 5100, (
        f"Expected truncated output (<= 5100 chars), got length {len(result)}"
    )


def test_clean_text_empty_string():
    """Empty string input must return empty string (no crash, no garbage)."""
    result = clean_text("")
    assert result == "" or result.strip() == "", (
        f"Expected '' for empty input, got: {result!r}"
    )


def test_clean_text_removes_special_chars():
    """Special characters like @#$%^&*() must be removed."""
    result = clean_text("Hello!!! @#$%^&*()")
    for bad_char in ["@", "#", "$", "%"]:
        assert bad_char not in result, (
            f"Expected {bad_char!r} to be removed, but output was: {result!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# mask_entities() tests
# ═════════════════════════════════════════════════════════════════════════════

@needs_spacy
def test_mask_entities_known_source():
    """
    'Reuters' must be masked — either as [SOURCE] (regex path) or [ORG] (spaCy NER path).

    The supplementary regex replaces 'Reuters' → '[SOURCE]', but spaCy NER may tag it
    as ORG *first*, replacing the literal text 'Reuters' with '[ORG]' before the regex
    runs. In that case the regex sees no 'Reuters' and produces '[ORG]' in the output.
    Both outcomes are correct — the entity bias is removed either way.
    """
    result = mask_entities("Reuters confirmed the story today")
    # Accept either outcome: spaCy-first → [ORG], regex-first → [SOURCE]
    assert "[SOURCE]" in result or "[ORG]" in result, (
        f"Expected '[SOURCE]' or '[ORG]' in output for 'Reuters', got: {result!r}"
    )
    # The raw word 'reuters' must no longer appear
    assert "reuters" not in result.lower(), (
        f"Expected 'reuters' to be fully masked, but got: {result!r}"
    )


@needs_spacy
def test_mask_entities_empty():
    """Empty string must return empty string without crashing."""
    result = mask_entities("")
    assert result == "", f"Expected '' for empty input, got: {result!r}"


@needs_spacy
def test_mask_entities_unicode_no_crash():
    """Unicode / non-ASCII text must not raise an exception."""
    try:
        result = mask_entities("これはフェイクニュースです")
        assert isinstance(result, str), "Expected a string return value for unicode input"
    except Exception as e:
        pytest.fail(f"mask_entities() raised an exception on unicode input: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# remove_stopwords_and_lemmatize() tests
# ═════════════════════════════════════════════════════════════════════════════

@needs_spacy
def test_lemmatize_removes_stopwords():
    """
    'the' is a stopword and must be filtered out.
    Output should be shorter (fewer tokens) than the input.
    """
    text = "the cats are running quickly"
    result = remove_stopwords_and_lemmatize(text)
    # 'the' is a classic stopword — must not appear as a standalone token
    tokens = result.lower().split()
    assert "the" not in tokens, (
        f"Expected 'the' to be removed as a stopword, got tokens: {tokens}"
    )
    # Sanity: output should be non-empty (real content words survive)
    assert len(result.strip()) > 0, "Expected non-empty output for non-trivial input"


@needs_spacy
def test_lemmatize_empty():
    """Empty string must return empty string without crashing."""
    result = remove_stopwords_and_lemmatize("")
    assert result == "", f"Expected '' for empty input, got: {result!r}"


# ═════════════════════════════════════════════════════════════════════════════
# preprocess_dataframe() tests
# ═════════════════════════════════════════════════════════════════════════════

@needs_spacy
def test_preprocess_dataframe_output_columns():
    """
    preprocess_dataframe() must produce all four expected columns:
    text, cleaned_text, masked_text, processed_text — plus the original label.
    """
    df = pd.DataFrame({
        "text": [
            "Scientists at NASA discovered new planets beyond our solar system.",
            "Breaking: President signs historic climate bill at the White House.",
            "Local team wins championship after a thrilling final game.",
        ],
        "label": [1, 0, 1],
    })

    result = preprocess_dataframe(df, enable_masking=config.ENABLE_ENTITY_MASKING)

    required_columns = {"text", "cleaned_text", "masked_text", "processed_text", "label"}
    missing = required_columns - set(result.columns)
    assert not missing, f"Output DataFrame is missing columns: {missing}"


@needs_spacy
def test_preprocess_dataframe_no_crash_on_short_text():
    """
    A single-row DataFrame with minimal text must not crash and must
    produce at least one output row.
    """
    df = pd.DataFrame({
        "text": ["Hello world"],
        "label": [0],
    })

    try:
        result = preprocess_dataframe(df, enable_masking=False)
        assert len(result) >= 1, (
            f"Expected at least 1 output row, got {len(result)}"
        )
    except Exception as e:
        pytest.fail(f"preprocess_dataframe() crashed on short text: {e}")
