"""
TruthLens — Claim Detector
=============================
Classifies user input to route it to the correct pipeline:

  - "article"   → long text, route to ML pipeline (TF-IDF + Ensemble)
  - "temporal"   → date/time claim, route to temporal verifier
  - "claim"      → factual assertion, route to web + Wikipedia verification
  - "question"   → factual question, extract claim then verify
  - "opinion"    → subjective text, flag as unverifiable

Uses rule-based heuristics (fast, no ML overhead, no external dependencies).
"""

import re
from datetime import datetime


# ── Patterns ──────────────────────────────────────────────────────────────────

# Question starters
_QUESTION_PATTERN = re.compile(
    r'^\s*(is|are|was|were|did|does|do|has|have|had|can|could|will|would|'
    r'should|shall|may|might|who|what|when|where|why|how|which)\b',
    re.IGNORECASE,
)

# Temporal keywords
_TEMPORAL_KEYWORDS = re.compile(
    r'\b(today|yesterday|tomorrow|last\s+(?:week|month|year)|'
    r'this\s+(?:week|month|year)|current(?:ly)?|right\s+now|'
    r'(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december)\s+\d{1,2}(?:\s*,?\s*\d{4})?|'
    r'\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|'
    r'july|august|september|october|november|december)(?:\s+\d{4})?|'
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    re.IGNORECASE,
)

# Day-of-week patterns
_DAY_PATTERN = re.compile(
    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    re.IGNORECASE,
)

# Numerical claim patterns (number + context suggesting a factual assertion)
_NUMERICAL_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(%|percent|billion|million|trillion|thousand|'
    r'crore|lakh|dollars?|rupees?|pounds?|euros?|km|miles?|kg|tons?)\b',
    re.IGNORECASE,
)

# Strong assertion verbs (signals a factual claim, not just a description)
_ASSERTION_PATTERN = re.compile(
    r'\b(discovered|announced|confirmed|proved|revealed|launched|'
    r'landed|won|lost|passed|signed|banned|approved|rejected|'
    r'invaded|attacked|elected|resigned|died|born|invented|'
    r'broke|set\s+record|first\s+ever|largest|smallest|oldest|newest)\b',
    re.IGNORECASE,
)

# Opinion/subjective markers
_OPINION_PATTERN = re.compile(
    r'\b(i\s+think|i\s+believe|in\s+my\s+opinion|probably|maybe|perhaps|'
    r'i\s+feel|it\s+seems|allegedly|supposedly|rumor|rumour)\b',
    re.IGNORECASE,
)


def classify_input(text):
    """
    Classify user input to determine the best verification pipeline.

    Args:
        text: Raw user input string

    Returns:
        dict with:
            type: "article" | "temporal" | "claim" | "question" | "opinion"
            has_question: bool
            has_temporal: bool
            has_numbers: bool
            has_assertion: bool
            has_opinion: bool
            extracted_dates: list of date-like strings found
            extracted_numbers: list of (number, unit) tuples
            word_count: int
            claim_text: cleaned version of the input for searching
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "type": "claim",
            "has_question": False,
            "has_temporal": False,
            "has_numbers": False,
            "has_assertion": False,
            "has_opinion": False,
            "extracted_dates": [],
            "extracted_numbers": [],
            "word_count": 0,
            "claim_text": "",
        }

    text_clean = text.strip()
    words = text_clean.split()
    word_count = len(words)

    # Extract features
    has_question = bool(_QUESTION_PATTERN.search(text_clean)) or text_clean.rstrip().endswith("?")
    has_temporal = bool(_TEMPORAL_KEYWORDS.search(text_clean)) or bool(_DAY_PATTERN.search(text_clean))
    has_opinion = bool(_OPINION_PATTERN.search(text_clean))

    # Extract dates
    extracted_dates = _TEMPORAL_KEYWORDS.findall(text_clean)
    day_matches = _DAY_PATTERN.findall(text_clean)
    extracted_dates = extracted_dates + day_matches

    # Extract numbers
    num_matches = _NUMERICAL_PATTERN.findall(text_clean)
    extracted_numbers = [(m[0], m[1]) for m in num_matches]
    has_numbers = len(extracted_numbers) > 0

    # Assertions
    has_assertion = bool(_ASSERTION_PATTERN.search(text_clean))

    # Build clean claim text (for search queries)
    # Remove question words at the start for cleaner searches
    claim_text = re.sub(r'^\s*(is|are|was|were|did|does|do|has|have|had)\s+', '', text_clean, flags=re.IGNORECASE)
    claim_text = claim_text.rstrip("?").strip()

    # ── Classification logic ──────────────────────────────────────────────

    if has_opinion and not has_temporal and not has_numbers:
        input_type = "opinion"

    elif has_temporal and (has_question or word_count < 30):
        # Short text with date references → temporal verification
        input_type = "temporal"

    elif has_question and word_count < 50:
        # Short question → treat as factual question → verify
        input_type = "question"

    elif word_count > 50 and not has_question:
        # Long text without question markers → news article → ML pipeline
        input_type = "article"

    elif has_assertion or has_numbers:
        # Contains factual assertions or specific numbers → verify
        input_type = "claim"

    elif word_count <= 50:
        # Short text, no strong signals → default to claim verification
        input_type = "claim"

    else:
        # Fallback for longer text with mixed signals
        input_type = "article"

    return {
        "type": input_type,
        "has_question": has_question,
        "has_temporal": has_temporal,
        "has_numbers": has_numbers,
        "has_assertion": has_assertion,
        "has_opinion": has_opinion,
        "extracted_dates": extracted_dates,
        "extracted_numbers": extracted_numbers,
        "word_count": word_count,
        "claim_text": claim_text,
    }


if __name__ == "__main__":
    # Quick test
    tests = [
        "Is today April 11th 2026?",
        "Was yesterday Thursday?",
        "India landed on the moon in 2023",
        "The GDP of India is $3.7 trillion",
        "BREAKING!!! Scientists discover cure for cancer!!! " * 5,
        "I think the government is hiding something",
        "What day is it today?",
        "Did NASA find water on Mars?",
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday.",
    ]

    for t in tests:
        result = classify_input(t)
        print(f"[{result['type']:>10}] {t[:70]}...")
        if result["extracted_dates"]:
            print(f"             dates: {result['extracted_dates']}")
        if result["extracted_numbers"]:
            print(f"             numbers: {result['extracted_numbers']}")
        print()
