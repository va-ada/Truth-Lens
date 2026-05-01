"""
TruthLens — Fact Checker
==========================
Multi-source fact verification pipeline:

1. Temporal Verification  — dates, days, "today/yesterday" claims
2. Wikipedia Verification — entity/event lookup + cross-reference
3. Web Search Verification — DuckDuckGo snippets + similarity scoring
4. Evidence Aggregation    — weighted combination into final verdict

All APIs are FREE and require no API keys.
"""

import re
import warnings
from datetime import datetime, timedelta
from collections import Counter

import numpy as np


# ── Temporal Verification ─────────────────────────────────────────────────────

def verify_temporal(text, extracted_dates=None):
    """
    Verify date/time claims against the system clock.

    Handles:
      - "Is today April 11th 2026?"
      - "Was yesterday Thursday?"
      - "What day is it today?"
      - "Is it 2026?"

    Args:
        text: Original user input
        extracted_dates: Date strings from claim_detector (optional)

    Returns:
        dict with verdict, evidence, confidence
    """
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    text_lower = text.lower().strip()

    results = {
        "source": "Temporal Check",
        "verdict": "Unverifiable",
        "evidence": "",
        "confidence": 0.0,
        "details": {},
    }

    # ── "What day is it today?" / "What is today's date?" ──
    what_day_match = re.search(
        r'what\s+(?:day|date)\s+is\s+(?:it\s+)?today', text_lower
    )
    if what_day_match:
        results["verdict"] = "Informational"
        results["evidence"] = f"Today is {now.strftime('%A, %B %d, %Y')}."
        results["confidence"] = 0.99
        results["details"]["today"] = str(today)
        return results

    # ── Try to parse dates from text using dateparser ──
    try:
        import dateparser
        _has_dateparser = True
    except ImportError:
        _has_dateparser = False

    # ── "Is today [date]?" / "Today is [date]" ──
    today_claim = re.search(
        r'(?:is\s+)?today\s+(?:is\s+)?(.+?)(?:\?|$)', text_lower
    )
    if today_claim:
        date_str = today_claim.group(1).strip().rstrip("?. ")
        parsed = _parse_date(date_str, _has_dateparser)
        if parsed:
            if parsed == today:
                results["verdict"] = "Verified"
                results["evidence"] = f"Correct. Today is {now.strftime('%A, %B %d, %Y')}."
                results["confidence"] = 0.95
            else:
                results["verdict"] = "False"
                results["evidence"] = (
                    f"Incorrect. Today is {now.strftime('%A, %B %d, %Y')}, "
                    f"not {parsed.strftime('%B %d, %Y')}."
                )
                results["confidence"] = 0.95
            results["details"]["claimed_date"] = str(parsed)
            results["details"]["actual_date"] = str(today)
            return results

    # ── "Was yesterday [date/day]?" ──
    yesterday_claim = re.search(
        r'(?:was\s+)?yesterday\s+(?:was\s+)?(.+?)(?:\?|$)', text_lower
    )
    if yesterday_claim:
        date_str = yesterday_claim.group(1).strip().rstrip("?. ")
        # Check day-of-week
        day_name = _match_day_name(date_str)
        if day_name is not None:
            actual_day = yesterday.strftime("%A").lower()
            if day_name == actual_day:
                results["verdict"] = "Verified"
                results["evidence"] = f"Correct. Yesterday was {yesterday.strftime('%A, %B %d, %Y')}."
                results["confidence"] = 0.95
            else:
                results["verdict"] = "False"
                results["evidence"] = (
                    f"Incorrect. Yesterday was {yesterday.strftime('%A')}, "
                    f"not {date_str.title()}."
                )
                results["confidence"] = 0.95
            return results

        # Check date
        parsed = _parse_date(date_str, _has_dateparser)
        if parsed:
            if parsed == yesterday:
                results["verdict"] = "Verified"
                results["evidence"] = f"Correct. Yesterday was {yesterday.strftime('%A, %B %d, %Y')}."
                results["confidence"] = 0.95
            else:
                results["verdict"] = "False"
                results["evidence"] = (
                    f"Incorrect. Yesterday was {yesterday.strftime('%B %d, %Y')}, "
                    f"not {parsed.strftime('%B %d, %Y')}."
                )
                results["confidence"] = 0.95
            return results

    # ── "Is tomorrow [day/date]?" ──
    tomorrow_claim = re.search(
        r'(?:is\s+)?tomorrow\s+(?:is\s+)?(.+?)(?:\?|$)', text_lower
    )
    if tomorrow_claim:
        date_str = tomorrow_claim.group(1).strip().rstrip("?. ")
        day_name = _match_day_name(date_str)
        if day_name is not None:
            actual_day = tomorrow.strftime("%A").lower()
            if day_name == actual_day:
                results["verdict"] = "Verified"
                results["evidence"] = f"Correct. Tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}."
                results["confidence"] = 0.95
            else:
                results["verdict"] = "False"
                results["evidence"] = (
                    f"Incorrect. Tomorrow is {tomorrow.strftime('%A')}, "
                    f"not {date_str.title()}."
                )
                results["confidence"] = 0.95
            return results

    # ── Day-of-week check: "Is it [day] today?" ──
    day_today = re.search(
        r'is\s+(?:it\s+)?(\w+day)\s+today', text_lower
    )
    if day_today:
        claimed_day = day_today.group(1).lower()
        actual_day = now.strftime("%A").lower()
        if claimed_day == actual_day:
            results["verdict"] = "Verified"
            results["evidence"] = f"Correct. Today is {now.strftime('%A')}."
            results["confidence"] = 0.95
        else:
            results["verdict"] = "False"
            results["evidence"] = f"Incorrect. Today is {now.strftime('%A')}, not {claimed_day.title()}."
            results["confidence"] = 0.95
        return results

    # ── Year check: "Is it 2026?" ──
    year_match = re.search(r'is\s+it\s+(\d{4})', text_lower)
    if year_match:
        claimed_year = int(year_match.group(1))
        actual_year = now.year
        if claimed_year == actual_year:
            results["verdict"] = "Verified"
            results["evidence"] = f"Correct. The current year is {actual_year}."
            results["confidence"] = 0.95
        else:
            results["verdict"] = "False"
            results["evidence"] = f"Incorrect. The current year is {actual_year}, not {claimed_year}."
            results["confidence"] = 0.95
        return results

    # ── Fallback: try to find any date in the text and contextualize ──
    if extracted_dates:
        results["verdict"] = "Informational"
        results["evidence"] = (
            f"Date references found: {', '.join(extracted_dates)}. "
            f"Today is {now.strftime('%A, %B %d, %Y')}."
        )
        results["confidence"] = 0.5
        return results

    results["evidence"] = f"Could not parse a specific date claim. Today is {now.strftime('%A, %B %d, %Y')}."
    return results


def _parse_date(date_str, has_dateparser=False):
    """Try to parse a date string into a date object."""
    import calendar

    # Manual parsing for common formats
    date_str_clean = date_str.strip().lower()

    # "april 11th 2026", "11th april 2026", "april 11 2026"
    month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    month_abbr = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
    all_months = {**month_names, **month_abbr}

    # Pattern: "month day year" or "month day"
    m1 = re.match(r'(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?', date_str_clean)
    if m1:
        month_str, day_str, year_str = m1.groups()
        if month_str in all_months:
            month = all_months[month_str]
            day = int(day_str)
            year = int(year_str) if year_str else datetime.now().year
            try:
                return datetime(year, month, day).date()
            except ValueError:
                pass

    # Pattern: "day month year"
    m2 = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)(?:\s+(\d{4}))?', date_str_clean)
    if m2:
        day_str, month_str, year_str = m2.groups()
        if month_str in all_months:
            month = all_months[month_str]
            day = int(day_str)
            year = int(year_str) if year_str else datetime.now().year
            try:
                return datetime(year, month, day).date()
            except ValueError:
                pass

    # Fallback to dateparser if available
    if has_dateparser:
        try:
            import dateparser
            parsed = dateparser.parse(date_str, settings={
                "PREFER_DATES_FROM": "current_period",
                "RELATIVE_BASE": datetime.now(),
            })
            if parsed:
                return parsed.date()
        except Exception:
            pass

    return None


def _match_day_name(text):
    """Extract a day-of-week name from text, or return None."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    text_lower = text.lower().strip()
    for day in days:
        if day in text_lower:
            return day
    return None


# ── Wikipedia Verification ────────────────────────────────────────────────────

def verify_wikipedia(claim_text, entities=None):
    """
    Look up entities/events on Wikipedia and cross-reference with the claim.

    Uses the `wikipedia-api` library (free, no key required).

    Args:
        claim_text: The factual claim to verify
        entities: Optional list of entity names to search

    Returns:
        dict with verdict, evidence, source, confidence
    """
    result = {
        "source": "Wikipedia",
        "verdict": "No Data",
        "evidence": "",
        "confidence": 0.0,
        "url": "",
    }

    try:
        import wikipediaapi
    except ImportError:
        result["evidence"] = "wikipedia-api not installed. Run: pip install wikipedia-api"
        return result

    wiki = wikipediaapi.Wikipedia(
        user_agent="TruthLens/1.0 (college project; contact: truthlens@example.com)",
        language="en",
    )

    # Build search terms from the claim
    search_terms = []
    if entities:
        search_terms.extend(entities)
    else:
        # Extract capitalized phrases as likely entity names
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', claim_text)
        search_terms.extend(caps)

    # Build a composite search from significant words (better than single entities)
    significant = [w for w in claim_text.split() if len(w) > 3][:7]
    if len(significant) >= 2:
        search_terms.insert(0, " ".join(significant))  # Try composite first

    if not search_terms:
        result["evidence"] = "No searchable entities found in claim."
        return result

    # ── Step 1: Use MediaWiki search API to find relevant page titles ──
    page_titles = []
    try:
        import requests as _req
        _headers = {"User-Agent": "TruthLens/1.0 (college project)"}
        for term in search_terms[:2]:
            resp = _req.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": term,
                        "srlimit": 3, "format": "json"},
                timeout=10,
                headers=_headers,
            )
            if resp.ok:
                for hit in resp.json().get("query", {}).get("search", []):
                    title = hit.get("title", "")
                    if title and title not in page_titles:
                        page_titles.append(title)
    except Exception:
        pass

    # Fallback: also try direct page lookup for entity names
    for term in search_terms[:3]:
        if term not in page_titles:
            page_titles.append(term)

    # ── Step 2: Fetch pages and score them ──
    best_match = None
    best_similarity = 0.0

    for title in page_titles[:5]:  # Limit to 5 pages
        try:
            page = wiki.page(title)
            if not page.exists():
                continue

            summary = page.summary[:2000]
            if not summary:
                continue

            similarity = _word_overlap_similarity(claim_text, summary)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "title": page.title,
                    "summary": summary[:500],
                    "url": page.fullurl,
                    "similarity": similarity,
                }
        except Exception:
            continue

    if best_match is None:
        result["evidence"] = f"No relevant Wikipedia articles found for: {', '.join(search_terms[:3])}"
        return result

    # Determine verdict based on similarity
    result["url"] = best_match["url"]

    if best_similarity > 0.3:
        result["verdict"] = "Supported"
        result["confidence"] = min(0.6 + best_similarity * 0.3, 0.85)
        result["evidence"] = (
            f"Wikipedia article '{best_match['title']}' appears relevant "
            f"(similarity: {best_similarity:.0%}).\n\n"
            f"Excerpt: {best_match['summary'][:300]}..."
        )
    elif best_similarity > 0.15:
        result["verdict"] = "Partially Relevant"
        result["confidence"] = 0.4
        result["evidence"] = (
            f"Wikipedia article '{best_match['title']}' has some overlap "
            f"(similarity: {best_similarity:.0%}).\n\n"
            f"Excerpt: {best_match['summary'][:300]}..."
        )
    else:
        result["verdict"] = "Insufficient"
        result["confidence"] = 0.2
        result["evidence"] = f"Found '{best_match['title']}' but low relevance to claim."

    return result


# ── Web Search Verification ───────────────────────────────────────────────────

def verify_web(claim_text):
    """
    Search the web for the claim and check for supporting/contradicting evidence.

    Uses DuckDuckGo (free, no API key required).

    Args:
        claim_text: The factual claim to search for

    Returns:
        dict with verdict, sources, confidence
    """
    result = {
        "source": "Web Search",
        "verdict": "No Results",
        "evidence": "",
        "confidence": 0.0,
        "sources": [],
    }

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            result["evidence"] = "ddgs not installed. Run: pip install ddgs"
            return result

    # Search DuckDuckGo
    try:
        search_results = list(DDGS().text(claim_text, max_results=5))
    except Exception as e:
        result["evidence"] = f"Web search failed: {e}"
        return result

    if not search_results:
        result["evidence"] = "No web results found for this claim."
        return result

    # Analyze each result
    supporting = 0
    contradicting = 0
    sources = []

    # Contradiction keywords (present in snippets that refute claims)
    _CONTRADICT_WORDS = {"false", "fake", "hoax", "debunked", "misleading",
                         "incorrect", "not true", "myth", "disproven", "untrue"}
    _SUPPORT_WORDS = {"confirmed", "verified", "true", "correct", "fact",
                      "accurate", "official", "reported", "announced", "according to"}

    for sr in search_results:
        title = sr.get("title", "")
        snippet = sr.get("body", "")
        url = sr.get("href", "")

        if not snippet:
            continue

        # Compute similarity between claim and snippet
        similarity = _word_overlap_similarity(claim_text, snippet)

        # Check for contradiction/support signals in snippet
        snippet_lower = snippet.lower()
        has_contradict = any(w in snippet_lower for w in _CONTRADICT_WORDS)
        has_support = any(w in snippet_lower for w in _SUPPORT_WORDS)

        # Source reliability score
        reliability = _source_reliability(url)

        # Only count keyword signals if snippet is actually relevant to claim
        # (similarity > 0.1 means meaningful word overlap with the claim)
        if similarity < 0.05:
            stance = "neutral"
        elif has_contradict and not has_support:
            contradicting += 1
            stance = "contradicts"
        elif has_support and not has_contradict and similarity > 0.15:
            supporting += 1
            stance = "supports"
        elif similarity > 0.25:
            supporting += 1  # High similarity without contradiction = soft support
            stance = "relevant"
        else:
            stance = "neutral"

        sources.append({
            "title": title[:100],
            "url": url,
            "snippet": snippet[:200],
            "similarity": round(similarity, 3),
            "stance": stance,
            "reliability": reliability,
        })

    result["sources"] = sources
    total = supporting + contradicting
    n_results = len(sources)

    if total == 0:
        result["verdict"] = "Uncertain"
        result["confidence"] = 0.3
        result["evidence"] = f"Found {n_results} results but none clearly support or contradict the claim."
    elif supporting > contradicting:
        ratio = supporting / max(total, 1)
        result["verdict"] = "Supported"
        result["confidence"] = min(0.5 + ratio * 0.35, 0.85)
        result["evidence"] = f"{supporting}/{n_results} sources support the claim."
    elif contradicting > supporting:
        ratio = contradicting / max(total, 1)
        result["verdict"] = "Contradicted"
        result["confidence"] = min(0.5 + ratio * 0.35, 0.85)
        result["evidence"] = f"{contradicting}/{n_results} sources contradict the claim."
    else:
        result["verdict"] = "Mixed"
        result["confidence"] = 0.4
        result["evidence"] = f"Sources are split: {supporting} support, {contradicting} contradict."

    return result


def _source_reliability(url):
    """Score source reliability based on domain (0.0 to 1.0)."""
    if not url:
        return 0.3

    url_lower = url.lower()

    # High reliability
    if any(d in url_lower for d in [".gov", ".edu", "reuters.com", "apnews.com",
                                      "bbc.com", "bbc.co.uk", "nature.com",
                                      "sciencedirect.com", "who.int", "un.org"]):
        return 0.9

    # Medium-high
    if any(d in url_lower for d in ["wikipedia.org", "nytimes.com", "washingtonpost.com",
                                      "theguardian.com", "economist.com"]):
        return 0.8

    # Medium
    if any(d in url_lower for d in [".org", "cnn.com", "nbcnews.com", "abcnews.com"]):
        return 0.6

    # Default
    return 0.4


# ── Evidence Aggregation ──────────────────────────────────────────────────────

def aggregate_evidence(temporal_result=None, wikipedia_result=None,
                       web_result=None, ml_result=None):
    """
    Combine verification signals from all sources into a final verdict.

    Weighting:
      - Temporal check:  0.95 (near-certain when applicable)
      - Wikipedia:       0.80 (reliable but may be outdated)
      - Web search:      0.65 (broad but noisy)
      - ML style:        0.50 (supplementary, not decisive for facts)

    Returns:
        dict with final verdict, confidence, evidence trail, disclaimer
    """
    evidence_trail = []
    verdicts = []  # (verdict_string, confidence, weight)

    # Temporal
    if temporal_result and temporal_result.get("verdict") not in ("Unverifiable", ""):
        evidence_trail.append(temporal_result)
        v = temporal_result["verdict"]
        c = temporal_result.get("confidence", 0.5)

        if v == "Verified":
            verdicts.append((1.0, c, 0.95))
        elif v == "False":
            verdicts.append((0.0, c, 0.95))
        elif v == "Informational":
            verdicts.append((0.5, c, 0.3))  # Neutral, low weight

    # Wikipedia
    if wikipedia_result and wikipedia_result.get("verdict") not in ("No Data", ""):
        evidence_trail.append(wikipedia_result)
        v = wikipedia_result["verdict"]
        c = wikipedia_result.get("confidence", 0.5)

        if v == "Supported":
            verdicts.append((0.8, c, 0.80))
        elif v == "Contradicted":
            verdicts.append((0.2, c, 0.80))
        elif v == "Partially Relevant":
            verdicts.append((0.5, c, 0.40))

    # Web search
    if web_result and web_result.get("verdict") not in ("No Results", ""):
        evidence_trail.append(web_result)
        v = web_result["verdict"]
        c = web_result.get("confidence", 0.5)

        if v == "Supported":
            verdicts.append((0.8, c, 0.65))
        elif v == "Contradicted":
            verdicts.append((0.15, c, 0.65))
        elif v == "Mixed":
            verdicts.append((0.5, c, 0.40))
        elif v == "Uncertain":
            verdicts.append((0.5, c, 0.20))

    # ML style analysis (supplementary)
    if ml_result:
        evidence_trail.append({
            "source": "ML Style Analysis",
            "verdict": ml_result.get("label", "Unknown"),
            "evidence": ml_result.get("detail", ""),
            "confidence": ml_result.get("confidence", 0.5),
        })

    # ── Weighted aggregation ──
    if not verdicts:
        return {
            "final_verdict": "Cannot Verify",
            "confidence": 0.0,
            "evidence_trail": evidence_trail,
            "disclaimer": "AI-generated analysis — no sources found. Verify independently.",
        }

    # Weighted average of truth scores
    total_weight = sum(w * c for _, c, w in verdicts)
    if total_weight == 0:
        truth_score = 0.5
    else:
        truth_score = sum(score * conf * weight for score, conf, weight in verdicts) / total_weight

    # Map truth score to verdict
    confidence = min(max(sum(c * w for _, c, w in verdicts) / max(sum(w for _, _, w in verdicts), 1), 0.0), 0.95)

    if truth_score >= 0.75:
        final_verdict = "Verified"
    elif truth_score >= 0.60:
        final_verdict = "Likely True"
    elif truth_score >= 0.40:
        final_verdict = "Uncertain"
    elif truth_score >= 0.25:
        final_verdict = "Likely False"
    else:
        final_verdict = "False"

    return {
        "final_verdict": final_verdict,
        "confidence": round(confidence, 3),
        "truth_score": round(truth_score, 3),
        "evidence_trail": evidence_trail,
        "disclaimer": "AI-generated analysis — verify independently before citing.",
    }


def check_claim(text, claim_info):
    """
    Main entry point: run all applicable verifiers and aggregate.

    Args:
        text: Original user input
        claim_info: Output from classify_input()

    Returns:
        Aggregated verdict dict
    """
    temporal_result = None
    wikipedia_result = None
    web_result = None

    claim_text = claim_info.get("claim_text", text)

    # Run temporal check if dates detected
    if claim_info.get("has_temporal") or claim_info.get("type") == "temporal":
        temporal_result = verify_temporal(text, claim_info.get("extracted_dates"))

        # If temporal check is definitive, skip other checks
        if temporal_result.get("verdict") in ("Verified", "False") and \
           temporal_result.get("confidence", 0) > 0.9:
            return aggregate_evidence(temporal_result=temporal_result)

    # Run Wikipedia check
    try:
        wikipedia_result = verify_wikipedia(claim_text)
    except Exception as e:
        wikipedia_result = {
            "source": "Wikipedia",
            "verdict": "Error",
            "evidence": f"Wikipedia lookup failed: {e}",
            "confidence": 0.0,
        }

    # Run web search
    try:
        web_result = verify_web(claim_text)
    except Exception as e:
        web_result = {
            "source": "Web Search",
            "verdict": "Error",
            "evidence": f"Web search failed: {e}",
            "confidence": 0.0,
        }

    return aggregate_evidence(
        temporal_result=temporal_result,
        wikipedia_result=wikipedia_result,
        web_result=web_result,
    )


# ── Utility ───────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "while", "that", "this", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "what", "which", "who", "whom",
})


def _word_overlap_similarity(text_a, text_b):
    """
    Compute word overlap using recall (fraction of claim words found in evidence).
    Recall-based is better than Jaccard here because evidence texts (Wikipedia
    summaries, web snippets) are much longer than claims.
    """
    words_a = set(w.lower() for w in re.findall(r'\b\w+\b', text_a) if len(w) > 2) - _STOPWORDS
    words_b = set(w.lower() for w in re.findall(r'\b\w+\b', text_b) if len(w) > 2) - _STOPWORDS

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b

    # Recall: what fraction of claim words (text_a) appear in evidence (text_b)
    recall = len(intersection) / len(words_a)
    # Also consider precision (avoid scoring irrelevant long texts high)
    precision = len(intersection) / len(words_b) if len(words_b) > 0 else 0.0

    # F1-like harmonic mean, biased toward recall (2:1)
    if recall + precision == 0:
        return 0.0
    return (3 * recall * precision) / (2 * precision + recall)


if __name__ == "__main__":
    # Quick tests
    print("=== Temporal Tests ===")
    tests = [
        "Is today April 11th 2026?",
        "Was yesterday April 10th 2026?",
        "What day is it today?",
        "Is it 2026?",
        "Was yesterday Monday?",
    ]
    for t in tests:
        r = verify_temporal(t)
        print(f"  [{r['verdict']:>15}] {t}")
        print(f"                   {r['evidence']}")
        print()
