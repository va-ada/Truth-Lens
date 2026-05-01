"""
TruthLens — Text Preprocessor
===============================
Handles text cleaning and ENTITY MASKING (key debiasing technique).

Entity masking replaces person names, organizations, and locations with
generic tokens ([PERSON], [ORG], [LOC]) so the model learns content patterns
instead of associating specific entities with fake/real labels.

Produces two versions of each text:
  - cleaned_text: Basic cleaning (for GloVe embeddings)
  - masked_text:  Cleaned + entity-masked (for TF-IDF)
  - raw text is preserved for stylometric feature extraction
"""

import os
import re
import sys
import subprocess
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Lazy-load heavy libraries
_nlp = None
_lemmatizer = None
_stopwords = None

# Known news sources that spaCy NER often misses — supplementary regex masking
_NEWS_SOURCE_PATTERNS = re.compile(
    r'\b(reuters|associated press|afp|agence france[- ]presse|'
    r'fox news|cnn|bbc|breitbart|infowars|msnbc|nbc news|'
    r'abc news|cbs news|the guardian|washington post|new york times|'
    r'huffington post|huffpost|daily mail|buzzfeed|politifact|snopes|'
    r'bloomberg|wall street journal|wsj|ap news|ap wire|rt|russia today|'
    r'al jazeera|aljazeera|usa today|newsweek|the hill|vox|vice|slate|'
    r'the atlantic|natural news|zero hedge|epoch times|oan|newsmax|'
    r'truthout|the intercept|propublica|daily beast|salon|mother jones|'
    r'politico|axios|the wrap|variety|deadline|the verge|techcrunch|'
    r'wired|fortune|time magazine)\b',
    re.IGNORECASE,
)

# Aggressive masking — only used by the A2 debiasing loop. These hit the
# Reuters/AP-style boilerplate (datelines + reporting verbs) that survives
# the standard NER pass. The trade-off: in aggressive mode we lose some
# legitimate signal, so this should NOT be the default for production.
_AGGRESSIVE_DATELINE_PATTERN = re.compile(
    r'^\s*[A-Z][A-Z\s,/]{2,40}\s*\([A-Za-z\s]+\)\s*[-–—]?\s*',
    re.MULTILINE,
)
_AGGRESSIVE_BOILERPLATE_PATTERN = re.compile(
    r'\b(said|reported|announced|stated|noted|added|told\s+\w+|'
    r'according\s+to|wrote|claimed|confirmed|denied|tweeted)\b',
    re.IGNORECASE,
)


def _load_spacy():
    """Load spaCy model lazily."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(config.SPACY_MODEL, disable=["parser"])
        except OSError:
            print(f"[PREPROCESS] Downloading spaCy model '{config.SPACY_MODEL}'...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", config.SPACY_MODEL],
                check=True,
            )
            _nlp = spacy.load(config.SPACY_MODEL, disable=["parser"])
        # Increase max length for long articles
        _nlp.max_length = 2_000_000
    return _nlp


def _load_stopwords():
    """Load NLTK stopwords lazily."""
    global _stopwords
    if _stopwords is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words("english"))
    return _stopwords


def clean_text(text, lowercase=True):
    """
    Basic text cleaning. Does NOT remove entities.

    Steps:
        1. Strip news agency bylines (Reuters, AP, AFP)
        2. Optionally convert to lowercase (default: True)
        3. Remove URLs
        4. Remove HTML tags
        5. Remove email addresses
        6. Remove special characters (keep letters, numbers, spaces)
        7. Remove extra whitespace
        8. Truncate to MAX_TEXT_LENGTH

    Args:
        text: Raw text string
        lowercase: If True, convert to lowercase (set False to preserve case for NER)

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Strip Reuters/AP byline patterns (e.g. "WASHINGTON (Reuters) -")
    # These are dead giveaways that leak source identity into features
    text = re.sub(r'^[A-Z][A-Z\s,/]+\s*\(Reuters\)\s*[-–—]?\s*', '', text)
    text = re.sub(r'^[A-Z][A-Z\s,/]+\s*\(AP\)\s*[-–—]?\s*', '', text)
    text = re.sub(r'^[A-Z][A-Z\s,/]+\s*\(AFP\)\s*[-–—]?\s*', '', text)

    if lowercase:
        text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove special characters but keep basic punctuation for sentence detection
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate
    if len(text) > config.MAX_TEXT_LENGTH:
        text = text[:config.MAX_TEXT_LENGTH]

    return text


def mask_entities_batch(texts, batch_size=500):
    """
    Batch entity masking using nlp.pipe() — 3-5x faster than one-at-a-time.

    Replaces PERSON/ORG/GPE/LOC/DATE with generic tokens.
    """
    nlp = _load_spacy()

    results = []
    valid_indices = []
    valid_texts = []

    # Separate valid vs empty texts
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            valid_indices.append(i)
            valid_texts.append(text)
        else:
            pass  # Will fill with ""

    # Pre-fill results with empty strings
    results = [""] * len(texts)

    # Batch process with nlp.pipe()
    for idx, doc in tqdm(
        zip(valid_indices, nlp.pipe(valid_texts, batch_size=batch_size)),
        total=len(valid_texts),
        desc="Entity Masking",
    ):
        text = texts[idx]
        masked = text
        entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

        for ent in entities:
            if ent.label_ == "PERSON":
                replacement = "[PERSON]"
            elif ent.label_ == "ORG":
                replacement = "[ORG]"
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                replacement = "[LOC]"
            elif ent.label_ == "DATE":
                replacement = "[DATE]"
            elif ent.label_ == "NORP":
                replacement = "[GROUP]"
            elif ent.label_ == "EVENT":
                replacement = "[EVENT]"
            else:
                continue

            masked = masked[:ent.start_char] + replacement + masked[ent.end_char:]

        # Supplementary regex masking for news sources spaCy misses
        masked = _NEWS_SOURCE_PATTERNS.sub("[SOURCE]", masked)
        results[idx] = masked

    return results


def lemmatize_batch(texts, batch_size=500):
    """
    Batch lemmatization using nlp.pipe() — much faster than row-by-row.
    """
    nlp = _load_spacy()
    stop_words = _load_stopwords()

    results = []
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            valid_indices.append(i)
            valid_texts.append(text)

    results = [""] * len(texts)

    for idx, doc in tqdm(
        zip(valid_indices, nlp.pipe(valid_texts, batch_size=batch_size)),
        total=len(valid_texts),
        desc="Lemmatizing",
    ):
        tokens = [
            token.lemma_
            for token in doc
            if (token.text.lower() not in stop_words
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 1)
        ]
        results[idx] = " ".join(tokens)

    return results


def mask_entities(text, level="standard"):
    """Replace named entities with generic tokens to reduce entity bias.

    Args:
        text: input string
        level: "standard" (default) — NER + supplementary news-source regex.
               "aggressive" — also mask Reuters/AP-style datelines and
               reporting-verb boilerplate. Used by the A2 debiasing loop.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    nlp = _load_spacy()
    doc = nlp(text)

    masked = text
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

    for ent in entities:
        if ent.label_ == "PERSON":
            replacement = "[PERSON]"
        elif ent.label_ == "ORG":
            replacement = "[ORG]"
        elif ent.label_ in ("GPE", "LOC", "FAC"):
            replacement = "[LOC]"
        elif ent.label_ == "DATE":
            replacement = "[DATE]"
        elif ent.label_ == "NORP":
            replacement = "[GROUP]"
        elif ent.label_ == "EVENT":
            replacement = "[EVENT]"
        else:
            continue

        masked = masked[:ent.start_char] + replacement + masked[ent.end_char:]

    # Supplementary regex masking for news sources spaCy misses
    masked = _NEWS_SOURCE_PATTERNS.sub("[SOURCE]", masked)

    if level == "aggressive":
        masked = _AGGRESSIVE_DATELINE_PATTERN.sub("", masked)
        masked = _AGGRESSIVE_BOILERPLATE_PATTERN.sub("[SAID]", masked)

    return masked


def remove_stopwords_and_lemmatize(text):
    """
    Remove stopwords and lemmatize text using spaCy.
    Single-text version (used by LIME explainer).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    nlp = _load_spacy()
    stop_words = _load_stopwords()

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if (token.text.lower() not in stop_words
            and not token.is_punct
            and not token.is_space
            and len(token.text) > 1)
    ]

    return " ".join(tokens)


def preprocess_dataframe(df, enable_masking=None, mask_level="standard"):
    """
    Apply full preprocessing pipeline to a DataFrame.
    Uses BATCH processing (nlp.pipe) for 3-5x speedup.

    Args:
        df: input DataFrame with a `text` column
        enable_masking: override config.ENABLE_ENTITY_MASKING
        mask_level: "standard" or "aggressive". The aggressive mode is used
            by the A2 debiasing loop and additionally strips Reuters-style
            datelines and reporting-verb boilerplate that survives NER.

    Produces:
        - text:           Original raw text (for stylometric features)
        - cleaned_text:   Cleaned text (for SBERT embeddings)
        - masked_text:    Cleaned + entity-masked (for TF-IDF)
        - processed_text: Lemmatized masked text (for TF-IDF input)
    """
    if enable_masking is None:
        enable_masking = config.ENABLE_ENTITY_MASKING

    df = df.copy()
    total = len(df)

    if total == 0:
        print("[PREPROCESS] No texts to process. Returning empty DataFrame.")
        df["cleaned_text"] = pd.Series(dtype=str)
        df["masked_text"] = pd.Series(dtype=str)
        df["processed_text"] = pd.Series(dtype=str)
        return df

    # Step 1: Clean text WITHOUT lowercasing (spaCy NER needs capitalization)
    print(f"[PREPROCESS] Cleaning {total} texts (preserving case for NER)...")
    tqdm.pandas(desc="Cleaning")
    cleaned_mixed_case = df["text"].progress_apply(lambda t: clean_text(t, lowercase=False))

    # Step 2: Entity masking on MIXED-CASE text (spaCy NER relies on capitalization)
    if enable_masking:
        if mask_level == "aggressive":
            print(f"[PREPROCESS] Entity masking {total} texts (AGGRESSIVE mode — A2 loop)...")
            # Aggressive mode falls back to single-text masking (no batch helper) so
            # we benefit from the dateline + boilerplate scrub. This is slower but
            # only runs during the debiasing loop.
            df["masked_text"] = [mask_entities(t, level="aggressive") for t in
                                  tqdm(cleaned_mixed_case.tolist(), desc="Aggressive Mask")]
        else:
            print(f"[PREPROCESS] Entity masking {total} texts (batch mode, mixed-case)...")
            df["masked_text"] = mask_entities_batch(
                cleaned_mixed_case.tolist(), batch_size=500
            )
    else:
        print("[PREPROCESS] Entity masking DISABLED. Using cleaned text directly.")
        df["masked_text"] = cleaned_mixed_case

    # Step 3: Lowercase — cleaned_text for GloVe (no masking, real words needed)
    df["cleaned_text"] = cleaned_mixed_case.str.lower()

    # Step 4: Lowercase masked text, then lemmatize for TF-IDF
    masked_lower = df["masked_text"].str.lower().tolist()
    print(f"[PREPROCESS] Lemmatizing {total} texts (batch mode)...")
    df["processed_text"] = lemmatize_batch(masked_lower, batch_size=500)

    # Ensure string type before filtering
    df["processed_text"] = df["processed_text"].astype(str)

    # Remove rows that became empty after preprocessing
    before = len(df)
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[PREPROCESS] Removed {removed} empty rows after preprocessing.")

    print(f"[PREPROCESS] Done. {len(df)} texts processed.")
    return df


def save_processed(df, name):
    """Save processed DataFrame to CSV."""
    path = os.path.join(config.PROCESSED_DIR, f"{name}_processed.csv")
    df.to_csv(path, index=False)
    print(f"[PREPROCESS] Saved processed data to: {path}")
    return path


def load_processed(name):
    """Load processed DataFrame from CSV if it exists and is valid."""
    path = os.path.join(config.PROCESSED_DIR, f"{name}_processed.csv")
    if os.path.exists(path):
        # Validate file isn't empty/corrupt (e.g. from a crashed run)
        file_size = os.path.getsize(path)
        if file_size < 200:  # Headers-only file is ~67 bytes
            print(f"[PREPROCESS] Cached file too small ({file_size}B), re-processing: {path}")
            os.remove(path)
            return None
        print(f"[PREPROCESS] Loading cached processed data: {path}")
        df = pd.read_csv(path)
        if len(df) == 0:
            print(f"[PREPROCESS] Cached file has no data rows, re-processing: {path}")
            os.remove(path)
            return None
        return df
    return None


if __name__ == "__main__":
    # Quick test — mirrors the production pipeline exactly
    test_texts = [
        "Donald Trump announced new policies at the White House yesterday, according to Reuters.",
        "BREAKING: Scientists at Google discovered a cure for cancer!!! Click here to learn more!!!",
        "The economy is growing steadily according to recent Federal Reserve reports.",
    ]

    print("=== Preprocessing Demo ===\n")
    for text in test_texts:
        # Step 1: Clean WITHOUT lowercasing (NER needs capitalization)
        cleaned_mixed = clean_text(text, lowercase=False)
        # Step 2: Entity masking on mixed-case text
        masked = mask_entities(cleaned_mixed)
        # Step 3: Lowercase for downstream use
        masked_lower = masked.lower()
        # Step 4: Lemmatize
        lemmatized = remove_stopwords_and_lemmatize(masked_lower)

        print(f"Original:    {text}")
        print(f"Cleaned:     {cleaned_mixed}")
        print(f"Masked:      {masked}")
        print(f"Lemmatized:  {lemmatized}")
        print()
