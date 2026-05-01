"""
TruthLens — Sentence-Level Scoring (Originality.ai parity)
============================================================
Splits an input document into sentences and runs the trained ensemble on
each one independently. Surfaces the top-K most-suspect sentences so the
UI can highlight them in-line, the way Originality.ai surfaces "potentially
false" passages inside long-form text.

Public API:
    score_sentences(text, engine, model, top_k=None) -> list[dict]

Each returned dict contains:
    sentence: str              — the original sentence
    prob_fake: float           — model probability that this sentence is fake
    prob_real: float           — 1 - prob_fake
    risk: str                  — "low" | "medium" | "high"
    char_start: int            — position in the original text
    char_end: int

This module is intentionally dependency-light: it only needs an already-
fitted `TruthLensFeatureEngine` and a model exposing `predict_proba`.
"""

from __future__ import annotations

import os
import re
import sys

# Make `import config` and `from src...` work whether this is run as a script
# or imported from elsewhere in the codebase.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRUTHLENS_DIR = os.path.dirname(_HERE)
if _TRUTHLENS_DIR not in sys.path:
    sys.path.insert(0, _TRUTHLENS_DIR)

import numpy as np


# Reuse the abbreviation-aware regex from feature_engineer so sentence
# boundaries match what the rest of the pipeline already considers a
# "sentence" (avoids drift between the stylometric features and the
# Originality-style highlighter).
_ABBREV_PATTERN = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs|etc|U\.S|U\.K|U\.N|Jan|Feb|Mar|Apr'
    r'|Jun|Jul|Aug|Sep|Oct|Nov|Dec|St|Ave|Blvd|Corp|Dept|Gov|Sen|Rep)\.',
    re.IGNORECASE,
)
_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return [(sentence, char_start, char_end), ...]."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Mask abbreviations so we don't oversplit (Mr.Smith → MrSmith).
    masked = _ABBREV_PATTERN.sub(lambda m: m.group(1), text)
    sentences: list[tuple[str, int, int]] = []
    cursor = 0
    for chunk in _SPLIT_PATTERN.split(masked):
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            cursor += len(chunk) + 1
            continue
        # Find the original (un-masked) substring so char offsets line up
        # with what the frontend will render.
        idx = text.find(chunk_stripped, cursor)
        if idx < 0:
            idx = cursor
        end = idx + len(chunk_stripped)
        sentences.append((chunk_stripped, idx, end))
        cursor = end
    return sentences


def _classify_risk(prob_fake: float) -> str:
    if prob_fake >= 0.70:
        return "high"
    if prob_fake >= 0.45:
        return "medium"
    return "low"


def score_sentences(text, engine, model, preprocessor_module=None, top_k=None):
    """Score every sentence in `text` independently.

    Args:
        text: raw input string
        engine: a fitted TruthLensFeatureEngine
        model: any classifier exposing predict_proba returning (n, 2) where
               column 1 is the "fake" class
        preprocessor_module: optional reference to src.preprocessor; used to
               run the same cleaning + entity masking the model was trained
               with. If omitted, the function imports it lazily.
        top_k: if set, return only the top_k highest-prob_fake sentences

    Returns: list of dicts (see module docstring).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Lazy import to keep this script importable in unit-test contexts that
    # mock the model out and don't want to load spaCy.
    if preprocessor_module is None:
        from src import preprocessor as preprocessor_module  # type: ignore

    raw = [s for s, _, _ in sentences]

    # Run the same preprocessing the model was trained on. preprocess_dataframe
    # adds 'cleaned_text', 'masked_text', 'processed_text' columns.
    import pandas as pd
    df = pd.DataFrame([{"text": s} for s in raw])
    processed_df = preprocessor_module.preprocess_dataframe(df)
    if len(processed_df) == 0:
        return []

    p_text = processed_df["processed_text"].tolist()
    c_text = processed_df["cleaned_text"].tolist()
    r_text = processed_df["text"].tolist()

    X = engine.transform(p_text, c_text, r_text)
    probas = model.predict_proba(X)
    # Column index of the "fake" class. Standard sklearn binary models train
    # with labels [0, 1] so column 1 is fake. Fall back to column 0 if for
    # some reason classes_ is reversed.
    fake_col = 1
    if hasattr(model, "classes_") and len(model.classes_) == 2:
        fake_col = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else 1

    results = []
    for i, (sent, start, end) in enumerate(sentences):
        if i >= len(probas):
            break
        prob_fake = float(probas[i, fake_col])
        results.append({
            "sentence": sent,
            "prob_fake": round(prob_fake, 4),
            "prob_real": round(1.0 - prob_fake, 4),
            "risk": _classify_risk(prob_fake),
            "char_start": int(start),
            "char_end": int(end),
        })

    if top_k is not None:
        results = sorted(results, key=lambda r: r["prob_fake"], reverse=True)[:top_k]
    return results


if __name__ == "__main__":
    # Quick demo — requires trained models present
    import joblib
    from src.feature_engineer import TruthLensFeatureEngine
    import config as _cfg

    engine = TruthLensFeatureEngine().load(
        os.path.join(_cfg.MODELS_DIR, "feature_engine.pkl")
    )
    ensemble = joblib.load(os.path.join(_cfg.MODELS_DIR, "ensemble.joblib"))

    sample = (
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday. "
        "BREAKING: Scientists found a SHOCKING cure for cancer that doctors are HIDING! "
        "Markets closed slightly higher amid mixed earnings reports."
    )
    for row in score_sentences(sample, engine, ensemble):
        print(f"  [{row['risk']:>6}] {row['prob_fake']:.2f}  {row['sentence'][:80]}")
