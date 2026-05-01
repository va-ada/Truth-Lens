"""
TruthLens — Interactive Demo
==============================
Unified interface: paste any news article, claim, question, or date
to get real-time verification.

  - Articles (>50 words)  → ML pipeline (TF-IDF + GloVe + Ensemble)
  - Claims / Questions    → Fact-verification (temporal + Wikipedia + web)
  - Temporal claims       → Date/time verification against system clock
  - Opinions              → Flagged as unverifiable

Usage:
    pip install gradio
    python app.py

Requires trained models in models/ directory (run main.py first).
"""

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.preprocessor import clean_text, mask_entities, remove_stopwords_and_lemmatize
from src.feature_engineer import TruthLensFeatureEngine
from src.model_trainer import TruthLensTrainer
from src.claim_detector import classify_input
from src.fact_checker import check_claim
from predict import classify_confidence

# ── Load ML models at startup ────────────────────────────────────────────────
_ml_available = False
try:
    print("[DEMO] Loading ML models...")
    engine = TruthLensFeatureEngine()
    engine.load()
    trainer = TruthLensTrainer()
    trainer.load()
    _ml_available = True
    print("[DEMO] ML models loaded.")
except Exception as e:
    print(f"[DEMO] ML models not available ({e}). Article classification disabled.")
    print("[DEMO] Fact-checking (claims/questions/temporal) still works.")


# ── ML-based article prediction ─────────────────────────────────────────────

def predict_article(text: str) -> tuple[dict, str]:
    """Full ML pipeline: raw text -> prediction + confidence + explanation."""
    if not _ml_available:
        return (
            {"REAL": 0.5, "FAKE": 0.5},
            "ML models not loaded. Run `python main.py` first to train models.",
        )

    # Preprocess — same pipeline as training
    cleaned_mixed = clean_text(text, lowercase=False)
    if config.ENABLE_ENTITY_MASKING:
        masked = mask_entities(cleaned_mixed)
    else:
        masked = cleaned_mixed
    masked_lower = masked.lower()
    lemmatized = remove_stopwords_and_lemmatize(masked_lower)
    cleaned_lower = cleaned_mixed.lower()

    if not lemmatized.strip():
        return {"REAL": 0.5, "FAKE": 0.5}, "Text became empty after preprocessing."

    features = engine.transform([lemmatized], [cleaned_lower], [text])
    proba = trainer.predict_proba(features, model_name="Ensemble")[0]
    label = "FAKE" if proba[1] > 0.5 else "REAL"
    max_prob = max(proba)
    confidence_level, confidence_desc = classify_confidence(max_prob)
    confidence_pct = max_prob * 100

    explanation = f"### Style Analysis: **{label}**\n\n"
    explanation += f"**Confidence:** {confidence_level} ({confidence_pct:.1f}%) — {confidence_desc}\n\n"
    explanation += "| Class | Probability |\n|---|---|\n"
    explanation += f"| Real | {proba[0]:.4f} |\n| Fake | {proba[1]:.4f} |\n\n"

    # Uncertainty warning
    if confidence_level == "UNCERTAIN":
        explanation += (
            "> **Warning:** The model is not confident about this classification. "
            "The text may be too short, ambiguous, or outside the training "
            "distribution. Do not rely on this verdict.\n\n"
        )

    # Stylometric indicators
    words = text.split()
    excl_count = text.count("!")
    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    indicators = []
    if excl_count >= 3:
        indicators.append("Excessive punctuation (!) — sensationalism signal")
    if caps > 0.25:
        indicators.append("Heavy capitalization — emotional emphasis signal")
    if len(words) < 20:
        indicators.append("Very short text — classification confidence may be lower")
    if indicators:
        explanation += "**Style Indicators:**\n"
        for ind in indicators:
            explanation += f"- {ind}\n"

    explanation += "\n---\n*AI-generated analysis — review before citing.*"
    return {"REAL": float(proba[0]), "FAKE": float(proba[1])}, explanation


# ── Fact-verification for claims/questions/temporal ──────────────────────────

def predict_claim(text: str, claim_info: dict) -> tuple[dict, str]:
    """Fact-check a claim/question using temporal + Wikipedia + web search."""
    try:
        result = check_claim(text, claim_info)
    except Exception as e:
        return (
            {"Error": 1.0},
            f"Fact-checking failed: {e}\n\n{traceback.format_exc()}",
        )

    verdict = result.get("final_verdict", "Unknown")
    conf = result.get("confidence", 0.0)
    truth = result.get("truth_score", 0.5)

    # Build label dict for gr.Label
    verdict_map = {
        "Verified": {"VERIFIED": conf, "FALSE": 1 - conf},
        "Likely True": {"LIKELY TRUE": conf, "UNCERTAIN": 1 - conf},
        "Uncertain": {"UNCERTAIN": conf, "NEEDS MORE INFO": 1 - conf},
        "Likely False": {"LIKELY FALSE": conf, "UNCERTAIN": 1 - conf},
        "False": {"FALSE": conf, "VERIFIED": 1 - conf},
        "Cannot Verify": {"CANNOT VERIFY": 1.0},
    }
    label_dict = verdict_map.get(verdict, {"UNCERTAIN": 1.0})

    # Build explanation
    md = f"### Verdict: **{verdict}**\n"
    md += f"**Confidence:** {conf:.0%} | **Truth Score:** {truth}\n\n"

    # Evidence trail
    trail = result.get("evidence_trail", [])
    if trail:
        md += "### Evidence\n\n"
        for ev in trail:
            src = ev.get("source", "Unknown")
            vrd = ev.get("verdict", "")
            evd = ev.get("evidence", "")
            url = ev.get("url", "")
            md += f"**{src}** — {vrd}\n"
            md += f"> {evd}\n\n"
            if url:
                md += f"Source: {url}\n\n"

            # Show web sources if available
            for s in ev.get("sources", []):
                title = s.get("title", "")
                surl = s.get("url", "")
                stance = s.get("stance", "")
                snippet = s.get("snippet", "")[:100]
                md += f"- [{title}]({surl}) *({stance})* — {snippet}...\n"
            if ev.get("sources"):
                md += "\n"

    # Input classification info
    md += "### Input Analysis\n"
    md += f"- **Type detected:** {claim_info.get('type', 'unknown')}\n"
    if claim_info.get("extracted_dates"):
        md += f"- **Dates found:** {', '.join(str(d) for d in claim_info['extracted_dates'])}\n"
    if claim_info.get("extracted_numbers"):
        md += f"- **Numbers found:** {claim_info['extracted_numbers']}\n"

    md += f"\n---\n*{result.get('disclaimer', 'AI-generated analysis — verify independently.')}*"

    return label_dict, md


# ── Unified entry point ─────────────────────────────────────────────────────

def predict(text: str) -> tuple[dict, str]:
    """Auto-detect input type and route to the correct pipeline."""
    if not text or len(text.strip()) < 3:
        return {"N/A": 1.0}, "Please enter at least a few words of text."

    # Classify input
    claim_info = classify_input(text)
    input_type = claim_info["type"]

    # Route based on classification
    if input_type == "article":
        return predict_article(text)

    elif input_type == "opinion":
        return (
            {"OPINION": 0.8, "UNVERIFIABLE": 0.2},
            f"### Opinion Detected\n\n"
            f"This text contains subjective language and cannot be fact-checked.\n\n"
            f"Detected markers: subjective phrasing\n\n"
            f"---\n*Opinions are not verifiable claims.*",
        )

    else:
        # temporal, claim, question → fact verification
        return predict_claim(text, claim_info)


# ── Gradio UI ────────────────────────────────────────────────────────────────

def main():
    import gradio as gr

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=8,
            placeholder=(
                "Paste a news article, claim, or question...\n\n"
                "Examples:\n"
                "  - 'Is today April 11th 2026?'\n"
                "  - 'India landed on the moon in 2023'\n"
                "  - 'The capital of France is Paris'\n"
                "  - Paste a full news article for style analysis"
            ),
            label="Input Text",
        ),
        outputs=[
            gr.Label(label="Verdict", num_top_classes=2),
            gr.Markdown(label="Explanation"),
        ],
        title="TruthLens — Fake News & Claim Verifier",
        description=(
            "**Unified fact-checking pipeline.** Paste any text — TruthLens "
            "auto-detects the type and routes it:\n\n"
            "- **Articles** (>50 words) → ML style analysis (TF-IDF + GloVe + Ensemble)\n"
            "- **Claims / Questions** → Fact verification (Wikipedia + Web search)\n"
            "- **Temporal claims** → Date/time verification (system clock)\n"
            "- **Opinions** → Flagged as unverifiable"
        ),
        examples=[
            ["Is today April 11th 2026?"],
            ["Was yesterday Thursday?"],
            ["The capital of France is Paris"],
            ["India landed on the moon in 2023"],
            ["Did NASA find water on Mars?"],
            ["I think the government is hiding something"],
            ["Scientists at NASA have discovered evidence of water on Mars using new spectroscopic analysis techniques published in the journal Nature."],
            ["BREAKING!!! Government HIDING the truth about vaccines!!! They don't want you to know!!! Share before they delete this!!!"],
        ],
        flagging_mode="never",
    )

    demo.launch()


if __name__ == "__main__":
    main()
