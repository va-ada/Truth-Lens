"""
TruthLens — Predict
=====================
Load trained models and classify text instantly. No retraining needed.

Usage:
    python predict.py "Some news article or claim"
    python predict.py --file article.txt
    python predict.py --interactive
    python predict.py --all-models "text here"

Requires trained models in models/ directory (run main.py first).
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.preprocessor import clean_text, mask_entities, remove_stopwords_and_lemmatize
from src.feature_engineer import TruthLensFeatureEngine
from src.model_trainer import TruthLensTrainer


# ── Confidence classification ──────────────────────────────────────────────

def classify_confidence(probability):
    """
    Map prediction probability to a confidence level.

    Args:
        probability: float in [0.5, 1.0] — the probability of the predicted class

    Returns:
        (level, description) tuple
    """
    if probability >= 0.90:
        return "HIGH", "Strong signal — model is very confident"
    elif probability >= 0.75:
        return "MODERATE", "Reasonable confidence — likely correct"
    elif probability >= 0.60:
        return "LOW", "Weak signal — treat with caution"
    else:
        return "UNCERTAIN", "Near coin-flip — model cannot reliably classify this"


# ── Core prediction ────────────────────────────────────────────────────────

def predict_text(text, engine, trainer, model_name="Ensemble"):
    """
    Full prediction pipeline: raw text -> verdict + confidence.

    Args:
        text: Raw input text string
        engine: Fitted TruthLensFeatureEngine
        trainer: Fitted TruthLensTrainer with loaded models
        model_name: Which model to use for prediction

    Returns:
        dict with prediction, confidence, probabilities, metadata
    """
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
        return {
            "prediction": "UNKNOWN",
            "confidence_level": "UNCERTAIN",
            "confidence_desc": "Text became empty after preprocessing.",
            "confidence_pct": 50.0,
            "probability_real": 0.5,
            "probability_fake": 0.5,
            "word_count": len(text.split()),
            "model_used": model_name,
        }

    # Extract features and predict
    features = engine.transform([lemmatized], [cleaned_lower], [text])
    proba = trainer.predict_proba(features, model_name=model_name)[0]

    prediction = "FAKE" if proba[1] > 0.5 else "REAL"
    max_prob = max(proba)
    confidence_level, confidence_desc = classify_confidence(max_prob)

    return {
        "prediction": prediction,
        "confidence_level": confidence_level,
        "confidence_desc": confidence_desc,
        "confidence_pct": max_prob * 100,
        "probability_real": float(proba[0]),
        "probability_fake": float(proba[1]),
        "word_count": len(text.split()),
        "model_used": model_name,
    }


def predict_all_models(text, engine, trainer):
    """
    Run prediction with all available models for comparison.

    Returns:
        list of result dicts, one per model
    """
    model_names = list(trainer.base_models.keys())
    if trainer.ensemble is not None:
        model_names.append("Ensemble")

    results = []
    for name in model_names:
        result = predict_text(text, engine, trainer, model_name=name)
        results.append(result)

    return results


# ── Model loading ──────────────────────────────────────────────────────────

def load_models():
    """
    Load trained feature engine and models from disk.

    Returns:
        (engine, trainer) tuple

    Raises:
        FileNotFoundError if model files are missing
    """
    fe_path = os.path.join(config.MODELS_DIR, "feature_engine.pkl")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"Feature engine not found at {fe_path}. "
            "Run 'python main.py' first to train models."
        )

    engine = TruthLensFeatureEngine()
    engine.load()

    trainer = TruthLensTrainer()
    trainer.load()

    if trainer.ensemble is None and not trainer.base_models:
        raise FileNotFoundError(
            "No trained models found. Run 'python main.py' first."
        )

    return engine, trainer


# ── Output formatting ──────────────────────────────────────────────────────

def format_result(result):
    """Format a single prediction result for terminal output."""
    pred = result["prediction"]
    conf = result["confidence_level"]
    pct = result["confidence_pct"]

    lines = [
        "",
        "=" * 60,
        "  TruthLens — Prediction Result",
        "=" * 60,
        "",
        f"  Verdict:      {pred}",
        f"  Confidence:   {conf} ({pct:.1f}%)",
        f"  Reasoning:    {result['confidence_desc']}",
        "",
        f"  P(Real) = {result['probability_real']:.4f}",
        f"  P(Fake) = {result['probability_fake']:.4f}",
        "",
        f"  Words: {result['word_count']} | Model: {result['model_used']}",
    ]

    if conf == "UNCERTAIN":
        lines += [
            "",
            "  ** The model is NOT confident about this prediction. **",
            "  The text may be too short, ambiguous, or outside the",
            "  training distribution. Do not rely on this verdict.",
        ]

    lines += [
        "",
        "=" * 60,
        "  AI-generated analysis — review before citing.",
        "=" * 60,
    ]

    return "\n".join(lines)


def format_all_models(results):
    """Format multi-model comparison for terminal output."""
    lines = [
        "",
        "=" * 60,
        "  TruthLens — All Models Comparison",
        "=" * 60,
        "",
        f"  {'Model':<22} {'Verdict':<8} {'Confidence':<12} {'P(Fake)':>8}  {'P(Real)':>8}",
        "  " + "-" * 58,
    ]

    for r in results:
        lines.append(
            f"  {r['model_used']:<22} {r['prediction']:<8} "
            f"{r['confidence_level']:<12} {r['probability_fake']:>8.4f}  {r['probability_real']:>8.4f}"
        )

    # Agreement check
    predictions = [r["prediction"] for r in results]
    if len(set(predictions)) == 1:
        lines.append(f"\n  All models agree: {predictions[0]}")
    else:
        lines.append(f"\n  DISAGREEMENT: Models do not agree on the verdict.")
        lines.append(f"  Predictions: {', '.join(f'{r['model_used']}={r['prediction']}' for r in results)}")

    # Ensemble confidence
    ensemble = [r for r in results if r["model_used"] == "Ensemble"]
    if ensemble:
        e = ensemble[0]
        if e["confidence_level"] == "UNCERTAIN":
            lines.append(f"\n  ** Ensemble is UNCERTAIN — do not rely on this verdict. **")

    lines += [
        "",
        "=" * 60,
        "  AI-generated analysis — review before citing.",
        "=" * 60,
    ]

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TruthLens — Instant fake news prediction using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python predict.py "Scientists discover new cure for cancer"\n'
            "  python predict.py --file article.txt\n"
            "  python predict.py --interactive\n"
            '  python predict.py --all-models "BREAKING: Government hiding truth!!!"'
        ),
    )
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--file", "-f", help="Read text from a file")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode — keep predicting until 'quit'")
    parser.add_argument("--model", "-m", default="Ensemble",
                        choices=["Ensemble", "SVM", "LogisticRegression", "RandomForest"],
                        help="Model to use (default: Ensemble)")
    parser.add_argument("--all-models", "-a", action="store_true",
                        help="Show predictions from ALL models for comparison")
    parser.add_argument("--no-glove", action="store_true",
                        help="Disable GloVe (use if model was trained with --no-glove)")
    args = parser.parse_args()

    if args.no_glove:
        config.ENABLE_GLOVE = False

    # Load models
    print("[TruthLens] Loading models...")
    try:
        engine, trainer = load_models()
        print("[TruthLens] Models loaded.\n")
    except (FileNotFoundError, Exception) as e:
        print(f"\nERROR: {e}")
        print("Run 'python main.py' (or 'python main.py --quick --no-glove') first.")
        sys.exit(1)

    # Resolve input text
    if args.interactive:
        print("[TruthLens] Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                text = input("Enter text > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            if args.all_models:
                results = predict_all_models(text, engine, trainer)
                print(format_all_models(results))
            else:
                result = predict_text(text, engine, trainer, model_name=args.model)
                print(format_result(result))
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if not text:
        print("ERROR: No text provided.")
        sys.exit(1)

    # Predict
    if args.all_models:
        results = predict_all_models(text, engine, trainer)
        print(format_all_models(results))
    else:
        result = predict_text(text, engine, trainer, model_name=args.model)
        print(format_result(result))


if __name__ == "__main__":
    main()
