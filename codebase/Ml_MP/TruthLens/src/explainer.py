"""
TruthLens — Explainability Module
====================================
Provides LIME and SHAP explanations for model predictions.

Key features:
  - LIME text explanations: highlights words that triggered fake/real classification
  - SHAP feature importance: global and local explanations
  - Bias auditing: flags if entity/source names appear as top features
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def explain_with_lime(text, predict_fn, num_features=None, num_samples=None):
    """
    Generate LIME explanation for a single text prediction.

    LIME perturbs the input text and observes how predictions change,
    identifying which words contribute most to the classification.

    Args:
        text: Raw or cleaned text string
        predict_fn: Function that takes list of strings → probabilities array
        num_features: Number of top features to show
        num_samples: Number of perturbation samples

    Returns:
        lime Explanation object
    """
    from lime.lime_text import LimeTextExplainer

    if num_features is None:
        num_features = config.LIME_NUM_FEATURES
    if num_samples is None:
        num_samples = config.LIME_NUM_SAMPLES

    explainer = LimeTextExplainer(
        class_names=["Real", "Fake"],
        split_expression=r"\W+",
        random_state=config.RANDOM_STATE,
    )

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    return explanation


def explain_with_shap(model, X, feature_names=None, max_samples=None, num_explain=None):
    """
    Generate SHAP explanations for model predictions.

    SHAP provides both global (overall feature importance) and
    local (per-instance) explanations based on game theory.

    Args:
        model: Fitted sklearn model
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        max_samples: Number of background samples for KernelExplainer
        num_explain: Number of instances to explain

    Returns:
        shap_values, explainer object
    """
    import shap

    if max_samples is None:
        max_samples = config.SHAP_MAX_SAMPLES
    if num_explain is None:
        num_explain = config.SHAP_NUM_EXPLAIN

    # Use a sample for background data (KernelExplainer needs this)
    n_bg = min(max_samples, len(X))
    bg_indices = np.random.RandomState(config.RANDOM_STATE).choice(
        len(X), n_bg, replace=False
    )
    background = X[bg_indices]

    # Determine which SHAP explainer to use
    model_type = type(model).__name__

    try:
        if model_type == "RandomForestClassifier":
            explainer = shap.TreeExplainer(model)
            n_explain = min(num_explain, len(X))
            shap_values = explainer.shap_values(X[:n_explain])
            # For binary classification, take values for class 1 (Fake)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif model_type == "LogisticRegression":
            explainer = shap.LinearExplainer(model, background)
            n_explain = min(num_explain, len(X))
            shap_values = explainer.shap_values(X[:n_explain])
        else:
            # KernelExplainer for SVM, ensemble, or any model
            def predict_proba_fn(x):
                return model.predict_proba(x)

            explainer = shap.KernelExplainer(predict_proba_fn, background)
            n_explain = min(num_explain, len(X))
            shap_values = explainer.shap_values(X[:n_explain])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

    except Exception as e:
        import traceback
        warnings.warn(f"SHAP explanation failed ({type(e).__name__}): {e}")
        traceback.print_exc()
        return None, None

    return shap_values, explainer


def plot_shap_summary(shap_values, X, feature_names=None, save_path=None, title="SHAP Feature Importance"):
    """
    Plot SHAP summary (beeswarm) showing global feature importance.

    This reveals which features the model relies on most.
    If entity-related features appear at the top, it indicates BIAS.
    """
    import shap

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "shap_summary.png")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize list output from binary TreeExplainer → single 2D array (class 1 = Fake)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    n_explain = min(len(shap_values), len(X))
    shap.summary_plot(
        shap_values[:n_explain],
        X[:n_explain],
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[EXPLAIN] SHAP summary plot saved: {save_path}")
    return save_path


def plot_shap_bar(shap_values, feature_names=None, save_path=None, top_n=20):
    """Plot SHAP mean absolute values as bar chart."""
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "shap_bar.png")

    # TreeExplainer on binary classifiers returns a list [class_0_shap, class_1_shap].
    # Extract class-1 (Fake) values; average both classes if ambiguous.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]          # shape: (n_samples, n_features)
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]   # shape: (n_samples, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)   # shape: (n_features,)

    feature_names = list(feature_names) if feature_names is not None else \
        [f"feature_{i}" for i in range(len(mean_abs_shap))]

    # Get top N features
    top_indices = np.argsort(mean_abs_shap)[-top_n:]
    top_names = [feature_names[int(i)] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_names)), top_values, color="#3498db")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features by SHAP Importance", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[EXPLAIN] SHAP bar plot saved: {save_path}")
    return save_path


def audit_bias(shap_values, feature_names, top_n=20):
    """
    Audit model for bias by checking if entity-related features
    appear in the top SHAP features.

    If features like person names, organization names, or source-specific
    terms dominate, the model has learned shortcuts instead of content patterns.

    Args:
        shap_values: SHAP values matrix
        feature_names: List of feature names
        top_n: Number of top features to check

    Returns:
        dict with bias audit results
    """
    print("\n[EXPLAIN] Bias Audit via SHAP Feature Importance")
    print("-" * 50)

    # Normalize list/3D output from binary TreeExplainer → single 2D array (class 1 = Fake)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    # Check for suspicious feature names
    suspicious_keywords = [
        "person", "org", "loc", "gpe", "reuters", "trump",
        "obama", "clinton", "fox", "cnn", "bbc",
    ]

    suspicious_features = []
    print(f"\n  Top {top_n} features by importance:")
    for rank, idx in enumerate(top_indices):
        idx = int(idx)
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        value = mean_abs_shap[idx]

        is_suspicious = any(kw in name.lower() for kw in suspicious_keywords)
        flag = " 🔴 BIAS?" if is_suspicious else ""

        print(f"    {rank+1:>3}. {name:<35} {value:.6f}{flag}")

        if is_suspicious:
            suspicious_features.append(name)

    results = {
        "top_features": [feature_names[int(i)] for i in top_indices if int(i) < len(feature_names)],
        "suspicious_features": suspicious_features,
        "bias_detected": len(suspicious_features) > 0,
    }

    if suspicious_features:
        print(f"\n  🔴 BIAS WARNING: {len(suspicious_features)} suspicious features detected!")
        print(f"     Entity masking may need improvement.")
    else:
        print(f"\n  ✅ No obvious entity/source bias in top features.")

    return results


def save_lime_explanation(explanation, save_path=None, text=None, prediction=None):
    """Save LIME explanation as HTML file for easy viewing."""
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "lime_explanation.html")

    html = explanation.as_html()

    # Add custom header with prediction info
    header = """
    <div style="padding: 20px; background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
        <h2 style="color: #2c3e50;">TruthLens — LIME Explanation</h2>
    """
    if text:
        import html as _html_mod
        header += f'<p><b>Text:</b> {_html_mod.escape(text[:300])}...</p>'
    if prediction is not None:
        label = "FAKE" if prediction == 1 else "REAL"
        color = "#e74c3c" if prediction == 1 else "#27ae60"
        header += f'<p><b>Prediction:</b> <span style="color:{color};font-weight:bold">{label}</span></p>'
    header += "</div>"

    html = html.replace("<body>", f"<body>{header}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[EXPLAIN] LIME explanation saved: {save_path}")
    return save_path


def create_prediction_explainer(feature_engine, preprocessor_module, model):
    """
    Create a predict function for LIME that goes from raw text → prediction.

    This wraps the entire pipeline (preprocess → features → predict) so LIME
    can perturb the original text and see how predictions change.

    Args:
        feature_engine: Fitted TruthLensFeatureEngine
        preprocessor_module: The preprocessor module (for clean_text, mask_entities)
        model: Fitted sklearn model

    Returns:
        predict_fn: function(list[str]) → np.ndarray of shape (n, 2)
    """

    # Capture masking setting at creation time (not inference time)
    _masking_enabled = config.ENABLE_ENTITY_MASKING

    def predict_fn(texts):
        """Takes raw texts, returns prediction probabilities."""
        processed = []
        cleaned = []

        for text in texts:
            # Clean WITHOUT lowercasing so NER sees proper capitalization
            c_mixed = preprocessor_module.clean_text(text, lowercase=False)
            m = preprocessor_module.mask_entities(c_mixed) if _masking_enabled else c_mixed
            m_lower = m.lower()
            lemmatized = preprocessor_module.remove_stopwords_and_lemmatize(m_lower)
            processed.append(lemmatized)
            cleaned.append(c_mixed.lower())  # GloVe needs lowercase, unmasked

        features = feature_engine.transform(processed, cleaned, list(texts))
        return model.predict_proba(features)

    return predict_fn
