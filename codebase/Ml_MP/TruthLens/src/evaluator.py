"""
TruthLens — Evaluator
=======================
Comprehensive evaluation that goes BEYOND typical accuracy reporting:

1. In-domain evaluation (stratified k-fold, multiple metrics)
2. Cross-dataset evaluation (train ISOT → test LIAR and vice versa)
3. Bias probing (detect if models learn source/entity/length/topic shortcuts)
4. Confidence calibration (reliability diagram)
5. Error analysis (categorize misclassifications)
"""

import os
import re
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_in_domain(model, X, y, model_name="Model"):
    """
    Rigorous in-domain evaluation using stratified k-fold cross-validation.

    Reports: Accuracy, Precision, Recall, F1-Score, AUC-ROC
    (NOT just accuracy — because accuracy hides class imbalance issues)

    Args:
        model: Fitted sklearn model
        X: Feature matrix
        y: Labels
        model_name: Name for display

    Returns:
        dict of mean metrics
    """
    print(f"\n[EVAL] In-domain cross-validation for {model_name}...")

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    metrics = {}
    print(f"\n  {'Metric':<20} {'Train':>10} {'Val':>10} {'Std':>10}")
    print("  " + "-" * 52)

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        train_mean = results[f"train_{metric}"].mean()
        val_mean = results[f"test_{metric}"].mean()
        val_std = results[f"test_{metric}"].std()

        metrics[metric] = {
            "train_mean": train_mean,
            "val_mean": val_mean,
            "val_std": val_std,
        }

        print(f"  {metric:<20} {train_mean:>10.4f} {val_mean:>10.4f} {val_std:>10.4f}")

    # Check for overfitting
    gap = metrics["f1"]["train_mean"] - metrics["f1"]["val_mean"]
    if gap > 0.05:
        print(f"\n  WARNING: OVERFITTING — Train-Val F1 gap = {gap:.4f}")
    else:
        print(f"\n  Train-Val F1 gap = {gap:.4f} (acceptable)")

    return metrics


def evaluate_cross_dataset(trainer, feature_engine, datasets_dict):
    """
    Cross-dataset evaluation — the KEY TEST for generalization.

    Tests: Train on one dataset, evaluate on another.
    This catches models that "memorize" dataset-specific patterns.

    Args:
        trainer: TruthLensTrainer (unused — fresh trainers created per combination)
        feature_engine: TruthLensFeatureEngine (unused — fresh engines created)
        datasets_dict: {name: DataFrame} with 'processed_text', 'cleaned_text', 'text', 'label'

    Returns:
        DataFrame with cross-dataset results
    """
    print("\n" + "=" * 60)
    print("CROSS-DATASET EVALUATION")
    print("=" * 60)

    from src.feature_engineer import TruthLensFeatureEngine
    from src.model_trainer import TruthLensTrainer

    dataset_names = list(datasets_dict.keys())
    results = []

    for train_name in dataset_names:
        for test_name in dataset_names:
            if train_name == test_name:
                continue

            print(f"\n--- Train: {train_name.upper()} -> Test: {test_name.upper()} ---")

            train_df = datasets_dict[train_name]
            test_df = datasets_dict[test_name]

            # Create fresh feature engine for this combination
            fe = TruthLensFeatureEngine()

            # Fit on training data
            X_train = fe.fit_transform(
                train_df["processed_text"].tolist(),
                train_df["cleaned_text"].tolist(),
                train_df["text"].tolist(),
            )
            y_train = train_df["label"].values

            # Transform test data using training vocabulary
            X_test = fe.transform(
                test_df["processed_text"].tolist(),
                test_df["cleaned_text"].tolist(),
                test_df["text"].tolist(),
            )
            y_test = test_df["label"].values

            # Train a fresh ensemble
            t = TruthLensTrainer()
            t.train_base_models(X_train, y_train)
            t.train_ensemble(X_train, y_train)

            # Evaluate each model
            for model_name in list(t.base_models.keys()) + ["Ensemble"]:
                model = t._get_model(model_name)
                y_pred = model.predict(X_test)

                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = np.nan

                results.append({
                    "Train_On": train_name.upper(),
                    "Test_On": test_name.upper(),
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "AUC-ROC": auc,
                })

    results_df = pd.DataFrame(results)

    print("\n\n=== CROSS-DATASET RESULTS ===")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    # Save results
    results_df.to_csv(os.path.join(config.RESULTS_DIR, "cross_dataset_results.csv"), index=False)

    return results_df


def bias_probe(texts, labels, source_info=None):
    """
    Bias probing test: can a SIMPLE model predict labels from source/entity info alone?

    If a Naive Bayes trained on source names achieves >60% accuracy,
    the dataset has significant source bias.

    Args:
        texts: List of texts (raw or processed)
        labels: List of binary labels
        source_info: Optional list of source names

    Returns:
        dict with bias probe results
    """
    print("\n" + "=" * 60)
    print("BIAS PROBE ANALYSIS")
    print("=" * 60)

    results = {}

    # Probe 1: Can entity names alone predict the label?
    print("\n--- Probe 1: Named Entity Bias ---")
    try:
        import spacy
        nlp = spacy.load(config.SPACY_MODEL, disable=["parser", "tagger", "lemmatizer"])
    except Exception:
        nlp = None

    if nlp is not None:
        print("  Extracting entities from stratified sample (max 2000 texts)...")
        sample_size = min(2000, len(texts))

        # Stratified sampling instead of first-N (avoids ordering bias)
        from sklearn.model_selection import train_test_split
        if sample_size < len(texts):
            _, sample_idx = train_test_split(
                np.arange(len(texts)),
                test_size=sample_size / len(texts),
                stratify=labels,
                random_state=config.RANDOM_STATE,
            )
            sample_texts = [texts[i] for i in sample_idx]
            sample_labels = np.array(labels)[sample_idx]
        else:
            sample_texts = list(texts)
            sample_labels = np.array(labels)

        entity_texts = []
        for text in sample_texts:
            if isinstance(text, str):
                doc = nlp(text[:2000])  # Limit text length for speed
                entities = " ".join([ent.text for ent in doc.ents])
                entity_texts.append(entities)
            else:
                entity_texts.append("")

        # Train simple BOW + NB on entities only
        vec = CountVectorizer(max_features=5000)
        X_ent = vec.fit_transform(entity_texts)

        from sklearn.model_selection import cross_val_score
        nb = MultinomialNB()
        scores = cross_val_score(nb, X_ent, sample_labels, cv=5, scoring="accuracy")
        ent_accuracy = scores.mean()

        results["entity_bias_accuracy"] = ent_accuracy

        if ent_accuracy > config.BIAS_PROBE_THRESHOLD:
            print(f"  ENTITY BIAS DETECTED: NB accuracy = {ent_accuracy:.4f}")
            print(f"     (Threshold: {config.BIAS_PROBE_THRESHOLD})")
            print(f"     The model might be learning entity associations, not content patterns.")
        else:
            print(f"  Entity bias below threshold: NB accuracy = {ent_accuracy:.4f}")

    # Probe 2: Can text length alone predict the label?
    print("\n--- Probe 2: Text Length Bias ---")
    lengths = np.array([len(str(t)) for t in texts]).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    lr_probe = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    length_scores = cross_val_score(lr_probe, lengths, labels, cv=5, scoring="accuracy")
    length_accuracy = length_scores.mean()

    results["length_bias_accuracy"] = length_accuracy

    if length_accuracy > config.BIAS_PROBE_THRESHOLD:
        print(f"  LENGTH BIAS DETECTED: LR accuracy = {length_accuracy:.4f}")
    else:
        print(f"  Length bias below threshold: LR accuracy = {length_accuracy:.4f}")

    # Probe 3: Can topic alone predict the label?
    print("\n--- Probe 3: Topic Bias ---")
    topic_results = topic_bias_probe(texts, labels)
    results.update(topic_results)

    # Probe 4: Can the SOURCE NAME alone (not all entities) predict the label?
    # This is the headline bias probe behind A2 — the "Reuters detector" check.
    print("\n--- Probe 4: Source-Name-Only Bias (A2 headline metric) ---")
    src_results = source_only_bias_probe(texts, labels)
    results.update(src_results)

    return results


# News-source identifier regex — matches the well-known publishers our
# preprocessor's supplementary masker also targets. Defined at module level so
# both the bias probe and any future "find_bias_features" caller share the
# same dictionary.
_SOURCE_NAME_PATTERN = re.compile(
    r'\b(reuters|associated\s+press|\bap\b|afp|bbc(?:\s+news)?|cnn|fox\s+news|'
    r'msnbc|cbs|abc|nbc|new\s+york\s+times|nyt|washington\s+post|wapo|'
    r'guardian|telegraph|times|wall\s+street\s+journal|wsj|bloomberg|'
    r'al\s+jazeera|dw|france\s*24|npr|pbs|usa\s+today|forbes|'
    r'huffington\s+post|huffpost|breitbart|infowars|buzzfeed|vox|slate|'
    r'politico|axios|economist|atlantic|mother\s+jones|salon|daily\s+mail|'
    r'daily\s+caller|daily\s+wire|natural\s+news|gateway\s+pundit|rt\b|'
    r'sputnik|telesur|xinhua|tass|hindustan\s+times|indian\s+express|'
    r'times\s+of\s+india|the\s+hindu|ndtv)\b',
    re.IGNORECASE,
)
# Reuters-style dateline pattern: "WASHINGTON (Reuters) -"
_DATELINE_PATTERN = re.compile(
    r'^\s*[A-Z][A-Z\s]{2,30}\s*\([A-Za-z\s]+\)\s*[-–—]',
)


def _extract_source_features(text):
    """Pull out source-name indicators only — no other content.

    Returns a string like "reuters bbc dateline" so a count-vector
    classifier sees only the source signal, not the article body.
    """
    if not isinstance(text, str):
        return ""
    tokens = []
    for m in _SOURCE_NAME_PATTERN.finditer(text):
        tokens.append(m.group(1).lower().replace(" ", "_"))
    if _DATELINE_PATTERN.search(text):
        tokens.append("__DATELINE__")
    return " ".join(tokens) if tokens else "__no_source__"


def source_only_bias_probe(texts, labels, threshold=None):
    """Train a Logistic Regression on source-name features ALONE.

    This is the audit metric A2 reports before and after the debiasing loop.
    A high accuracy means the model could classify articles as fake or real
    using nothing but the publisher name — i.e. ISOT-style source leakage.

    Args:
        texts: list of raw or processed strings
        labels: binary labels (0=Real, 1=Fake) parallel to texts
        threshold: bias threshold; defaults to config.BIAS_PROBE_THRESHOLD

    Returns:
        dict with `source_only_bias_accuracy` and the boolean
        `source_only_bias_detected`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    threshold = config.BIAS_PROBE_THRESHOLD if threshold is None else threshold
    source_corpus = [_extract_source_features(t) for t in texts]
    # Vocabulary is small (~50 tokens). min_df=1 so every source survives.
    vec = CountVectorizer(min_df=1)
    try:
        X_src = vec.fit_transform(source_corpus)
    except ValueError:
        # All inputs collapsed to a single token (extremely small corpus)
        return {"source_only_bias_accuracy": 0.5, "source_only_bias_detected": False}

    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE,
                            class_weight="balanced")
    scores = cross_val_score(lr, X_src, labels, cv=5, scoring="accuracy")
    acc = float(scores.mean())
    detected = acc > threshold
    print(f"  Source-only LR accuracy: {acc:.4f} "
          f"({'BIAS DETECTED' if detected else 'OK'} — threshold {threshold})")
    return {
        "source_only_bias_accuracy": acc,
        "source_only_bias_detected": detected,
    }


def run_bias_audit(texts, labels, label="audit"):
    """Run the full bias_probe and return a flat dict.

    Used by main.py's `--debias-loop` orchestration to capture before/after
    snapshots that are written to results/bias_audit.txt.
    """
    print(f"\n[BIAS AUDIT — {label}] running probes on {len(texts)} samples")
    return bias_probe(list(texts), list(labels))


def find_bias_features(trainer, X_train, y_train, feature_names, top_n=50,
                       sample_size=300):
    """Identify feature columns that correlate with source-leakage bias.

    Strategy: run SHAP on the trainer's RandomForest base model (TreeExplainer
    is fast), then ask the existing `audit_bias` helper which features the
    SHAP values flag as suspicious (matches "person", "org", "loc", "reuters",
    etc. in the feature names). Returns the list of column INDICES so the
    debiased trainer can zero them out before fitting.
    """
    from src import explainer as _explainer

    rf = trainer.base_models.get("RandomForest")
    if rf is None:
        print("[BIAS] No RandomForest model available; skipping bias-feature discovery.")
        return []

    # Cap the sample for SHAP — TreeExplainer scales with n_samples * n_trees.
    if X_train.shape[0] > sample_size:
        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train

    shap_values, _ = _explainer.explain_with_shap(
        rf, X_sample, feature_names=feature_names,
    )
    if shap_values is None:
        print("[BIAS] SHAP unavailable; skipping bias-feature discovery.")
        return []

    audit = _explainer.audit_bias(shap_values, feature_names, top_n=top_n)
    suspicious = audit.get("suspicious_features", []) if isinstance(audit, dict) else []
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = []
    for entry in suspicious:
        # audit_bias may return either feature names or (name, importance) tuples.
        name = entry[0] if isinstance(entry, (list, tuple)) else entry
        if name in name_to_idx:
            indices.append(name_to_idx[name])
    print(f"[BIAS] {len(indices)} bias-correlated feature columns discovered.")
    return indices


def topic_bias_probe(texts, labels):
    """
    Test whether topic keywords alone can predict fake vs real.

    If a simple model on topic assignments exceeds the bias threshold,
    the model might be learning topic shortcuts (e.g. "politics = fake").

    Returns:
        dict with topic_bias_accuracy
    """
    topic_keywords = {
        "politics": ["president", "congress", "senate", "republican", "democrat",
                      "election", "vote", "government", "policy", "legislation"],
        "health": ["vaccine", "health", "doctor", "hospital", "disease",
                    "treatment", "medical", "virus", "pandemic", "cure"],
        "economy": ["economy", "stock", "market", "trade", "gdp",
                     "inflation", "interest rate", "bank", "financial", "tax"],
        "science": ["scientist", "research", "study", "discover", "climate",
                     "technology", "experiment", "evidence", "data", "journal"],
        "military": ["military", "army", "war", "attack", "defense",
                      "troops", "missile", "nato", "terrorism", "security"],
    }

    topic_assignments = []
    for text in texts:
        text_lower = str(text).lower()
        best_topic = "other"
        best_count = 0
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_count:
                best_count = matches
                best_topic = topic
        topic_assignments.append(best_topic)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    le = LabelEncoder()
    topic_encoded = le.fit_transform(topic_assignments).reshape(-1, 1)

    lr = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)
    scores = cross_val_score(lr, topic_encoded, labels, cv=5, scoring="accuracy")
    accuracy = scores.mean()

    if accuracy > config.BIAS_PROBE_THRESHOLD:
        print(f"  TOPIC BIAS DETECTED: LR accuracy = {accuracy:.4f}")
    else:
        print(f"  Topic bias below threshold: LR accuracy = {accuracy:.4f}")

    return {"topic_bias_accuracy": accuracy}


def plot_confusion_matrices(trainer, X_test, y_test, save_dir=None):
    """Plot confusion matrices for all models."""
    if save_dir is None:
        save_dir = config.PLOTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    all_models = list(trainer.base_models.keys())
    if trainer.ensemble is not None:
        all_models.append("Ensemble")

    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, all_models):
        model = trainer._get_model(name)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices — TruthLens Models", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Confusion matrices saved: {path}")

    return path


def plot_roc_curves(trainer, X_test, y_test, save_dir=None):
    """Plot ROC curves for all models."""
    if save_dir is None:
        save_dir = config.PLOTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    all_models = list(trainer.base_models.keys())
    if trainer.ensemble is not None:
        all_models.append("Ensemble")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for name, color in zip(all_models, colors):
        model = trainer._get_model(name)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.4f})")
        except Exception as e:
            print(f"[EVAL] WARNING: ROC curve failed for {name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — TruthLens Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] ROC curves saved: {path}")

    return path


def plot_calibration_curve(trainer, X_test, y_test, model_name="Ensemble", save_dir=None):
    """
    Plot reliability diagram showing predicted probability vs actual frequency.

    A well-calibrated model's curve should follow the diagonal.
    This is especially relevant since CalibratedClassifierCV is used for SVM.

    Args:
        trainer: TruthLensTrainer with fitted models
        X_test: Test feature matrix
        y_test: True test labels
        model_name: Which model to plot
        save_dir: Directory to save plot
    """
    if save_dir is None:
        save_dir = config.PLOTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    from sklearn.calibration import calibration_curve

    model = trainer._get_model(model_name)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"[EVAL] Calibration plot failed for {model_name}: {e}")
        return None

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color="#e74c3c",
             lw=2, label=f"{model_name}")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives (Actual)", fontsize=11)
    ax1.set_title(f"Calibration Curve — {model_name}", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Histogram of predicted probabilities
    ax2.hist(y_proba, bins=20, range=(0, 1), color="#3498db", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "calibration_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Calibration curve saved: {path}")

    return path


def analyze_errors(trainer, X_test, y_test, test_df, model_name="Ensemble", save_dir=None):
    """
    Categorize misclassified samples by text length, confidence, and direction.

    Helps identify systematic failure modes (e.g., "short texts are always wrong").

    Args:
        trainer: TruthLensTrainer
        X_test: Test feature matrix
        y_test: True labels
        test_df: Original test DataFrame (for text column)
        model_name: Which model to analyze

    Returns:
        DataFrame of misclassified samples with metadata
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    model = trainer._get_model(model_name)
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = np.full(len(y_test), 0.5)

    error_mask = y_pred != y_test
    n_errors = error_mask.sum()

    print(f"\n[EVAL] Error Analysis ({model_name})")
    print(f"  Total errors: {n_errors}/{len(y_test)} ({n_errors/len(y_test):.2%})")

    if n_errors == 0:
        print("  No errors to analyze.")
        return pd.DataFrame()

    errors = test_df.loc[error_mask].copy()
    errors["true_label"] = y_test[error_mask]
    errors["predicted"] = y_pred[error_mask]
    errors["confidence"] = y_proba[error_mask]
    errors["text_length"] = errors["text"].astype(str).str.len()

    # Categorize by text length
    errors["length_bin"] = pd.cut(
        errors["text_length"],
        bins=[0, 100, 500, 2000, float("inf")],
        labels=["short (<100)", "medium (100-500)", "long (500-2K)", "very long (2K+)"],
    )

    # Categorize by confidence
    errors["confidence_bin"] = pd.cut(
        errors["confidence"],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=["uncertain (<0.4)", "borderline (0.4-0.6)", "moderate (0.6-0.8)", "confident (0.8+)"],
    )

    # Categorize by error direction
    errors["error_type"] = errors.apply(
        lambda r: "False Positive (Real->Fake)" if r["true_label"] == 0 else "False Negative (Fake->Real)",
        axis=1,
    )

    print(f"\n  By text length:")
    print(errors["length_bin"].value_counts().to_string(dtype=False))
    print(f"\n  By confidence:")
    print(errors["confidence_bin"].value_counts().to_string(dtype=False))
    print(f"\n  By error type:")
    print(errors["error_type"].value_counts().to_string(dtype=False))

    # Save
    save_cols = ["text", "true_label", "predicted", "confidence", "text_length",
                 "length_bin", "confidence_bin", "error_type"]
    save_cols = [c for c in save_cols if c in errors.columns]
    path = os.path.join(save_dir, "error_analysis.csv")
    errors[save_cols].to_csv(path, index=False)
    print(f"\n[EVAL] Error analysis saved: {path}")

    return errors


def generate_results_report(trainer, in_domain_metrics, cross_dataset_df,
                            bias_results, save_dir=None):
    """
    Generate a comprehensive results report as a text file.
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    lines = []
    lines.append("=" * 70)
    lines.append("TruthLens — Results Report")
    lines.append("Debiased, Explainable Fake News Detection")
    lines.append("=" * 70)

    # Reproducibility metadata
    lines.append("\n\n## ENVIRONMENT")
    lines.append("-" * 50)
    try:
        import sklearn, numpy, pandas, spacy
        lines.append(f"  scikit-learn: {sklearn.__version__}")
        lines.append(f"  numpy: {numpy.__version__}")
        lines.append(f"  pandas: {pandas.__version__}")
        lines.append(f"  spacy: {spacy.__version__}")
    except Exception:
        pass
    lines.append(f"  random_state: {config.RANDOM_STATE}")
    lines.append(f"  quick_mode: {config.QUICK_MODE}")
    lines.append(f"  entity_masking: {config.ENABLE_ENTITY_MASKING}")
    lines.append(f"  glove: {config.ENABLE_GLOVE}")

    # Training results
    lines.append("\n\n## TRAINING RESULTS")
    lines.append("-" * 50)
    for name, res in trainer.training_results.items():
        lines.append(f"\n{name}:")
        for k, v in res.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

    # In-domain results
    if in_domain_metrics:
        lines.append("\n\n## IN-DOMAIN EVALUATION (Stratified 5-Fold CV)")
        lines.append("-" * 50)
        for metric, vals in in_domain_metrics.items():
            lines.append(f"  {metric:<20} Val: {vals['val_mean']:.4f} +/- {vals['val_std']:.4f}")

    # Cross-dataset results
    if cross_dataset_df is not None:
        lines.append("\n\n## CROSS-DATASET EVALUATION")
        lines.append("-" * 50)
        lines.append("(This is the TRUE test of generalization)")
        lines.append(cross_dataset_df.to_string(index=False, float_format="{:.4f}".format))

    # Bias probe
    if bias_results:
        lines.append("\n\n## BIAS PROBE RESULTS")
        lines.append("-" * 50)
        for k, v in bias_results.items():
            status = "BIAS" if v > config.BIAS_PROBE_THRESHOLD else "OK"
            lines.append(f"  {k}: {v:.4f} [{status}]")

    report_text = "\n".join(lines)

    path = os.path.join(save_dir, "results_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[EVAL] Results report saved: {path}")
    return report_text
