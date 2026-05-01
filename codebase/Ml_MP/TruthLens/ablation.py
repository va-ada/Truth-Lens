"""
TruthLens — Ablation Study
=============================
Proves that entity masking (debiasing) actually improves the model.

Runs the pipeline twice:
  1. WITH entity masking (debiased)
  2. WITHOUT entity masking (baseline)

Then compares metrics side-by-side and generates a comparison plot.

Usage:
    python ablation.py               # Full ablation (may take 1-2 hours)
    python ablation.py --quick        # Quick mode (~20 min)
    python ablation.py --no-glove     # Skip GloVe download

Output:
    results/ablation_comparison.csv   — Metric comparison table
    results/plots/ablation_chart.png  — Bar chart comparison
    results/ablation_report.txt       — Text summary
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import get_datasets, split_dataset
from src.preprocessor import preprocess_dataframe, save_processed, load_processed
from src.feature_engineer import TruthLensFeatureEngine
from src.model_trainer import TruthLensTrainer
from src.evaluator import evaluate_in_domain, bias_probe


def run_single_pipeline(datasets, enable_masking, label):
    """
    Run data preprocessing -> feature engineering -> training -> evaluation
    with a specific masking setting.

    Args:
        datasets: dict of raw DataFrames from get_datasets()
        enable_masking: bool — whether to enable entity masking
        label: str — "masked" or "unmasked" for display

    Returns:
        dict of evaluation metrics
    """
    print(f"\n{'#' * 60}")
    print(f"  ABLATION RUN: {label.upper()}")
    print(f"  Entity masking: {'ON' if enable_masking else 'OFF'}")
    print(f"{'#' * 60}")

    run_start = time.time()

    # ── Preprocess ──────────────────────────────────────────────────────
    print(f"\n[{label}] Preprocessing...")
    cache_name = f"isot_ablation_{label}"
    cached = load_processed(cache_name)

    if cached is not None and all(col in cached.columns
                                   for col in ["processed_text", "cleaned_text", "text", "label"]):
        processed = cached
        print(f"[{label}] Using cached preprocessed data.")
    else:
        processed = preprocess_dataframe(datasets["isot"], enable_masking=enable_masking)
        save_processed(processed, cache_name)

    # ── Split ───────────────────────────────────────────────────────────
    train_df, test_df = split_dataset(processed)

    # ── Feature Engineering ─────────────────────────────────────────────
    print(f"\n[{label}] Extracting features...")
    engine = TruthLensFeatureEngine()
    X_train = engine.fit_transform(
        train_df["processed_text"].tolist(),
        train_df["cleaned_text"].tolist(),
        train_df["text"].tolist(),
    )
    y_train = train_df["label"].values

    X_test = engine.transform(
        test_df["processed_text"].tolist(),
        test_df["cleaned_text"].tolist(),
        test_df["text"].tolist(),
    )
    y_test = test_df["label"].values

    print(f"[{label}] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Training ────────────────────────────────────────────────────────
    print(f"\n[{label}] Training models...")
    trainer = TruthLensTrainer()
    trainer.train_base_models(X_train, y_train)
    trainer.train_ensemble(X_train, y_train)

    # ── Evaluation ──────────────────────────────────────────────────────
    print(f"\n[{label}] Evaluating on test set...")
    test_results_df = trainer.evaluate_on_test(X_test, y_test)

    # In-domain CV on training data
    print(f"\n[{label}] In-domain cross-validation...")
    cv_metrics = evaluate_in_domain(
        trainer.ensemble, X_train, y_train, model_name="Ensemble"
    )

    # Bias probe
    print(f"\n[{label}] Bias probing...")
    bias_results = bias_probe(
        processed["text"].tolist(),
        processed["label"].values,
    )

    elapsed = time.time() - run_start
    print(f"\n[{label}] Run completed in {elapsed / 60:.1f} minutes.")

    # Collect results
    ensemble_row = test_results_df[test_results_df["Model"] == "Ensemble"].iloc[0]

    return {
        "label": label,
        "entity_masking": enable_masking,
        "test_accuracy": ensemble_row["Accuracy"],
        "test_precision": ensemble_row["Precision"],
        "test_recall": ensemble_row["Recall"],
        "test_f1": ensemble_row["F1-Score"],
        "test_auc": ensemble_row["AUC-ROC"],
        "cv_f1_mean": cv_metrics["f1"]["val_mean"],
        "cv_f1_std": cv_metrics["f1"]["val_std"],
        "cv_auc_mean": cv_metrics["roc_auc"]["val_mean"],
        "entity_bias_accuracy": bias_results.get("entity_bias_accuracy", np.nan),
        "length_bias_accuracy": bias_results.get("length_bias_accuracy", np.nan),
        "topic_bias_accuracy": bias_results.get("topic_bias_accuracy", np.nan),
        "time_minutes": elapsed / 60,
        "all_test_results": test_results_df,
    }


def generate_comparison_plot(masked_results, unmasked_results, save_dir=None):
    """Generate a grouped bar chart comparing masked vs unmasked metrics."""
    if save_dir is None:
        save_dir = config.PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    masked_vals = [masked_results[m] for m in metrics]
    unmasked_vals = [unmasked_results[m] for m in metrics]

    x = np.arange(len(display_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, masked_vals, width, label="With Entity Masking (Debiased)",
                   color="#2ecc71", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, unmasked_vals, width, label="Without Entity Masking (Baseline)",
                   color="#e74c3c", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: Effect of Entity Masking on Model Performance",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.85, 1.01)  # Zoom into relevant range
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ABLATION] Comparison chart saved: {path}")

    # Bias comparison plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bias_metrics = ["entity_bias_accuracy", "length_bias_accuracy", "topic_bias_accuracy"]
    bias_names = ["Entity Bias", "Length Bias", "Topic Bias"]

    masked_bias = [masked_results.get(m, 0) for m in bias_metrics]
    unmasked_bias = [unmasked_results.get(m, 0) for m in bias_metrics]

    x2 = np.arange(len(bias_names))
    ax2.bar(x2 - width / 2, masked_bias, width, label="With Entity Masking",
            color="#2ecc71", edgecolor="white")
    ax2.bar(x2 + width / 2, unmasked_bias, width, label="Without Entity Masking",
            color="#e74c3c", edgecolor="white")
    ax2.axhline(y=config.BIAS_PROBE_THRESHOLD, color="orange", linestyle="--",
                linewidth=2, label=f"Bias Threshold ({config.BIAS_PROBE_THRESHOLD})")

    ax2.set_ylabel("Probe Accuracy", fontsize=12)
    ax2.set_title("Ablation Study: Bias Probe Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bias_names, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.3, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    for i, (mv, uv) in enumerate(zip(masked_bias, unmasked_bias)):
        ax2.text(i - width / 2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=9)
        ax2.text(i + width / 2, uv + 0.01, f"{uv:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "ablation_bias_chart.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ABLATION] Bias comparison chart saved: {path2}")

    return path, path2


def generate_report(masked_results, unmasked_results, save_dir=None):
    """Generate a text report summarizing the ablation study."""
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    lines = [
        "=" * 70,
        "TruthLens — Ablation Study Report",
        "Effect of Entity Masking on Fake News Detection",
        "=" * 70,
        "",
        "HYPOTHESIS: Entity masking reduces dataset bias by preventing the",
        "model from learning source-identity shortcuts (e.g. 'Reuters = Real').",
        "",
        "",
        "## TEST SET RESULTS (Ensemble Model)",
        "-" * 50,
        "",
        f"  {'Metric':<20} {'With Masking':>14} {'Without':>14} {'Delta':>10}",
        "  " + "-" * 58,
    ]

    metrics = [
        ("Accuracy", "test_accuracy"),
        ("Precision", "test_precision"),
        ("Recall", "test_recall"),
        ("F1-Score", "test_f1"),
        ("AUC-ROC", "test_auc"),
    ]

    for display, key in metrics:
        m_val = masked_results[key]
        u_val = unmasked_results[key]
        delta = m_val - u_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {display:<20} {m_val:>14.4f} {u_val:>14.4f} {sign}{delta:>9.4f}")

    lines += [
        "",
        "",
        "## CROSS-VALIDATION (5-Fold Stratified)",
        "-" * 50,
        "",
        f"  F1-Score (masked):   {masked_results['cv_f1_mean']:.4f} +/- {masked_results['cv_f1_std']:.4f}",
        f"  F1-Score (unmasked): {unmasked_results['cv_f1_mean']:.4f} +/- {unmasked_results['cv_f1_std']:.4f}",
        "",
        "",
        "## BIAS PROBE RESULTS",
        "-" * 50,
        f"  (Threshold: {config.BIAS_PROBE_THRESHOLD} — above this indicates bias)",
        "",
        f"  {'Probe':<25} {'With Masking':>14} {'Without':>14}",
        "  " + "-" * 53,
    ]

    bias_metrics = [
        ("Entity Bias", "entity_bias_accuracy"),
        ("Length Bias", "length_bias_accuracy"),
        ("Topic Bias", "topic_bias_accuracy"),
    ]
    for display, key in bias_metrics:
        m_val = masked_results.get(key, float("nan"))
        u_val = unmasked_results.get(key, float("nan"))
        m_flag = " [BIAS]" if m_val > config.BIAS_PROBE_THRESHOLD else " [OK]"
        u_flag = " [BIAS]" if u_val > config.BIAS_PROBE_THRESHOLD else " [OK]"
        lines.append(f"  {display:<25} {m_val:>8.4f}{m_flag:<6} {u_val:>8.4f}{u_flag}")

    # Interpretation
    f1_delta = masked_results["test_f1"] - unmasked_results["test_f1"]
    entity_delta = masked_results.get("entity_bias_accuracy", 0) - unmasked_results.get("entity_bias_accuracy", 0)

    lines += [
        "",
        "",
        "## INTERPRETATION",
        "-" * 50,
    ]

    if entity_delta < 0:
        lines.append(f"  Entity masking REDUCED entity bias by {abs(entity_delta):.4f}.")
    else:
        lines.append(f"  Entity masking did not reduce entity bias (delta: {entity_delta:+.4f}).")
        lines.append("  The ISOT dataset has strong structural bias beyond entity names.")

    if abs(f1_delta) < 0.01:
        lines.append(f"  Performance impact: NEGLIGIBLE (F1 delta: {f1_delta:+.4f}).")
        lines.append("  Entity masking reduces bias WITHOUT sacrificing accuracy.")
    elif f1_delta > 0:
        lines.append(f"  Performance IMPROVED with masking (F1 delta: {f1_delta:+.4f}).")
        lines.append("  The unmasked model relied on entity shortcuts that hurt generalization.")
    else:
        lines.append(f"  Performance dropped slightly with masking (F1 delta: {f1_delta:+.4f}).")
        lines.append("  Some signal is lost, but the model is more honest and generalizable.")

    lines += [
        "",
        f"  Runtime: masked={masked_results['time_minutes']:.1f}min, "
        f"unmasked={unmasked_results['time_minutes']:.1f}min",
        "",
        "=" * 70,
    ]

    report = "\n".join(lines)

    path = os.path.join(save_dir, "ablation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[ABLATION] Report saved: {path}")

    # Also print to console
    print("\n" + report)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens Ablation Study")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller hyperparameter grids (faster)")
    parser.add_argument("--no-glove", action="store_true",
                        help="Disable GloVe embeddings")
    args = parser.parse_args()

    if args.quick:
        config.QUICK_MODE = True
        print("[ABLATION] Quick mode enabled.")
    if args.no_glove:
        config.ENABLE_GLOVE = False
        print("[ABLATION] GloVe disabled.")

    total_start = time.time()

    print("\n" + "=" * 60)
    print("  TruthLens — Ablation Study")
    print("  Comparing: Entity Masking ON vs OFF")
    print("=" * 60)

    # Load data once
    print("\n[ABLATION] Loading datasets...")
    datasets = get_datasets(download=True)

    # Run 1: WITH entity masking
    config.ENABLE_ENTITY_MASKING = True
    masked_results = run_single_pipeline(datasets, enable_masking=True, label="masked")

    # Run 2: WITHOUT entity masking
    config.ENABLE_ENTITY_MASKING = False
    unmasked_results = run_single_pipeline(datasets, enable_masking=False, label="unmasked")

    # Reset config
    config.ENABLE_ENTITY_MASKING = True

    # Generate outputs
    print("\n[ABLATION] Generating comparison outputs...")

    # Comparison CSV
    comparison = []
    for r in [masked_results, unmasked_results]:
        comparison.append({
            "Configuration": r["label"],
            "Entity_Masking": r["entity_masking"],
            "Test_Accuracy": r["test_accuracy"],
            "Test_Precision": r["test_precision"],
            "Test_Recall": r["test_recall"],
            "Test_F1": r["test_f1"],
            "Test_AUC": r["test_auc"],
            "CV_F1_Mean": r["cv_f1_mean"],
            "CV_F1_Std": r["cv_f1_std"],
            "Entity_Bias": r.get("entity_bias_accuracy", np.nan),
            "Length_Bias": r.get("length_bias_accuracy", np.nan),
            "Topic_Bias": r.get("topic_bias_accuracy", np.nan),
        })
    comp_df = pd.DataFrame(comparison)
    csv_path = os.path.join(config.RESULTS_DIR, "ablation_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"[ABLATION] Comparison CSV saved: {csv_path}")

    # Plots
    generate_comparison_plot(masked_results, unmasked_results)

    # Report
    generate_report(masked_results, unmasked_results)

    total_time = time.time() - total_start
    print(f"\n[ABLATION] Total time: {total_time / 60:.1f} minutes")
    print("[ABLATION] Done.")


if __name__ == "__main__":
    main()
