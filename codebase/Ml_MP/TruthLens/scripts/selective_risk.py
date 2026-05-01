"""
TruthLens — Selective-Risk Plot for the Vocabulary-Coverage Abstention Gate (A1)
=================================================================================
Compares two abstention strategies on a cross-domain task (ISOT-trained
ensemble evaluated on LIAR):

  1. TruthLens A1 — abstain when vocab_coverage(x) < t
  2. MaxProb baseline — abstain when max(predict_proba) < t

For each threshold, we record:
  - % abstained
  - accuracy on the *non-abstained* subset

A coverage-based gate that strictly dominates MaxProb at any operating
point is the empirical justification for A1.

Output:
  results/plots/selective_risk.png
  results/selective_risk.csv
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRUTHLENS_DIR = os.path.dirname(_HERE)
if _TRUTHLENS_DIR not in sys.path:
    sys.path.insert(0, _TRUTHLENS_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import config
from src.feature_engineer import TruthLensFeatureEngine
from src.preprocessor import preprocess_dataframe
from src.data_loader import get_datasets


def _evaluate_with_abstention(scores, preds, y_true, thresholds, descending=True):
    """Generic selective-risk computation.

    Args:
        scores: per-sample confidence (e.g. coverage ratio or max-proba)
        preds: model predictions (0/1)
        y_true: ground truth (0/1)
        thresholds: thresholds to evaluate
        descending: when True, larger score = more confident (don't abstain).
                    When False, smaller score = more confident.

    Returns DataFrame with columns: threshold, abstained_pct, kept_n, accuracy_kept.
    """
    scores = np.asarray(scores)
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)

    rows = []
    n = len(y_true)
    for t in thresholds:
        if descending:
            mask = scores >= t
        else:
            mask = scores <= t
        kept = mask.sum()
        abstained = n - kept
        if kept > 0:
            acc = float((preds[mask] == y_true[mask]).mean())
        else:
            acc = float("nan")
        rows.append({
            "threshold": float(t),
            "abstained_pct": round(100.0 * abstained / n, 2),
            "kept_n": int(kept),
            "accuracy_kept": round(acc, 4) if not np.isnan(acc) else None,
        })
    return pd.DataFrame(rows)


def main():
    print("[A1] Loading datasets and trained ensemble...")
    datasets = get_datasets(download=False)
    isot = datasets.get("isot")
    liar = datasets.get("liar")
    if liar is None or isot is None:
        raise RuntimeError("Need both ISOT and LIAR datasets cached on disk.")

    # Use a smaller LIAR subset for speed (~5k stratified)
    if len(liar) > 5000:
        from sklearn.model_selection import train_test_split
        _, liar = train_test_split(
            liar, test_size=5000 / len(liar),
            stratify=liar["label"], random_state=config.RANDOM_STATE,
        )
        liar = liar.reset_index(drop=True)

    print(f"[A1] LIAR sample: {len(liar)} rows. Preprocessing...")
    liar_proc = preprocess_dataframe(liar[["text", "label"]].copy())

    print("[A1] Loading feature engine + ensemble...")
    engine = TruthLensFeatureEngine().load(
        os.path.join(config.MODELS_DIR, "feature_engine.pkl")
    )
    ensemble = joblib.load(os.path.join(config.MODELS_DIR, "ensemble.joblib"))

    print("[A1] Computing features + predictions on LIAR...")
    X = engine.transform(
        liar_proc["processed_text"].tolist(),
        liar_proc["cleaned_text"].tolist(),
        liar_proc["text"].tolist(),
    )
    y = liar_proc["label"].values
    proba = ensemble.predict_proba(X)
    preds = proba.argmax(axis=1)
    maxproba = proba.max(axis=1)

    print("[A1] Computing per-sample vocabulary coverage...")
    coverage = engine.vocab_coverage(liar_proc["processed_text"].tolist())
    coverage = np.asarray(coverage)

    thresholds = np.linspace(0.0, 0.95, 20)

    a1_df = _evaluate_with_abstention(coverage, preds, y, thresholds, descending=True)
    a1_df.insert(0, "strategy", "A1: vocab-coverage")

    mp_df = _evaluate_with_abstention(maxproba, preds, y, thresholds, descending=True)
    mp_df.insert(0, "strategy", "Baseline: MaxProb")

    out_csv = os.path.join(config.RESULTS_DIR, "selective_risk.csv")
    pd.concat([a1_df, mp_df], ignore_index=True).to_csv(out_csv, index=False)
    print(f"[A1] CSV written: {out_csv}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a1_df["abstained_pct"], a1_df["accuracy_kept"],
            marker="o", linewidth=2, label="A1: vocab-coverage")
    ax.plot(mp_df["abstained_pct"], mp_df["accuracy_kept"],
            marker="s", linewidth=2, linestyle="--", label="Baseline: MaxProb")
    ax.set_xlabel("Abstained (%)")
    ax.set_ylabel("Accuracy on non-abstained subset")
    ax.set_title("Selective-Risk: TruthLens A1 vs MaxProb (ISOT→LIAR)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = os.path.join(config.PLOTS_DIR, "selective_risk.png")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[A1] Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
