"""Unit tests for the A2 bias-probe + debiasing loop helpers.

These tests construct a synthetic corpus where the source name perfectly
predicts the label. The expected behaviour:

1. `source_only_bias_probe` returns ~1.0 accuracy on the raw corpus
   (a Reuters detector trivially separates the classes).
2. After aggressive entity masking strips the source identifier,
   `source_only_bias_probe` falls toward chance.

The test does NOT load spaCy or the full TruthLens pipeline — it operates
on plain strings so it stays fast and deterministic.
"""

import os
import sys

# ── Path setup so `import config` resolves and we don't load spaCy / SBERT ──
TRUTHLENS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRUTHLENS_DIR)

import config  # noqa: E402
config.ENABLE_SBERT = False
config.ENABLE_GLOVE = False

import pytest  # noqa: E402

from src.evaluator import source_only_bias_probe  # noqa: E402


def _build_synthetic_corpus(n=60):
    """Build a corpus where the source name is the only predictive signal.

    Half the samples carry a "WASHINGTON (Reuters) -" dateline + "reuters"
    in the body and are labelled 0 (real). The other half carry no source
    identifier and are labelled 1 (fake). A classifier looking only at
    source-name features should hit 100% on the unmasked corpus.
    """
    real_template = (
        "WASHINGTON (Reuters) - The president signed the bill into law today, "
        "according to Reuters reporting from the White House."
    )
    fake_template = (
        "BREAKING: shocking truth they don't want you to know about the cure "
        "doctors are HIDING for years."
    )
    texts, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            texts.append(real_template + f" Article {i}.")
            labels.append(0)
        else:
            texts.append(fake_template + f" Item {i}.")
            labels.append(1)
    return texts, labels


def test_source_only_probe_detects_perfect_leakage():
    texts, labels = _build_synthetic_corpus(60)
    res = source_only_bias_probe(texts, labels)
    assert res["source_only_bias_detected"] is True
    assert res["source_only_bias_accuracy"] >= 0.85, (
        f"expected near-perfect probe accuracy on a leaky synthetic corpus; "
        f"got {res['source_only_bias_accuracy']:.3f}"
    )


def test_source_only_probe_drops_after_masking():
    texts, labels = _build_synthetic_corpus(60)

    # Manual aggressive-style masking: remove the dateline + the explicit
    # "reuters" mentions so source-name features collapse.
    import re
    masked_texts = []
    for t in texts:
        t2 = re.sub(r'^[A-Z][A-Z\s]+\([A-Za-z]+\)\s*-\s*', '', t)
        t2 = re.sub(r'\b(reuters|associated press|cnn|bbc)\b', '[SOURCE]',
                    t2, flags=re.IGNORECASE)
        masked_texts.append(t2)

    res_after = source_only_bias_probe(masked_texts, labels)
    # After masking, the only source-name token left is "[SOURCE]" which
    # appears in BOTH classes (we substituted in real) — which means it has
    # zero predictive power. Source-only probe should fall to ~chance (0.5).
    assert res_after["source_only_bias_accuracy"] <= 0.65, (
        f"expected probe accuracy ≤ 0.65 after masking; "
        f"got {res_after['source_only_bias_accuracy']:.3f}"
    )


def test_source_only_probe_no_sources_baseline():
    """Sanity check: corpus with no source identifiers at all → ~chance."""
    texts = ["fact A " + str(i) for i in range(30)] + ["claim B " + str(i) for i in range(30)]
    labels = [0] * 30 + [1] * 30
    res = source_only_bias_probe(texts, labels)
    # The probe correctly fails to learn anything when no source signal exists.
    # Accuracy floor is whatever LR can squeeze from the constant fallback
    # token "__no_source__" — it should not exceed the bias threshold.
    assert res["source_only_bias_accuracy"] <= 0.65
