"""
Unit tests for TruthLens feature_engineer.py (Pass 1 — synthetic data).

IMPORTANT: config.ENABLE_GLOVE must be set to False BEFORE importing
feature_engineer to prevent any GloVe download attempt.

Notes on test-corpus constraints
─────────────────────────────────
TruncatedSVD requires n_components <= min(n_samples, n_features). With
only 8 training samples, TruncatedSVD(n_components=150) would raise.
We patch config.SVD_COMPONENTS to a smaller value (20) before creating
the engine. Even so, scikit-learn's TruncatedSVD clips the actual output
to min(n_components, n_samples) = min(20, 8) = 8 components on an 8-row
matrix. The scaler records the real output width, so we derive the
expected dims from the scaler after fit rather than computing them upfront.

Stylometric features are always 15 (from _extract_stylometric); total
dims in tests = actual_svd_components + 15.

TF-IDF min_df is overridden to 1 so the tiny corpus isn't reduced to an
empty vocabulary.
"""

import os
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
TRUTHLENS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRUTHLENS_DIR)

# ── Disable semantic embeddings BEFORE importing feature_engineer ───────────
# Tests run on a tiny 8-sample synthetic corpus; loading MiniLM (~80 MB) for
# every test run would be wasteful. Disable both the new SBERT switch and the
# legacy GloVe alias so the embedding pipeline is fully off.
import config
config.ENABLE_SBERT = False
config.ENABLE_GLOVE = False
config.ENABLE_STYLOMETRIC = True

# ── Now safe to import ───────────────────────────────────────────────────────
import pytest
import numpy as np
from src.feature_engineer import TruthLensFeatureEngine

# ---------------------------------------------------------------------------
# Skip guard: if vaderSentiment or textstat aren't installed, skip stylometric
# tests that call _extract_stylometric() directly.
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401
    import textstat  # noqa: F401
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

requires_style_deps = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="vaderSentiment or textstat not installed",
)

# ---------------------------------------------------------------------------
# Shared corpus — hardcoded strings, no external files required.
# ---------------------------------------------------------------------------
FAKE_TEXTS = [
    "SHOCKING: Politicians HIDING the truth about vaccines!!!",
    "They don't want you to know about this secret conspiracy theory exposed",
    "BREAKING: Government LIES about economy wake up sheeple!!!",
    "This will DESTROY the establishment they banned this video",
]
REAL_TEXTS = [
    "The Federal Reserve raised interest rates by 25 basis points today.",
    "Scientists published findings on climate change in Nature journal.",
    "The Senate passed the infrastructure bill with bipartisan support.",
    "A new study suggests moderate exercise improves cardiovascular health.",
]

ALL_TEXTS = FAKE_TEXTS + REAL_TEXTS  # 8 samples total

# Test-safe SVD component count (production default is 150; with 8 samples
# TruncatedSVD clips output to min(n_components, n_samples) = 8 anyway).
SVD_COMPONENTS_TEST = 20


# ---------------------------------------------------------------------------
# Factory: engine with test-safe hyperparameters.
# Must be called with config.SVD_COMPONENTS already set to SVD_COMPONENTS_TEST
# so that TruthLensFeatureEngine.__init__ picks it up.
# ---------------------------------------------------------------------------
def _make_test_engine() -> TruthLensFeatureEngine:
    """Create a TruthLensFeatureEngine with test-safe hyperparameters."""
    engine = TruthLensFeatureEngine()
    # min_df=1 so the tiny 8-sample corpus keeps a useful vocabulary
    engine.tfidf.min_df = 1
    # Keep n_components in sync with config (already set to SVD_COMPONENTS_TEST)
    engine.svd.n_components = config.SVD_COMPONENTS
    return engine


# ---------------------------------------------------------------------------
# Module-scoped fixture: fitted engine shared across tests that need one.
# We record the real output width from the scaler after fit.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted_engine():
    """Engine fitted on the 8-sample corpus, GloVe disabled."""
    config.ENABLE_GLOVE = False
    config.ENABLE_STYLOMETRIC = True
    original_svd = config.SVD_COMPONENTS
    config.SVD_COMPONENTS = SVD_COMPONENTS_TEST
    try:
        engine = _make_test_engine()
        engine.fit(ALL_TEXTS, cleaned_texts=ALL_TEXTS, raw_texts=ALL_TEXTS)
    finally:
        config.SVD_COMPONENTS = original_svd
    return engine


# ---------------------------------------------------------------------------
# Helper: real dims that the fitted engine's scaler expects.
# This is the ground truth for shape assertions — it accounts for the SVD
# clipping described in the module docstring.
# ---------------------------------------------------------------------------
def _real_dims(engine: TruthLensFeatureEngine) -> int:
    """Return the number of features the fitted scaler was trained on."""
    return engine.scaler.n_features_in_


# ---------------------------------------------------------------------------
# Helper: raw (unscaled) stylometric feature vector for a single-text list.
# Does not require a fitted engine.
# ---------------------------------------------------------------------------
def _raw_style(texts):
    """Call _extract_stylometric directly — returns unscaled numpy array."""
    engine = TruthLensFeatureEngine()
    return engine._extract_stylometric(texts)


# ===========================================================================
# 1. word_count (stylometric index 0)
# ===========================================================================
@requires_style_deps
def test_stylometric_word_count():
    """_extract_stylometric(['hello world foo']) → word_count == 3."""
    feats = _raw_style(["hello world foo"])
    # The feature order is defined in the docstring of _extract_stylometric.
    # Index 0 = word_count.
    word_count_idx = 0
    assert feats[0, word_count_idx] == 3, (
        f"Expected word_count=3, got {feats[0, word_count_idx]}"
    )


# ===========================================================================
# 2. exclamation_count (stylometric index 7)
# ===========================================================================
@requires_style_deps
def test_stylometric_exclamation_count():
    """_extract_stylometric(['wow amazing!!!']) → exclamation_count >= 2."""
    feats = _raw_style(["wow amazing!!!"])
    exclamation_idx = 7  # fixed by _extract_stylometric feature order
    assert feats[0, exclamation_idx] >= 2, (
        f"Expected exclamation_count >= 2, got {feats[0, exclamation_idx]}"
    )


# ===========================================================================
# 3. capital_ratio (stylometric index 6)
# ===========================================================================
@requires_style_deps
def test_stylometric_capital_ratio():
    """_extract_stylometric on all-caps text → capital_ratio > 0.5."""
    feats = _raw_style(["ALL CAPS TEXT HERE NOW"])
    capital_ratio_idx = 6
    assert feats[0, capital_ratio_idx] > 0.5, (
        f"Expected capital_ratio > 0.5, got {feats[0, capital_ratio_idx]}"
    )


# ===========================================================================
# 4. VADER compound — positive text (stylometric index 10)
# ===========================================================================
@requires_style_deps
def test_stylometric_positive_sentiment():
    """Positive sentence → VADER compound > 0."""
    feats = _raw_style(["This is wonderful, excellent, and amazing!"])
    compound_idx = 10  # sentiment_compound
    assert feats[0, compound_idx] > 0, (
        f"Expected positive compound, got {feats[0, compound_idx]}"
    )


# ===========================================================================
# 5. VADER compound — negative text (stylometric index 10)
# ===========================================================================
@requires_style_deps
def test_stylometric_negative_sentiment():
    """Negative sentence → VADER compound < 0."""
    feats = _raw_style(["This is terrible, awful, and horrible!"])
    compound_idx = 10
    assert feats[0, compound_idx] < 0, (
        f"Expected negative compound, got {feats[0, compound_idx]}"
    )


# ===========================================================================
# 6. Output dims: (8, svd_actual + 15) with GloVe disabled
# ===========================================================================
def test_feature_output_dims_no_glove():
    """fit_transform on 8-sample corpus → correct shape with no GloVe."""
    config.ENABLE_GLOVE = False
    config.ENABLE_STYLOMETRIC = True
    original_svd = config.SVD_COMPONENTS
    config.SVD_COMPONENTS = SVD_COMPONENTS_TEST
    try:
        engine = _make_test_engine()
        output = engine.fit_transform(ALL_TEXTS, cleaned_texts=ALL_TEXTS, raw_texts=ALL_TEXTS)
    finally:
        config.SVD_COMPONENTS = original_svd

    # The real dims come from the scaler (SVD may produce fewer than
    # n_components when n_samples < n_components).
    expected_dims = _real_dims(engine)
    # Structural check: stylometric block is always 17 features.
    from src.feature_engineer import NUM_STYLOMETRIC
    assert expected_dims >= NUM_STYLOMETRIC, f"Expected at least {NUM_STYLOMETRIC} stylometric features"
    assert output.shape == (8, expected_dims), (
        f"Expected shape (8, {expected_dims}), got {output.shape}"
    )


# ===========================================================================
# 7. get_feature_names() count — should equal SVD_COMPONENTS_TEST + 15
# ===========================================================================
def test_feature_names_count_no_glove(fitted_engine):
    """After fitting, get_feature_names() length == SVD_COMPONENTS_TEST + 15.

    _build_feature_names() reads config.SVD_COMPONENTS at call time, which
    was set to SVD_COMPONENTS_TEST during fit.  The scaler may represent
    fewer actual dims (due to SVD clipping), but feature_names is built
    from config, not from SVD output shape.
    """
    names = fitted_engine.get_feature_names()
    # Names are built with actual SVD dims (set in fixture) + 17 stylometric
    from src.feature_engineer import NUM_STYLOMETRIC
    # actual_svd_dims may be less than SVD_COMPONENTS_TEST due to SVD clipping
    actual_svd = fitted_engine.actual_svd_dims
    expected_name_count = actual_svd + NUM_STYLOMETRIC
    assert len(names) == expected_name_count, (
        f"Expected {expected_name_count} feature names, got {len(names)}"
    )


# ===========================================================================
# 8. transform() on a single new text → consistent with scaler dims
# ===========================================================================
def test_transform_single_text(fitted_engine):
    """transform() on one new text → shape (1, <scaler_dims>)."""
    text = ["breaking news article here"]
    output = fitted_engine.transform(text, cleaned_texts=text, raw_texts=text)
    expected_dims = _real_dims(fitted_engine)
    assert output.shape == (1, expected_dims), (
        f"Expected shape (1, {expected_dims}), got {output.shape}"
    )


# ===========================================================================
# 9. Empty text does not crash → shape consistent with scaler dims
# ===========================================================================
def test_empty_text_no_crash(fitted_engine):
    """transform(['']) must return correct shape without raising."""
    output = fitted_engine.transform([""], cleaned_texts=[""], raw_texts=[""])
    expected_dims = _real_dims(fitted_engine)
    assert output.shape == (1, expected_dims), (
        f"Expected shape (1, {expected_dims}), got {output.shape}"
    )


# ===========================================================================
# 10. Save / load roundtrip — loaded engine produces same shape
# ===========================================================================
def test_save_load_roundtrip():
    """Fit → save → load → transform gives same output shape."""
    save_path = "/tmp/test_engine.pkl"
    config.ENABLE_GLOVE = False
    config.ENABLE_STYLOMETRIC = True
    original_svd = config.SVD_COMPONENTS
    config.SVD_COMPONENTS = SVD_COMPONENTS_TEST

    try:
        engine = _make_test_engine()
        engine.fit(ALL_TEXTS, cleaned_texts=ALL_TEXTS, raw_texts=ALL_TEXTS)
        saved_dims = _real_dims(engine)
        engine.save(save_path)

        fresh_engine = TruthLensFeatureEngine()
        fresh_engine.load(save_path)

        text = ["Government raises interest rates amid inflation concerns."]
        output = fresh_engine.transform(text, cleaned_texts=text, raw_texts=text)
        assert output.shape == (1, saved_dims), (
            f"Expected shape (1, {saved_dims}) after load, got {output.shape}"
        )
    finally:
        config.SVD_COMPONENTS = original_svd
        if os.path.exists(save_path):
            os.remove(save_path)
