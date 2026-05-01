"""
TruthLens — Feature Engineering
=================================
Hybrid feature extraction that addresses TF-IDF limitations:

Pipeline A: TF-IDF → TruncatedSVD (150d)   — captures word importance
Pipeline B: Sentence-Transformer embeddings (384d) — captures SEMANTICS
            that TF-IDF misses; replaces the older averaged-GloVe pipeline
            because gensim is incompatible with Python 3.14
Pipeline C: Stylometric features (17d)     — captures writing style, detects
            AI text

Total feature vector: ~551 dimensions (dense, manageable, works with LIME/SHAP)
"""

import os
import re
import sys
import warnings
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Lazy-load for heavy libs
_sbert_model = None
_sbert_load_success = False
_vader_analyzer = None

# Stylometric feature names — single source of truth for ordering and count
STYLOMETRIC_NAMES = [
    "word_count", "char_count", "avg_word_length",
    "sentence_count", "avg_sentence_length",
    "vocabulary_richness", "capital_ratio",
    "exclamation_rate", "question_rate",
    "digit_ratio",
    "sentiment_compound", "sentiment_pos", "sentiment_neg",
    "flesch_reading_ease", "automated_readability_index",
    "burstiness", "zipf_coefficient",
]
NUM_STYLOMETRIC = len(STYLOMETRIC_NAMES)  # 17

# Common abbreviations — avoid splitting sentences on these
_ABBREV_PATTERN = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs|etc|U\.S|U\.K|U\.N|Jan|Feb|Mar|Apr'
    r'|Jun|Jul|Aug|Sep|Oct|Nov|Dec|St|Ave|Blvd|Corp|Dept|Gov|Sen|Rep)\.',
    re.IGNORECASE,
)


def _load_sbert():
    """Load sentence-transformer model lazily.

    Replaces the previous gensim-based GloVe loader. MiniLM is ~80MB and is
    cached locally after first download. We keep CPU-only inference because
    the corpus is small enough that GPU adds setup overhead with no payoff.
    """
    global _sbert_model, _sbert_load_success
    if _sbert_model is None and config.ENABLE_SBERT:
        print("[FEATURES] Loading sentence-transformer (first time may download ~80MB)...")
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME, device="cpu")
            _sbert_load_success = True
            print(f"[FEATURES] SBERT loaded: {config.SBERT_MODEL_NAME}, {config.SBERT_DIM}d")
        except Exception as e:
            warnings.warn(f"Failed to load SBERT: {e}. Skipping semantic embeddings.")
            _sbert_load_success = False
            return None
    return _sbert_model


def _load_vader():
    """Load VADER sentiment analyzer lazily."""
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


class TruthLensFeatureEngine:
    """
    Hybrid feature extraction pipeline.

    Combines TF-IDF (reduced via SVD), GloVe embeddings, and
    stylometric features into a single dense feature matrix.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
        )
        self.svd = TruncatedSVD(
            n_components=config.SVD_COMPONENTS,
            random_state=config.RANDOM_STATE,
        )
        self.scaler = StandardScaler()  # Normalize all features

        self.is_fitted = False
        self.feature_names = []
        # Number of stylometric features — may differ from NUM_STYLOMETRIC
        # when loading models trained with an older version of the code.
        self._num_stylometric = NUM_STYLOMETRIC

    def fit(self, processed_texts, cleaned_texts=None, raw_texts=None):
        """
        Fit all feature extractors on training data.

        Args:
            processed_texts: Lemmatized + masked texts (for TF-IDF)
            cleaned_texts:   Cleaned texts without masking (for GloVe)
            raw_texts:       Original raw texts (for stylometric features)
        """
        print("\n[FEATURES] Fitting feature extractors...")

        # Pipeline A: TF-IDF → SVD
        print("[FEATURES] Fitting TF-IDF + SVD...")
        tfidf_matrix = self.tfidf.fit_transform(processed_texts)
        self.svd.fit(tfidf_matrix)
        explained_var = self.svd.explained_variance_ratio_.sum()
        self.svd_explained_variance = explained_var
        self.actual_svd_dims = self.svd.components_.shape[0]
        print(f"[FEATURES] TF-IDF vocabulary: {len(self.tfidf.vocabulary_)} terms")
        print(f"[FEATURES] SVD {self.actual_svd_dims}d explains {explained_var:.1%} variance")

        # Pipeline B: SBERT (no fitting needed — pretrained)
        if config.ENABLE_SBERT:
            _load_sbert()

        # Track whether SBERT actually loaded (not just whether config enabled it)
        self.sbert_actually_loaded = _sbert_load_success and config.ENABLE_SBERT

        # Pipeline C: Stylometric (no fitting needed)

        # Now transform everything to get the full feature matrix for scaler fitting
        all_features = self._transform_all(processed_texts, cleaned_texts, raw_texts)

        # Fit the StandardScaler
        self.scaler.fit(all_features)

        # Build feature names
        self._build_feature_names()

        self.is_fitted = True
        self.trained_with_sbert = self.sbert_actually_loaded
        self.trained_with_stylometric = config.ENABLE_STYLOMETRIC
        print(f"[FEATURES] Total feature dimensions: {all_features.shape[1]}")

        return self

    def transform(self, processed_texts, cleaned_texts=None, raw_texts=None):
        """
        Transform texts to feature matrix using fitted extractors.

        Args:
            processed_texts: Lemmatized + masked texts (for TF-IDF)
            cleaned_texts:   Cleaned texts without masking (for GloVe)
            raw_texts:       Original raw texts (for stylometric features)

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Feature engine not fitted. Call fit() first.")

        all_features = self._transform_all(processed_texts, cleaned_texts, raw_texts)
        scaled = self.scaler.transform(all_features)

        return scaled

    def fit_transform(self, processed_texts, cleaned_texts=None, raw_texts=None):
        """Fit and transform in one step."""
        self.fit(processed_texts, cleaned_texts, raw_texts)
        return self.transform(processed_texts, cleaned_texts, raw_texts)

    def _transform_all(self, processed_texts, cleaned_texts, raw_texts):
        """Internal: extract all features and concatenate."""
        features_list = []

        # Pipeline A: TF-IDF → SVD
        tfidf_matrix = self.tfidf.transform(processed_texts)
        svd_features = self.svd.transform(tfidf_matrix)
        features_list.append(svd_features)

        # Pipeline B: SBERT embeddings
        # Use training-time flag if available to ensure dimension match
        use_sbert = getattr(self, "trained_with_sbert", None)
        if use_sbert is None:
            use_sbert = config.ENABLE_SBERT
        if use_sbert:
            texts_for_sbert = cleaned_texts if cleaned_texts is not None else processed_texts
            sbert_features = self._extract_sbert(texts_for_sbert)
            features_list.append(sbert_features)

        # Pipeline C: Stylometric features
        use_stylometric = getattr(self, "trained_with_stylometric", config.ENABLE_STYLOMETRIC)
        if use_stylometric:
            texts_for_style = raw_texts if raw_texts is not None else processed_texts
            style_features = self._extract_stylometric(texts_for_style)
            features_list.append(style_features)

        return np.hstack(features_list)

    def _extract_sbert(self, texts):
        """
        Extract document-level sentence-transformer embeddings.

        Unlike averaged-GloVe (which collapses word meaning into a single
        bag-of-vectors mean), MiniLM produces a contextual document vector
        in one forward pass, so phrases like "not effective" and "effective"
        actually differ in the encoded space.

        Empty / non-string inputs map to zero vectors so downstream feature
        concatenation still aligns shape-wise.
        """
        sbert = _load_sbert()
        if sbert is None:
            return np.zeros((len(texts), config.SBERT_DIM))

        # Replace empty / non-string inputs with a sentinel so the batch
        # encode call doesn't crash, then zero those rows back out.
        safe_texts, empty_mask = [], []
        for t in texts:
            if isinstance(t, str) and t.strip():
                safe_texts.append(t)
                empty_mask.append(False)
            else:
                safe_texts.append("")
                empty_mask.append(True)

        embeddings = sbert.encode(
            safe_texts,
            batch_size=64,
            show_progress_bar=len(safe_texts) > 500,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if any(empty_mask):
            embeddings = np.array(embeddings, copy=True)
            for i, is_empty in enumerate(empty_mask):
                if is_empty:
                    embeddings[i] = 0.0
        return embeddings

    def _extract_stylometric(self, raw_texts):
        """
        Extract stylometric (writing style) features from RAW text.

        These features help detect:
        - Sensationalism (excessive punctuation, caps)
        - AI-generated text (uniform sentence length, Zipf deviation)
        - Low-quality writing (readability scores)

        Features extracted (17 total):
        0.  word_count
        1.  char_count
        2.  avg_word_length
        3.  sentence_count
        4.  avg_sentence_length
        5.  vocabulary_richness (type-token ratio)
        6.  capital_ratio
        7.  exclamation_rate (per sentence, not raw count — avoids length bias)
        8.  question_rate (per sentence)
        9.  digit_ratio
        10. sentiment_compound (VADER)
        11. sentiment_pos
        12. sentiment_neg
        13. flesch_reading_ease
        14. automated_readability_index
        15. burstiness (sentence length variance / mean — AI text is unnaturally uniform)
        16. zipf_coefficient (log-log slope of word freq distribution)
        """
        import textstat

        vader = _load_vader()

        all_features = []
        for text in tqdm(raw_texts, desc="Stylometric features", leave=False):
            if not isinstance(text, str) or len(text.strip()) == 0:
                all_features.append(np.zeros(self._num_stylometric))
                continue

            try:
                words = text.split()
                word_count = len(words)
                char_count = len(text)
                avg_word_length = np.mean([len(w) for w in words]) if words else 0

                # Sentence detection — handle common abbreviations to avoid over-splitting
                abbrev_cleaned = _ABBREV_PATTERN.sub(r'\1', text)
                sentences = [s.strip() for s in re.split(r'[.!?]+', abbrev_cleaned) if s.strip()]
                sentence_count = max(len(sentences), 1)
                avg_sentence_length = word_count / sentence_count

                # Vocabulary richness (type-token ratio)
                unique_words = set(w.lower() for w in words)
                vocabulary_richness = len(unique_words) / max(word_count, 1)

                # Capitalization
                upper_chars = sum(1 for c in text if c.isupper())
                capital_ratio = upper_chars / max(char_count, 1)

                # Punctuation rates (per sentence — normalized to avoid length bias)
                exclamation_rate = text.count("!") / sentence_count
                question_rate = text.count("?") / sentence_count

                # Digit ratio
                digit_chars = sum(1 for c in text if c.isdigit())
                digit_ratio = digit_chars / max(char_count, 1)

                # Sentiment (VADER)
                sentiment = vader.polarity_scores(text[:5000])  # Limit for speed
                sentiment_compound = sentiment["compound"]
                sentiment_pos = sentiment["pos"]
                sentiment_neg = sentiment["neg"]

                # Readability — guard against NaN on very short texts
                if word_count >= 3:
                    flesch = textstat.flesch_reading_ease(text)
                    ari = textstat.automated_readability_index(text)
                    # Clamp to reasonable range to prevent outlier scaling issues
                    flesch = max(min(flesch, 120.0), -50.0)
                    ari = max(min(ari, 30.0), 0.0)
                else:
                    flesch = 0.0
                    ari = 0.0

                # Burstiness — std/mean of sentence lengths
                # AI-generated text tends to have unnaturally uniform sentence lengths
                if len(sentences) > 1:
                    sent_lengths = [len(s.split()) for s in sentences]
                    mean_sl = np.mean(sent_lengths)
                    burstiness = np.std(sent_lengths) / max(mean_sl, 1.0)
                else:
                    burstiness = 0.0

                # Zipf coefficient — log-log slope of word frequency distribution
                # Real human text follows Zipf's law (slope ~ -1); AI text deviates
                word_freqs = sorted(Counter(w.lower() for w in words).values(), reverse=True)
                if len(word_freqs) > 2:
                    ranks = np.arange(1, len(word_freqs) + 1, dtype=np.float64)
                    log_freqs = np.log(np.array(word_freqs, dtype=np.float64))
                    zipf_coefficient = np.polyfit(np.log(ranks), log_freqs, 1)[0]
                else:
                    zipf_coefficient = 0.0

                features = [
                    word_count, char_count, avg_word_length,
                    sentence_count, avg_sentence_length,
                    vocabulary_richness, capital_ratio,
                    exclamation_rate, question_rate,
                    digit_ratio,
                    sentiment_compound, sentiment_pos, sentiment_neg,
                    flesch, ari,
                    burstiness, zipf_coefficient,
                ]
                # Slice to match training-time count (handles old models with 15 features)
                features = features[:self._num_stylometric]
            except Exception:
                features = [0.0] * self._num_stylometric

            all_features.append(features)

        return np.array(all_features, dtype=np.float64)

    def _build_feature_names(self):
        """Build human-readable feature names for explainability."""
        names = []

        # SVD component names — use ACTUAL dims (may differ from config if n_samples < config)
        actual_dims = getattr(self, "actual_svd_dims", config.SVD_COMPONENTS)
        names += [f"tfidf_svd_{i}" for i in range(actual_dims)]

        # SBERT dimension names
        use_sbert = getattr(self, "trained_with_sbert", config.ENABLE_SBERT)
        if use_sbert:
            names += [f"sbert_{i}" for i in range(config.SBERT_DIM)]

        # Stylometric feature names
        use_stylometric = getattr(self, "trained_with_stylometric", config.ENABLE_STYLOMETRIC)
        if use_stylometric:
            names += STYLOMETRIC_NAMES[:self._num_stylometric]

        self.feature_names = names

    def get_feature_names(self):
        """Get list of feature names."""
        return self.feature_names

    def vocab_coverage(self, processed_texts):
        """
        For each processed_text, return the ratio of input tokens that exist
        in the fitted TF-IDF vocabulary.

        This powers the abstention gate: a TruthLens prediction is only as
        trustworthy as the model's familiarity with the input lexicon. When
        coverage is low, the verdict is driven almost entirely by stylometric
        features, which is unreliable for non-news text — so the backend
        scales confidence down (and emits "Uncertain" beyond a threshold).

        Returns one float in [0.0, 1.0] per input.
        """
        if not self.is_fitted:
            raise RuntimeError("Feature engine not fitted. Call fit() first.")
        if isinstance(processed_texts, str):
            processed_texts = [processed_texts]
        tfidf_matrix = self.tfidf.transform(processed_texts)
        coverages = []
        for i, text in enumerate(processed_texts):
            input_words = max(len(text.split()), 1) if isinstance(text, str) else 1
            matched = int(tfidf_matrix[i].nnz)  # nonzero unique vocab terms in this row
            coverages.append(min(1.0, matched / input_words))
        return coverages

    def get_tfidf_feature_names(self):
        """Get TF-IDF vocabulary (for LIME text explanations)."""
        return self.tfidf.get_feature_names_out()

    def save(self, path=None):
        """Save fitted feature engine to disk."""
        if path is None:
            path = os.path.join(config.MODELS_DIR, "feature_engine.pkl")
        sbert_flag = getattr(self, "trained_with_sbert", config.ENABLE_SBERT)
        with open(path, "wb") as f:
            pickle.dump({
                "tfidf": self.tfidf,
                "svd": self.svd,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
                "svd_explained_variance": getattr(self, "svd_explained_variance", None),
                "actual_svd_dims": getattr(self, "actual_svd_dims", config.SVD_COMPONENTS),
                "trained_with_sbert": sbert_flag,
                "trained_with_glove": sbert_flag,  # legacy alias for older code paths
                "embedding_dim": config.SBERT_DIM if sbert_flag else 0,
                "trained_with_stylometric": getattr(self, "trained_with_stylometric", config.ENABLE_STYLOMETRIC),
                "num_stylometric": self._num_stylometric,
            }, f)
        print(f"[FEATURES] Feature engine saved: {path}")

    def load(self, path=None):
        """Load fitted feature engine from disk."""
        if path is None:
            path = os.path.join(config.MODELS_DIR, "feature_engine.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.tfidf = data["tfidf"]
        self.svd = data["svd"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]
        self.svd_explained_variance = data.get("svd_explained_variance")
        self.actual_svd_dims = data.get("actual_svd_dims", config.SVD_COMPONENTS)
        # Restore training-time feature flags so inference matches training dimensions.
        expected_features = self.scaler.n_features_in_
        if "trained_with_sbert" in data:
            self.trained_with_sbert = data["trained_with_sbert"]
        elif "trained_with_glove" in data:
            # Legacy pickle from the GloVe era
            self.trained_with_sbert = data["trained_with_glove"]
        else:
            # Infer from scaler dimensions for very old saves
            self.trained_with_sbert = expected_features > (config.SVD_COMPONENTS + NUM_STYLOMETRIC + 5)
            print(f"[FEATURES] Inferred trained_with_sbert={self.trained_with_sbert} from {expected_features} features")
        # Embedding dim: prefer saved value; fall back to current SBERT_DIM
        self._embedding_dim = data.get("embedding_dim", config.SBERT_DIM if self.trained_with_sbert else 0)
        self.trained_with_stylometric = data.get("trained_with_stylometric", True)
        # Infer stylometric count from saved data, or from scaler dimensions
        if "num_stylometric" in data:
            self._num_stylometric = data["num_stylometric"]
        else:
            # Old models didn't save this — infer from scaler dimensions
            expected = self.scaler.n_features_in_
            svd_dims = self.actual_svd_dims
            embed_dims = self._embedding_dim if self.trained_with_sbert else 0
            self._num_stylometric = expected - svd_dims - embed_dims
            if self._num_stylometric < 0:
                self._num_stylometric = NUM_STYLOMETRIC
            print(f"[FEATURES] Inferred num_stylometric={self._num_stylometric} from saved model ({expected} total features)")
        print(f"[FEATURES] Feature engine loaded: {path}")
        return self


if __name__ == "__main__":
    # Quick test
    texts = [
        "the president announced new trade policies today",
        "breaking scientists discover amazing cure click now",
        "quarterly report shows steady economic growth in region",
    ]
    engine = TruthLensFeatureEngine()
    features = engine.fit_transform(texts, texts, texts)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature names ({len(engine.feature_names)}): {engine.feature_names[:5]}...{engine.feature_names[-5:]}")
