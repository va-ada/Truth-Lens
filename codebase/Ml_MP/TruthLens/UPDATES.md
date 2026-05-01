# TruthLens — Updates Log

## Update 1: Bug Fixes, Enhancements & Unique Features (2026-04-11)

### Critical Bug Fixes

| ID | Bug | File | Fix |
|---|---|---|---|
| BUG-1 | **Entity masking ran on lowercased text** — spaCy NER relies on capitalization (`Donald Trump` vs `donald trump`). Running NER on lowercase text meant most PERSON/ORG/LOC entities were missed. Core debiasing technique was crippled. | `preprocessor.py` | `clean_text()` now accepts `lowercase=False`; `preprocess_dataframe()` runs NER on mixed-case text, lowercases only after masking. |
| BUG-2 | **QUICK_MODE always True** — `config.py` had `QUICK_MODE = True` as default. Param grids were ternary expressions evaluated at import time, so the `--quick` CLI flag was a no-op. Full hyperparameter search was dead code. | `config.py` | Default changed to `False`; param grids converted to functions (`get_svm_params()`, `get_lr_params()`, `get_rf_params()`) evaluated at call time. |
| BUG-3 | **Cross-validation ran on test data** — `evaluate_in_domain()` received `X_test, y_test` instead of training data. `cross_validate()` re-fits new models on test data folds, giving misleadingly optimistic results. | `main.py` | Now passes `X_train, y_train` to `evaluate_in_domain()`. |
| BUG-4 | **Feature name count mismatch** — `_build_feature_names()` used `config.SVD_COMPONENTS` (requested) instead of actual SVD output dimensions. If SVD clips, feature names don't match features, corrupting SHAP/LIME explanations. | `feature_engineer.py` | Uses `self.actual_svd_dims` (real SVD output shape) instead of config. |
| BUG-5 | **`--skip-cross-dataset` flag defined but never checked** — The CLI flag was parsed but never read in `main()`. Cross-dataset evaluation always ran regardless of the flag. | `main.py` | Flag is now checked and passed to `run_phase5_evaluation()`. |

### Medium Bug Fixes

| ID | Bug | File | Fix |
|---|---|---|---|
| BUG-6 | Bias probe used first-N sampling (ordering bias) | `evaluator.py` | Stratified random sampling |
| BUG-7 | `textstat` returns NaN for 1-2 word texts | `feature_engineer.py` | Guarded with `word_count >= 3` + clamping to [-50, 120] |
| BUG-8 | Sentence splitting on `[.!?]+` over-splits on abbreviations (U.S., Dr., Mr.) | `feature_engineer.py` | Strips common abbreviation dots before splitting |
| BUG-9 | `exclamation_count` / `question_mark_count` were raw counts (encodes text length) | `feature_engineer.py` | Normalized to per-sentence rates |
| BUG-10 | GloVe load failure set `trained_with_glove=True` from config, not actual load status | `feature_engineer.py` | Tracks `_glove_load_success` separately |
| BUG-11 | LIME HTML output had XSS — raw text injected without escaping | `explainer.py` | `html.escape()` applied |
| BUG-12 | `evaluate_cross_dataset()` had unused `trainer` and `preprocessor_fn` params | `evaluator.py` | Cleaned up signature |
| BUG-13 | `create_prediction_explainer` read `config.ENABLE_ENTITY_MASKING` at inference time (not training time) | `explainer.py` | Captures config at creation time via closure |
| BUG-14 | ROC curve silently swallowed `predict_proba()` failures | `evaluator.py` | Now logs warning with error message |

### Enhancements

| Enhancement | File | Description |
|---|---|---|
| Duplicate detection | `data_loader.py` | `drop_duplicates(subset=["text"])` after loading ISOT (news syndication creates exact copies) |
| Class distribution logging | `main.py` | Prints `% Fake` for train/test splits after splitting |
| Reproducibility metadata | `evaluator.py` | Logs scikit-learn/numpy/pandas/spacy versions + config state in results report |
| Stale cache cleared | `data/processed/` | Deleted old preprocessed CSVs (NER pipeline changed — must re-preprocess) |

### Unique Features Added

| Feature | File | Description |
|---|---|---|
| **Burstiness** (AI text detection) | `feature_engineer.py` | Sentence length variance / mean. AI-generated text has unnaturally uniform sentence lengths. |
| **Zipf coefficient** (AI text detection) | `feature_engineer.py` | Log-log slope of word frequency distribution. Real text follows Zipf's law (slope ~ -1); AI text deviates. |
| **Calibration curve** | `evaluator.py` | Reliability diagram + probability histogram. Shows whether predicted confidence matches actual accuracy. |
| **Error analysis** | `evaluator.py` | Categorizes misclassifications by text length, confidence level, and direction (FP vs FN). |
| **Topic bias probe** | `evaluator.py` | Tests if topic keywords alone can predict fake/real (catches topic-based shortcuts). |
| **Gradio demo app** | `app.py` | Interactive web UI: paste text, get prediction + confidence + style indicators + disclaimer. |

### Feature Dimensions
- Before: 15 stylometric features (265d total with GloVe, 165d without)
- After: 17 stylometric features (267d total with GloVe, 167d without)

### Files Modified
`config.py`, `src/preprocessor.py`, `src/feature_engineer.py`, `src/model_trainer.py`, `src/evaluator.py`, `src/explainer.py`, `src/data_loader.py`, `main.py`, `requirements.txt`, `tests/test_feature_engineer.py`

### Files Created
`app.py` (Gradio demo)

---

## Update 2: Fact-Verification Pipeline (2026-04-11)

### Problem
TruthLens could only classify text by writing style. It could NOT:
- Fact-check claims ("Is today April 11th?", "Did NASA find water on Mars?")
- Verify dates, numbers, or statistics
- Answer questions
- Access any external knowledge

### Solution: Hybrid Fact-Verification Pipeline

```
User Input
    |
    v
Claim Classifier (regex + heuristics)
    |
    +-- Article (>50 words) --> Existing ML Pipeline (TF-IDF + GloVe + Ensemble)
    |
    +-- Factual Claim / Question --> Fact Verification:
            |
            +-- Temporal Check (dateparser + system clock)
            +-- Wikipedia Lookup (wikipedia-api)
            +-- Web Search (DuckDuckGo, free, no API key)
            +-- Evidence Aggregation (weighted voting)
            |
            v
        Verdict + Confidence + Evidence Trail + Disclaimer
```

### New Files
- `src/claim_detector.py` — Classifies input as article/claim/question/temporal
- `src/fact_checker.py` — Temporal + Wikipedia + Web search + Evidence aggregation
- `app.py` — Updated Gradio demo with unified interface for both modes

### New Dependencies
- `ddgs` — DuckDuckGo web search (free, no API key)
- `wikipedia-api` — Wikipedia article lookup (free)
- `dateparser` — Robust date/time parsing

### Key Design Decisions
1. **No paid APIs** — everything is free and works without signup
2. **Confidence capped at 95%** — never claims 100% certainty
3. **Evidence-based** — shows sources, not just labels
4. **Graceful offline fallback** — returns "Cannot verify (offline)" instead of crashing
5. **AI-generated disclaimer** on all outputs
6. **Source reliability weighting** — .gov/.edu/major outlets weighted higher than unknown blogs
7. **MediaWiki search API** for Wikipedia discovery — finds relevant articles (e.g., Chandrayaan-3 for "India moon landing") instead of just exact page lookups
8. **Recall-biased similarity metric** — measures what fraction of claim words appear in evidence, works better than Jaccard for asymmetric comparisons (short claims vs long Wikipedia summaries)

### Test Results
- **47 unit tests passed**, 7 skipped (spaCy model not downloaded)
- 31 new tests for fact-verification pipeline (claim detector, temporal verification, date parsing, similarity, evidence aggregation)
- End-to-end test: temporal claims verified/falsified correctly, Wikipedia verifies clear facts (e.g., "capital of France is Paris"), opinions detected, ML models load and classify articles

### Files Modified
`app.py` (rewritten with unified interface), `requirements.txt` (added ddgs, wikipedia-api, dateparser)

### Files Created
- `src/claim_detector.py` — Rule-based input classifier (regex + heuristics)
- `src/fact_checker.py` — Multi-source verifier (temporal + Wikipedia search API + DuckDuckGo + evidence aggregation)
- `tests/test_fact_verification.py` — 31 unit tests for the fact-verification pipeline
