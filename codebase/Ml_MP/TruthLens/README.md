# TruthLens — Debiased, Explainable Fake News Detection

## 🎯 Project Goal

Build a fake news detection system that **avoids the 7 critical mistakes** found in existing models.

## ❌ Problems in Existing Models → ✅ Our Solutions

| # | Problem | Our Solution |
|---|---|---|
| 1 | **Dataset Bias** (models learn "Reuters = Real") | Entity masking (replace names with [PERSON], [ORG], [LOC]) |
| 2 | **Generalization Failure** (99% on ISOT → 45% on LIAR) | Cross-dataset evaluation |
| 3 | **TF-IDF can't capture semantics** | Hybrid: TF-IDF + GloVe embeddings |
| 4 | **Ignoring writing style context** | Stylometric features (sentiment, readability, vocabulary richness) |
| 5 | **AI-generated text undetectable** | Perplexity, burstiness, type-token ratio features |
| 6 | **Black box models** | LIME + SHAP explainability with bias auditing |
| 7 | **Majority voting is naive** | Stacking ensemble with LR meta-learner |

## 🏗️ Architecture

```
Raw Text
    │
    ├──→ Clean Text ──→ Entity Masking ──→ TF-IDF ──→ SVD (150d)
    │                                                      │
    ├──→ Clean Text ──→ GloVe Average Embeddings (100d) ──→│
    │                                                      │──→ [265d Feature Vector]
    └──→ Raw Text ──→ Stylometric Features (15d) ─────────→│
                                                           │
                                                    ┌──────┴──────┐
                                                    │  Stacking   │
                                                    │  Ensemble   │
                                                    │ SVM+LR+RF   │
                                                    │ ↓ Meta-LR   │
                                                    └──────┬──────┘
                                                           │
                                                    ┌──────┴──────┐
                                                    │  LIME/SHAP  │
                                                    │  Explain    │
                                                    └─────────────┘
```

## 📂 Project Structure

```
TruthLens/
├── data/raw/              # Downloaded datasets (ISOT, LIAR)
├── data/processed/        # Preprocessed CSVs
├── models/                # Saved models & feature engine
├── results/               # Reports, LIME HTMLs
│   └── plots/             # Confusion matrices, ROC curves, SHAP plots
├── src/
│   ├── data_loader.py     # Auto-download & load datasets
│   ├── preprocessor.py    # Text cleaning + entity masking
│   ├── feature_engineer.py # TF-IDF + GloVe + stylometric features
│   ├── model_trainer.py   # SVM, LR, RF + stacking ensemble
│   ├── evaluator.py       # In-domain + cross-dataset + bias probing
│   └── explainer.py       # LIME + SHAP + bias audit
├── main.py                # Full pipeline runner
├── config.py              # All hyperparameters
├── requirements.txt       # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Kaggle (for ISOT dataset download)
```bash
# Get your API key from https://www.kaggle.com/settings
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%/.kaggle/ (Windows)
```

### 3. Run the Pipeline
```bash
# Full pipeline (recommended first run)
python main.py

# Quick mode (faster, smaller hyperparameter search)
python main.py --quick

# Without entity masking (to compare biased vs debiased)
python main.py --no-entity-masking

# Without GloVe (saves download time)
python main.py --no-glove

# Train on combined ISOT + LIAR (improves cross-dataset generalization):
python main.py --combined-training --no-glove

# All flags:
# --quick              Smaller hyperparameter grids (faster training)
# --combined-training  Train on ISOT + LIAR combined
# --no-glove           Disable GloVe (required on Python 3.14+)
# --no-entity-masking  Disable entity masking (for ablation studies)
# --skip-cross-dataset Skip cross-dataset evaluation
```

## 📊 Datasets

| Dataset | Type | Size | Purpose |
|---|---|---|---|
| **ISOT** | Full articles | ~44K | Primary training & evaluation |
| **LIAR** | Short claims | ~12.8K | Cross-dataset generalization test |

## 📈 Outputs

After running the pipeline, you'll find:

- `results/test_results.csv` — Model comparison table
- `results/results_report.txt` — Full analysis report
- `results/plots/confusion_matrices.png` — Confusion matrices
- `results/plots/roc_curves.png` — ROC curves with AUC
- `results/plots/shap_summary.png` — SHAP feature importance
- `results/lime_explanation_*.html` — Interactive LIME explanations

## 👥 Team

| Name | Roll No |
|---|---|
| Riddhi Patil | 52 |
| Prashant Jha | 54 |
| Aditya Soni | 57 |
| Riwan Pereira | 65 |

**Guide:** Dr. Joanne Gomes | **Institute:** St. Francis Institute of Technology

## Changelog (Improvements)

### Bug Fixes
- Fixed `os.system()` command injection risk in `preprocessor.py` (replaced with `subprocess.run()`)
- Fixed SHAP exception handling in `explainer.py` (now logs full traceback)
- Fixed SVD explained variance not being stored in `feature_engineer.py`
- Fixed unreachable return statements in backend `verify_wikipedia()` and `verify_address()`
- Fixed Wikipedia entity extraction filtering out legitimate geographic entities
- Fixed cache key collision in backend (changed from first-100-chars to MD5 hash)
- Removed unused GloVe preload in backend (saved 128MB RAM)
- Fixed hard-coded confidence overrides ignoring ML model probabilities
- Restricted CORS from wildcard to localhost origins

### Generalization Improvements
- Added Reuters/AP/AFP byline stripping to prevent source identity leakage
- Added supplementary regex-based news source masking (catches sources spaCy NER misses)
- Reduced SVD dimensions from 300 to 150 (faster training, less overfitting)
- Increased cross-validation folds from 3 to 5 (more robust evaluation)
- Added `max_features="sqrt"` to Random Forest (prevents memorizing all features)

## Explainability Methodology

TruthLens uses two complementary layers of explanation for each prediction:

### Layer 1: Content Keywords (SVD Backprojection)
For the TF-IDF portion of the feature vector, we use SVD backprojection to find which words contributed most to the 150-dimensional latent space the model operates in:

```
importance = |tfidf_vector @ SVD.components_.T @ SVD.components_|
```

This is more informative than raw TF-IDF weights because it shows which words "survive" dimensionality reduction — i.e., which words align with the variance directions the model actually uses. Words that happen to have high TF-IDF scores but project into low-variance SVD dimensions are correctly down-weighted.

### Layer 2: Style Indicators (Stylometric Features)
Three stylometric signals are surfaced as `[style]`-prefixed indicators when they exceed thresholds:
- **`[style] excessive punctuation (!)`** — 3+ exclamation marks (sensationalism signal)
- **`[style] heavy capitalization`** — capital ratio >25% (emotional emphasis signal)
- **`[style] out-of-domain vocabulary`** — vocabulary coverage <30% (text is out-of-distribution)

### Full LIME Explanations (Training Pipeline Only)
The training pipeline (`src/explainer.py`, `explain_with_lime()`) generates full LIME text explanations by perturbing the input 1,000 times and observing prediction changes. These are saved as interactive HTML files to `results/lime_explanation_*.html`. Full LIME is not run at inference time due to the ~5-second latency cost per prediction.

## Known Limitations

### Cross-Dataset Generalization Gap
The model trains primarily on ISOT (full news articles, 200–2000 words) and achieves ~99.7% F1 on its test set. However, when evaluated on LIAR (short political claims, 10–100 words), F1 drops to ~49.6%. **Root cause:** the two datasets have fundamentally different text length distributions. TF-IDF vocabulary and SVD components learned from long articles do not transfer well to short claims. The `--combined-training` flag trains on ISOT+LIAR together to mitigate this, but a full solution would require a transformer-based model (BERT/DistilBERT) fine-tuned on both domains.

### Entity Masking Limitations
spaCy's `en_core_web_sm` model does not catch all named entities. The supplementary regex covers 51 known news sources as a fallback. However, some entities (especially obscure sources or newly emerged names) will be missed. The bias probe (source-only accuracy) was 89.3% before the regex expansion; improved masking reduces this but does not eliminate it completely.

### GloVe Incompatibility with Python 3.14+
The `gensim` library is currently incompatible with Python 3.14+. Run with `--no-glove` on Python 3.14. This reduces the feature vector from 265 to 165 dimensions (TF-IDF SVD 150d + stylometric 15d). Performance is slightly lower than the 265d configuration.

### Explainability Scope
The web app's explainability display uses **SVD backprojection** (`tfidf_vec @ svd.components_.T @ svd.components_`) to identify which words contributed most to the TF-IDF portion of the features (150/265 dimensions). GloVe semantic embeddings (100d) are not directly interpretable at word level. Stylometric indicators (excessive punctuation, heavy capitalization, out-of-domain vocabulary) are shown separately as `[style]`-prefixed entries. Full LIME perturbation-based explanations are available in the training pipeline (`src/explainer.py`) but are not run at inference time due to performance constraints.
