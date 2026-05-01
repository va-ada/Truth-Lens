# 3. Method

## 3.1 Overview

TruthLens is a 6-phase pipeline (data → preprocess → features → train →
evaluate → explain) augmented by three audit-time mechanisms (A1 vocab-
coverage gate, A2 bias-probe loop, A4 conflict explainer) and a 7-source
retrieval-augmented verification layer at inference. Figure 3.1 (see
`workflow_diagram.png`) illustrates the data flow.

## 3.2 Preprocessing

Input texts are cleaned (URLs / HTML / emails stripped, case preserved for
NER), then entity-masked with spaCy `en_core_web_sm`. PERSON, ORG, GPE/LOC,
DATE, NORP, EVENT, FAC, and WORK_OF_ART entities are replaced with class
tokens (`[PERSON]`, `[ORG]`, etc.). A supplementary regex masks well-known
news-source identifiers (Reuters, AP, BBC, CNN, …) into a generic `[SOURCE]`
token. An "aggressive" mode (used by A2) additionally masks Reuters/AP-style
boilerplate verbs ("said", "reported", "according to") and SHAP-discovered
bias-correlated tokens.

Lemmatization produces `processed_text` for the lexical pipeline; the
unlemmatized but cleaned `cleaned_text` feeds the embedding pipeline; the
raw `text` feeds the stylometric pipeline.

## 3.3 Hybrid Feature Engine (551 dims)

| Pipeline | Method | Dims |
|---|---|---|
| A — Lexical | TF-IDF (10K vocab, 1-2 gram, sublinear TF) → TruncatedSVD | 150 |
| B — Semantic | sentence-transformers `all-MiniLM-L6-v2` (mean-pooled) | 384 |
| C — Stylometric | sentiment (VADER), readability (Flesch, ARI), burstiness (sentence-length std/mean), Zipf coefficient (log-log slope of word frequency rank), capital ratio, exclamation rate per sentence, question rate per sentence, digit ratio, vocabulary richness, word/char/sentence count, avg word length, avg sentence length | 17 |

All 551 dims are concatenated and standard-scaled.

The original GloVe-based pipeline B was replaced with MiniLM after the
ablation in §4 showed averaged-GloVe contributed zero accuracy. MiniLM
produces a contextual document vector in one forward pass, distinguishing
e.g. "not effective" from "effective" — a distinction lost by averaging
word vectors.

## 3.4 Stacking Ensemble

Three base models trained with stratified 5-fold GridSearchCV optimizing F1:

- **SVM** — `LinearSVC` wrapped in `CalibratedClassifierCV` (cv=3) for
  `predict_proba` support; class_weight=balanced.
- **Logistic Regression** — L2 penalty, lbfgs, balanced.
- **Random Forest** — n_estimators ∈ {100, 300, 500}, max_depth ∈ {10, 20, ∞},
  max_features=sqrt, balanced.

The meta-learner is a Logistic Regression over `predict_proba` outputs of
the three base models, fit via 5-fold internal stacking. This dominates
naive majority voting because the meta-learner can learn that, e.g., RF is
more reliable on long texts and SVM is more reliable on short ones.

## 3.5 A1 — Vocabulary-Coverage Abstention Gate

For each input, after lemmatization and entity masking, we compute the
fraction of input tokens that appear in the fitted TF-IDF vocabulary:

```
coverage(x) = | tokens(x) ∩ TFIDF.vocab | / max(| tokens(x) |, 1)
```

If `coverage(x) < 0.30` or fewer than 4 tokens were matched, we scale the
ensemble confidence by `min(matched_terms / 4, coverage / 0.30)` and emit
"Uncertain" if the scaled confidence falls below 0.55.

The justification, made empirical in §4: when coverage is low, the
prediction is driven almost entirely by stylometric features (which are
dataset-shift-sensitive — Schuster et al. 2020). Refusing to predict is
strictly more honest than emitting confident garbage.

`scripts/selective_risk.py` produces a selective-risk plot comparing
coverage-based abstention against the standard MaxProb baseline on the LIAR
cross-domain set.

## 3.6 A2 — Bias-Probe-Driven Debiasing Loop

```
1. Probe(D_train, baseline)           → bias_acc_before
2. shap = explain_with_shap(RF, X)
3. bias_features = audit_bias(shap)   → top suspicious feature indices
4. D_train' = mask_aggressively(D_train)
5. fit feature engine on D_train'
6. train debiased ensemble with bias_features zeroed in scaler
7. Probe(D_train', debiased)          → bias_acc_after
8. write bias_audit.txt with delta
```

`bias_probe()` (`src/evaluator.py`) trains a logistic regression on
features derived only from source-correlated tokens (the `[ORG]` and
`[SOURCE]` mask counts and explicit publisher names extracted via regex).
If that model can predict the label well, the dataset has source leakage.

The aggressive masker (`src/preprocessor.py` with `level="aggressive"`)
extends the standard NER masking with regex matches for Reuters/AP-style
datelines (`<CITY> (Reuters) –`) and verb phrases (`said`, `reported`,
`according to`). Bias-correlated tokens identified by SHAP are zeroed in
the scaled feature matrix before training.

The loop is iterative because the first masking pass may not catch
everything; SHAP after retraining identifies the residual leakage, and a
second pass closes it further.

## 3.7 A4 — Structured ML-vs-RAG Conflict Explainer

The backend collects 7 verifier results (timeline, Wikipedia, web RAG, news
sources, fact-checkers, Google Fact Check Tools, geographic) and applies 6
override rules (Rule 1: any fact-checker says false → Fake; Rule 2: ≥2
sources flag conflict → Fake; Rule 3: 1 conflict → cap confidence; Rule 4:
≥2 verified sources → flip low-confidence Fake to Real; Rule 5: fact-checker
confirmed True → boost; Rule 6: ML errored → fall back to RAG consensus).

Whichever rule fires, the system records:
- `ml_verdict_pre_rag`, `ml_confidence_pre_rag` — the model's raw output
- `rag_verdict`, `rag_confidence` — aggregated multi-source consensus
- `disagreement` — boolean (ML and RAG produced different verdicts)
- `winning_signal` — "ml" / "rag" / "consensus"
- `triggered_rule` — human-readable name of the rule that fired
- `flagging_verifiers` — names of verifiers that conflicted
- `vocab_coverage` — for transparency on whether A1 was active
- `bias_gate_active` — whether A1 forced abstention

This `ConflictReport` is returned in the API response and rendered as a
side-by-side panel in the UI when `disagreement=True` or
`bias_gate_active=True`. The user always sees *why* the verdict came out
the way it did.

## 3.8 Configuration

All thresholds — `VOCAB_COVERAGE_THRESHOLD = 0.30`, `MIN_MATCHED_TERMS = 4`,
`SVD_COMPONENTS = 150`, `SBERT_DIM = 384`, etc. — live in `config.py`. Both
the offline analysis scripts (`ablation_features.py`, `selective_risk.py`)
and the live FastAPI backend read from the same module, so the abstention
threshold reported in a paper plot can never drift from the threshold the
production endpoint uses.
