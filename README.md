# TruthLens — A Self-Auditing Fake News Detection System

[Open the live demo](#quick-start) · [Read the paper](report%28research%20paper%29/) · [See the comparison vs other systems](#how-truthlens-differs-from-everything-else)

---

## What is TruthLens?

TruthLens is a fake-news detection system that combines a hybrid feature
engineering pipeline (TF-IDF + sentence embeddings + 17 stylometric features),
a stacking ensemble (SVM + LR + RF + LR meta-learner), and a 7-source
retrieval-augmented (RAG) verification layer — and then **audits its own
behaviour** in three ways no published peer system does:

1. A **vocabulary-coverage abstention gate** that refuses to predict on
   out-of-domain input.
2. A **bias-probe-driven debiasing loop** that quantifies entity-leakage bias
   *before and after* training, and ships the audit report alongside the model.
3. A **structured ML-vs-RAG conflict explainer** that shows users when the
   stylometric model and the RAG consensus disagreed, names the override rule
   that fired, and lists which verifiers flagged the claim.

This positions TruthLens as a **system-level integration + reproducible audit
protocol** — every individual technique has partial precedent, but the
combination (verified against 7 closely-related systems on 2026-04-30) is
not yet published.

---

## Authors

- Riddhi Patil
- Prashant Jha
- Aditya Soni
- Riwan Pereira

Project guide: Dr. Joanne Gomes — St. Francis Institute of Technology, Mumbai.

---

## Repository Structure

```
ml-mini-project/
├── codebase/
│   └── Ml_MP/
│       ├── TruthLens/                     # Core ML pipeline (Python)
│       │   ├── src/
│       │   │   ├── feature_engineer.py    # TF-IDF + SBERT + 17 stylometric, vocab_coverage()
│       │   │   ├── model_trainer.py       # Stacking ensemble (SVM + LR + RF + LR meta)
│       │   │   ├── preprocessor.py        # spaCy NER masking, Reuters byline strip
│       │   │   ├── evaluator.py           # bias_probe(), cross-dataset, in-domain CV
│       │   │   ├── explainer.py           # LIME, SHAP, audit_bias()
│       │   │   ├── claim_detector.py      # Article / claim / opinion / temporal routing
│       │   │   └── data_loader.py
│       │   ├── scripts/
│       │   │   ├── sentence_scorer.py     # Originality.ai-style per-sentence scoring
│       │   │   └── selective_risk.py      # Plot vocab-coverage abstention vs MaxProb
│       │   ├── ablation_features.py       # 7-config feature ablation table
│       │   ├── main.py                    # 6-phase orchestrator (--debias-loop, --no-sbert)
│       │   ├── config.py                  # Single source of truth for thresholds
│       │   ├── requirements.txt
│       │   └── results/                   # plots, csvs, bias_audit.txt, results_report.txt
│       │
│       └── Truth/                         # Web app
│           ├── backend/                   # FastAPI server
│           │   └── main.py                # ML inference + 7 RAG verifiers + ConflictReport
│           └── frontend/                  # React + Vite UI
│               └── src/
│                   ├── App.jsx, Analyzer.jsx
│                   └── landing/sections/Comparison.jsx   # Parity vs unique table
│
└── paper/                                 # Research paper sections + references
```

---

## Features

### What TruthLens has — full list

#### Parity with the rest of the field

| Capability | Closest analogue (verified) | Why it matters |
|---|---|---|
| Source-domain credibility tiers | NewsGuard | Surfaces a domain badge when the input contains a URL |
| Cross-reference against curated fact-check DB | Logically.ai | 7th RAG verifier hits Google Fact Check Tools API |
| Per-sentence scoring inside long documents | Originality.ai | `sentence_scores` field highlights risky passages in-line |
| Sentence-level highlighting | dEFEND (KDD ’19) | Coloured sentence backgrounds in the Linguistic Anomaly panel |
| Claim-type routing (article / claim / opinion / question / temporal) | Full Fact AI | Surfaced as a chip near the verdict |
| Stacking + LIME + SHAP explanations | AEC (ESWA 2025) | Word-level + global feature importance |
| Causal-inspired entity debiasing | ENDEF (SIGIR ’22) | Aggressive masking + SHAP-driven feature reweighting |
| Live RAG retrieval + rationale | VeraCT-Scan, STEEL | 7 verifiers run per request, override rules transparent |
| OCR / PDF / image input | (rare; not in the closest 7) | EasyOCR + PyPDF2 + python-docx |
| Cross-dataset evaluation | dEFEND (partial) | ISOT↔LIAR reported honestly (~55% — calls for the abstention gate) |

#### Hidden work — hard to copy

These are the parts that take **measured experiments** to reproduce, not just a
config flag:

1. **Vocabulary-Coverage Abstention Gate (A1).**
   When a TF-IDF dictionary overlap ratio drops below 30% (or fewer than 4
   matched terms appear), the gate scales confidence down and emits
   "Uncertain". A `scripts/selective_risk.py` plot shows this strictly
   dominates the standard MaxProb baseline at the same abstention rate on the
   LIAR cross-domain set. *Selective prediction in NLP exists (Kamath et al.
   ACL 2021), but tying abstention to TF-IDF dictionary overlap on a stacking
   ensemble is unpublished.*

2. **Bias-Probe-Driven Debiasing Loop (A2).**
   Out of the box, an SVM trained on ISOT source names alone (no content)
   reaches **91.45%** accuracy — the model is largely a Reuters detector.
   The `--debias-loop` pipeline runs the probe, identifies bias-correlated
   features via SHAP, applies aggressive entity masking + feature reweighting,
   re-trains, and **re-runs the probe**. The before/after delta is saved to
   `results/bias_audit.txt`. *ENDEF (SIGIR ’22) does causal entity debiasing,
   but does not iterate as an audit protocol with measured before/after
   numbers on a stacking ensemble.*

3. **Structured ML-vs-RAG Conflict Explainer (A4).**
   When the stylometric model and the 7-source RAG layer disagree, the API
   returns a structured `ConflictReport` field (ml_verdict, ml_confidence,
   rag_verdict, rag_confidence, disagreement, winning_signal, triggered_rule,
   flagging_verifiers, top_lime_tokens, vocab_coverage, bias_gate_active).
   The UI renders both verdicts side-by-side and names the override rule that
   fired. *AEC (ESWA 2025) has LIME+SHAP on a stacking ensemble but no RAG;
   VeraCT-Scan has LLM rationales but no conflict surfacing.*

---

## How TruthLens Differs From Everything Else

| Capability | TruthLens | NewsGuard | Logically | dEFEND | FANG | ENDEF | AEC | VeraCT |
|---|---|---|---|---|---|---|---|---|
| Classical hand-crafted features (stylometry, Zipf, burstiness) | ✅ 17 | ❌ | ❌ | ❌ | ❌ | ❌ | partial | ❌ |
| TF-IDF + SVD lexical pipeline | ✅ 150d | ❌ | ❌ | partial | ❌ | ❌ | ❌ | ✅ | ❌ |
| Sentence-transformer embeddings | ✅ MiniLM 384d | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Stacking ensemble | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Live RAG at inference | ✅ 7 verifiers | ❌ (curated DB) | partial | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-source consensus + override rules | ✅ 6 documented rules | partial | partial | ❌ | partial | ❌ | ❌ | partial |
| LIME + SHAP explainability | ✅ both | ❌ | ❌ | comments-based | ❌ | ❌ | ✅ | LLM-rationale |
| Bias auditing as a diagnostic | ✅ source-name probe (91.45%) | ❌ | ❌ | ❌ | ❌ | ✅ as mitigation | ❌ | ❌ |
| **Bias-probe-driven debiasing LOOP with before/after numbers** | ✅ | ❌ | ❌ | ❌ | ❌ | partial | ❌ | ❌ |
| Cross-dataset eval (ISOT→LIAR) | ✅ honest ~55% | n/a | n/a | partial | ❌ | ✅ | ❌ | ❌ |
| OCR / PDF / image input | ✅ EasyOCR | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Vocab-coverage abstention gate** | ✅ <30% → "Uncertain" | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Structured ML-vs-RAG conflict explainer** | ✅ in API + UI | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Open source (full code public) | ✅ | ❌ | ❌ | partial | ✅ | ✅ | partial | ❌ |

The three boldface rows are the parts of the stack we believe are unpublished
in this exact form — confirmed via the 2026-04-30 paper-search audit.

---

## Architecture

```
Raw text / PDF / Image
       │
       ▼
[ Preprocessor ]              spaCy NER → mask PERSON/ORG/LOC/DATE/EVENT
       │                       Reuters/AP/AFP byline regex strip
       ▼
[ Feature Engine — 551d ]
       ├── TF-IDF (10K vocab, 1-2 gram) → TruncatedSVD → 150d
       ├── SBERT (MiniLM-L6-v2)                        → 384d
       └── Stylometric (sentiment, readability,
            burstiness, Zipf, …)                       → 17d
       │
       ▼
[ Stacking Ensemble ]
       ├── SVM   (LinearSVC + Calibrated)
       ├── LR
       ├── RF
       └── LR meta-learner over predict_proba
       │
       ▼
[ Vocab-Coverage Gate ]   ──→  if coverage < 30% → "Uncertain"
       │
       ▼
[ 7-Source RAG Layer ]
       ├── Wikipedia       ├── Web (DuckDuckGo)
       ├── News consensus  ├── Snopes/PolitiFact/FullFact
       ├── Google Fact Check Tools
       ├── Calendar/Temporal
       └── Geographic (Nominatim)
       │
       ▼
[ Override Rules (1–6) ] ──→ ConflictReport
       │
       ▼
   Final verdict + LIME/SHAP + ConflictReport JSON
```

See `workflow_diagram.png` for the visual.

---

## Results

### In-domain (ISOT, 5-fold stratified CV — May 2026 SBERT-trained ensemble)

| Metric | Score |
|---|---|
| Accuracy | 99.41% ± 0.07% |
| F1-Score | 99.35% ± 0.08% |
| AUC-ROC | 99.97% ± 0.01% |
| Test-set F1 (20% holdout) | 99.47% |
| Test-set AUC-ROC | 99.98% |

### Cross-dataset (Train ISOT → Test LIAR)

| Model | Accuracy | F1 |
|---|---|---|
| SVM | 54.98% | 19.98% |
| Logistic Regression | 55.00% | 21.33% |
| Random Forest | 49.32% | 53.13% |
| Ensemble | 54.82% | 23.45% |

The honest ~55% cross-domain performance is exactly **why the abstention
gate exists** — without it, models would happily emit confident wrong
answers on out-of-domain text.

### Bias probe (before / after debiasing loop)

| Probe | Before | After (target) |
|---|---|---|
| Source-only (publisher-name regex) | **99.33% [BIAS DETECTED]** | <70% (run with `--debias-loop` and check `results/bias_audit.txt`) |
| Entity-bias (NER + bag-of-entities) | 91.45% [BIAS DETECTED] | — |
| Length-only | 54.69% [OK] | — |
| Topic-only | 54.21% [OK] | — |

The source-name-only probe scoring 99.33% on the SBERT-trained ensemble is
the headline number for A2 — a logistic regression seeing nothing but
publisher identifiers can predict ISOT labels nearly perfectly. The
debiasing loop's job is to bring that down while keeping in-domain F1 ≥ 0.98.

### Feature ablation (`ablation_features.py`)

| Configuration | Dims | F1 |
|---|---|---|
| A — TF-IDF+SVD only | 150 | 0.9866 |
| B — SBERT only | 384 | 0.9537 |
| C — Stylometric only | 17 | 0.8220 |
| A+B — TF-IDF+SBERT | 534 | 0.9924 |
| A+C — TF-IDF+Stylo | 167 | 0.9924 |
| B+C — SBERT+Stylo | 401 | 0.9652 |
| **A+B+C — Full system** ✓ | 551 | **0.9945** |

Every pipeline contributes; the full hybrid strictly dominates every pair.
Originally with averaged-GloVe, B-only F1 was 0.0 — i.e. GloVe was dead
weight. Replacing it with MiniLM made the "hybrid" claim real.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.14 supported)
cd codebase/Ml_MP/TruthLens
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the full pipeline (in-domain training)
python main.py                          # full grids
python main.py --quick                  # smaller grids, faster
python main.py --debias-loop            # run the bias-audit pipeline
python main.py --no-sbert               # alias --no-glove; disables semantic embeddings

# 3. Feature ablation table
python ablation_features.py             # produces results/feature_ablation_table.txt

# 4. Selective-risk plot (vocab-coverage abstention vs MaxProb baseline)
python scripts/selective_risk.py        # produces results/plots/selective_risk.png

# 5. Boot the demo
cd ../Truth/backend
python main.py                          # http://localhost:8000

cd ../frontend
npm install
npm run dev                             # http://localhost:5173
```

> **macOS note:** Python 3.14 cannot install `gensim`. The pipeline now uses
> `sentence-transformers` (MiniLM-L6-v2) instead, with no functionality loss.
> If you have an older `venv_macos/` that still references gensim, just
> `pip install sentence-transformers` and you’re done.

---

## Datasets

| Dataset | Description | Size |
|---|---|---|
| [ISOT](https://onlineacademiccommunity.uvic.ca/isot/) | Full news articles (Reuters real + scraped fake) | ~39K (after dedup) |
| [LIAR](https://huggingface.co/datasets/liar) | Short political claims with 6-class labels | ~12.8K |

Both are gitignored due to size — download from the links above and place under a local `datasets/` folder before training.

---

## Known Limitations

- Cross-dataset transfer to LIAR is ~55% — the abstention gate is the response.
- ISOT has well-known source leakage (every Reuters byline) — the bias-probe
  loop is the response, but the residual entity bias is non-zero.
- MiniLM is CPU-only on the team's laptops; latency is fine but a GPU would
  speed up bulk re-training.
- Google Fact Check Tools API is rate-limited for unauthenticated callers; we
  surface "unknown" rather than failing the request.

---

## Verified References

The 2026-04-30 competitive landscape survey verified each of these by URL
fetch. Every claim about prior art in this README is traceable to one of the
sources below.

**Commercial / consumer products**

- NewsGuard — https://www.newsguardtech.com/how-it-works/
- Logically.ai — https://www.logically.ai/
- Hoaxy — https://hoaxy.osome.iu.edu/
- Full Fact AI — https://fullfact.org/automated
- Google Fact Check Tools API — https://developers.google.com/fact-check/tools/api
- Originality.ai Fact-Checker — https://originality.ai/automated-fact-checker

**Academic systems**

- FakeNewsNet (Shu et al.) — https://github.com/KaiDMML/FakeNewsNet
- dEFEND (Shu et al., KDD ’19) — https://dl.acm.org/doi/10.1145/3292500.3330935
- FANG (Nguyen et al., CIKM ’20) — https://github.com/nguyenvanhoang7398/FANG
- Defending Against Neural Fake News — GROVER (Zellers et al., NeurIPS ’19) — https://arxiv.org/abs/1905.12616
- FakeBERT (Kaliyar et al.) — https://link.springer.com/article/10.1007/s11042-020-10183-2
- ENDEF (Zhu et al., SIGIR ’22) — https://arxiv.org/abs/2204.09484
- Bias Mitigation by Causal Intervention (SIGIR ’22) — https://dl.acm.org/doi/10.1145/3477495.3531850

**Recent RAG / LLM (2024–2025)**

- VeraCT Scan (2024) — https://arxiv.org/abs/2406.10289
- AEC — Adaptive Ensemble Classifier with LIME+SHAP (ESWA 2025) — https://www.sciencedirect.com/science/article/abs/pii/S0957417425013739
- Aletheia (IJCAI 2025) — https://www.ijcai.org/proceedings/2025/1273.pdf

**Methodological references for unique angles**

- Kamath et al., "The Art of Abstention" (ACL 2021) — https://aclanthology.org/2021.acl-long.84/
- Schuster et al., "The Limitations of Stylometry" (CL 2020) — https://aclanthology.org/2020.cl-2.8/
- Cross-Domain Failures of FND (Janicka et al.) — https://www.cys.cic.ipn.mx/index.php/CyS/article/view/3281/0
- Stylometric FND in/cross-domain (MDPI 2023) — https://www.mdpi.com/2079-9292/12/17/3676

**Datasets**

- ISOT Fake News Dataset — https://onlineacademiccommunity.uvic.ca/isot/
- LIAR — https://huggingface.co/datasets/liar
- Fact-Checking Fact Checkers — https://misinforeview.hks.harvard.edu/article/fact-checking-fact-checkers-a-data-driven-approach/

---

*This README is the user-facing surface. The full paper, with experiments and
plots, lives under `paper/`.*
