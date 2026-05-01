# 4. Experiments

## 4.1 Datasets

- **ISOT** (Ahmed et al., University of Victoria) — ~44K full news articles,
  labelled real (Reuters) or fake (scraped). After deduplication our pipeline
  retains 39,105 articles (17,908 fake / 21,197 real).
- **LIAR** (Wang, ACL 2017) — 12,791 short political claims with 6-class
  labels collapsed to binary (`pants-fire` / `false` / `barely-true` → fake;
  rest → real).

Both datasets are used in two configurations:
- *In-domain*: Train ISOT, 5-fold stratified CV.
- *Cross-domain*: Train ISOT, evaluate on LIAR (and the reverse).

## 4.2 Feature Pipeline Ablation

Table 4.1 reports SVM F1 (CalibratedClassifierCV, C=1, balanced, max_iter=2000)
on every subset of {A=TF-IDF+SVD, B=Sentence-Transformer, C=Stylometric},
trained on ISOT 80%/tested on ISOT 20% (stratified). Numbers come from
`results/feature_ablation_table.txt` produced by `python ablation_features.py`.

| Configuration | Dims | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| A — TF-IDF+SVD only | 150 | 0.9877 | 0.9899 | 0.9832 | 0.9866 | 0.9991 |
| B — SBERT only | 384 | 0.9581 | 0.9651 | 0.9425 | 0.9537 | 0.9930 |
| C — Stylometric only | 17 | 0.8450 | 0.8671 | 0.7813 | 0.8220 | 0.9186 |
| A+B — TF-IDF+SBERT | 534 | 0.9931 | 0.9963 | 0.9886 | 0.9924 | 0.9997 |
| A+C — TF-IDF+Stylo | 167 | 0.9931 | 0.9952 | 0.9897 | 0.9924 | 0.9996 |
| B+C — SBERT+Stylo | 401 | 0.9684 | 0.9736 | 0.9570 | 0.9652 | 0.9960 |
| **A+B+C — Full system** ✓ | 551 | **0.9950** | **0.9966** | **0.9925** | **0.9945** | **0.9998** |

The original GloVe ablation (100d averaged-GloVe pipeline B) showed B-only
at 54.2% and A+B equal to A-only — i.e. averaged-GloVe contributed *zero*
in-domain accuracy. With MiniLM replacing GloVe:

- **Pipeline B is no longer dead weight.** SBERT-only F1 is 0.9537 (vs
  averaged-GloVe at ~0.0). The contextual document vector preserves enough
  semantic structure to classify on its own, where averaged-GloVe could not.
- **A+B+C strictly dominates A+C and A+B.** F1 0.9945 vs 0.9924 — small but
  reproducible across the held-out fold. Each pipeline adds something now.
- **Stylometry is still the weakest individually** (F1 0.8220), but its
  contribution to the full system is non-zero — A+B+C beats A+B by 0.0021
  in F1, all attributable to the 17 hand-crafted features.

The MiniLM swap was motivated by this exact finding (the original ablation
showed averaged-GloVe contributing zero), not by Python 3.14 / gensim
incompatibility, which was only the trigger.

## 4.3 In-Domain Evaluation (Stratified 5-Fold CV)

| Metric | Value (mean ± std) |
|---|---|
| Accuracy | 0.9927 ± 0.0011 |
| Precision | 0.9925 ± 0.0020 |
| Recall | 0.9916 ± 0.0015 |
| F1 | 0.9920 ± 0.0012 |
| AUC-ROC | 0.9995 ± 0.0001 |

Numbers from `results/results_report.txt`.

Confusion matrices, ROC curves, and calibration curves for the ensemble and
the three base models are saved under `results/plots/{confusion_matrices,
roc_curves,calibration_curve}.png`.

## 4.4 Cross-Dataset Evaluation

Trained on ISOT, evaluated on a stratified 5,000-sample LIAR subset (and
the reverse direction).

| Train | Test | Model | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| ISOT | LIAR | SVM | 0.5498 | 0.1998 | 0.5041 |
| ISOT | LIAR | LR | 0.5500 | 0.2133 | 0.5034 |
| ISOT | LIAR | RF | 0.4932 | 0.5313 | 0.5292 |
| ISOT | LIAR | Ensemble | 0.5482 | 0.2345 | 0.5144 |
| LIAR | ISOT | SVM | 0.5700 | 0.1281 | 0.4510 |
| LIAR | ISOT | LR | 0.5658 | 0.1463 | 0.4680 |
| LIAR | ISOT | RF | 0.5734 | 0.2195 | 0.5960 |
| LIAR | ISOT | Ensemble | 0.5706 | 0.1269 | 0.5038 |

The collapse from 99% in-domain to ~55% cross-domain is the empirical
motivation for §4.6.

## 4.5 Bias Probe — Before / After Debiasing Loop (A2)

Before any debiasing intervention, an LR classifier trained on source-name
features alone (no content) achieves 91.45% accuracy on ISOT — confirming
the source leakage suspected since Ahmed et al. published the dataset.

| Probe | Accuracy (before) | Threshold | Status |
|---|---|---|---|
| Source-only / entity bias | **91.45%** | 0.60 | **BIAS DETECTED** |
| Length-only | 54.69% | 0.60 | OK |
| Topic-only | 54.21% | 0.60 | OK |

After running `python main.py --debias-loop --skip-cross-dataset`, the
pipeline writes `results/bias_audit.txt` with the post-debiasing numbers.
Target: source-only accuracy < 70% AND in-domain ensemble F1 ≥ 0.98.

The before/after delta is the **measurable contribution** of A2. It is
reproducible — anyone can run the same flag and compare numbers — and it is
the exact protocol that makes A2 hard for a competitor to copy without
running the experiment themselves.

## 4.6 Selective-Risk Plot for the Vocab-Coverage Gate (A1)

`scripts/selective_risk.py` evaluates the ISOT-trained ensemble on the LIAR
test set under two abstention strategies:

1. **TruthLens A1**: abstain when `coverage(x) < t` for thresholds `t ∈ {0, 0.1, …, 0.9}`.
2. **MaxProb baseline**: abstain when `max(predict_proba) < t` for the same thresholds.

For each strategy and threshold, plot (% abstained, accuracy on remaining).

The actual finding (run on 5,000-sample LIAR subset, May 2026): the two
strategies behave **differently in shape, not in headline accuracy**.

| Strategy | Abstention curve | Accuracy on kept |
|---|---|---|
| A1 (vocab-coverage) | Slow ramp (0.1 % → 30 % across thresholds 0–0.95) | Hovers around 55.5 % across the curve |
| MaxProb baseline | Sharp ramp (0 % until t=0.5, then 0 → 16 %) | 55.6 % → 56.6 % at high abstention |

LIAR's distribution shift from ISOT is dominated by topic + length + claim-vs-
article style — vocabulary overlap remains relatively high, so A1 doesn't
strongly correlate with model error on this particular pair. The honest
takeaway: **A1 is a calibration mechanism, not a free-accuracy mechanism**.
It refuses to emit confident verdicts on inputs the model's lexical pipeline
genuinely cannot read (the original ISOT vocabulary), and its biggest impact
is on out-of-domain text whose vocabulary actually diverges (technical
prose, non-English translation, OCR noise).

The plot lives at `results/plots/selective_risk.png`; raw numbers in
`results/selective_risk.csv`. We discuss the implications for §6 (Discussion).

## 4.7 Conflict Explainer Case Study (A4)

We pick three canonical inputs and run them through the live API:

1. **Real**: a verified Reuters article. Expected: ML says Real, all RAG
   verifiers say Verified, `disagreement=False`, `winning_signal="ml"`.
2. **Fake**: a known Snopes-debunked claim. Expected: ML may say either,
   fact-checker RAG says Conflict, `triggered_rule="Rule 1"`,
   `winning_signal="rag"`.
3. **OOV**: a technical paragraph from a physics textbook. Expected:
   `vocab_coverage < 0.30`, `bias_gate_active=True`, `winning_signal="consensus"`,
   prediction "Uncertain".

The three raw API responses are saved to
`Truth/backend/tests/fixtures/{real,fake,oov}.json` for regression testing.
Screenshots of the rendered ConflictReport panel are in
`report(research paper)/figures/conflict_explainer_demo.png`.
