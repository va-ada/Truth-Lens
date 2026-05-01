# 5. Results

## 5.1 Headline numbers

- **In-domain ensemble F1:** 0.9935 ± 0.0008 (5-fold stratified CV on ISOT,
  May 2026 SBERT-trained model). AUC-ROC 0.9997.
- **In-domain test-set F1 (20% holdout):** 0.9947 (Ensemble), Acc 0.9951,
  AUC 0.9998.
- **Cross-domain F1 (ISOT → LIAR):** ~0.23 (Ensemble). Honest reporting of
  the dataset-shift problem the rest of the system was built to address.
- **Source-name probe (before A2 debias loop):** **99.33%** accuracy. The
  original entity-bias probe (which mixes source names with PERSON / GPE
  entities) scored 91.45%; the new dedicated source-name-only probe is
  even more conclusive — the SBERT-trained ensemble can be largely
  predicted by a Reuters detector. After `--debias-loop` (in progress):
  expected drop to < 70%, captured in `results/bias_audit.txt`.
- **Feature ablation (May 2026 with MiniLM):**
  | Config | Dims | F1 |
  |---|---|---|
  | A — TF-IDF+SVD | 150 | 0.9866 |
  | B — SBERT | 384 | 0.9537 |
  | C — Stylometric | 17 | 0.8220 |
  | A+B | 534 | 0.9924 |
  | A+C | 167 | 0.9924 |
  | B+C | 401 | 0.9652 |
  | A+B+C ✓ | 551 | **0.9945** |
  Each pipeline contributes; A+B+C strictly dominates pairwise combinations.

## 5.2 What the conflict explainer surfaces

Across the three canonical case-study inputs (§4.7):

| Input | ML verdict (pre-RAG) | RAG verdict | Triggered rule | Winning signal | bias_gate_active |
|---|---|---|---|---|---|
| Real Reuters article | Real (conf 0.94) | Real (cons 0.86) | None / Rule 5 boost | ml | false |
| Snopes-flagged claim | (varies) | Fake (cons 0.83) | Rule 1: Fact-checker flagged FALSE | rag | false |
| OOV physics paragraph | (scaled) | Unknown | None | consensus | **true** |

The OOV case is the cleanest demonstration of A1 + A4 working together:
because vocabulary coverage is below threshold, the ML pipeline never
generates a confident verdict in the first place; the conflict report then
attributes the "Uncertain" outcome to the bias gate rather than to RAG
disagreement. Users see *why* and have correct calibration for the answer.

## 5.3 What surprised us

- **GloVe contributed nothing → MiniLM contributes a lot.** Across the
  original ablation, A-only and A+B were tied at 0.9877 (averaged-GloVe was
  a passenger). After the MiniLM swap, B-only reaches 0.9537 standalone and
  A+B+C strictly dominates every pair. Replacing the dead pipeline B was
  the right call and the numbers confirm it.
- **The Random Forest is overfit.** Training F1 of 0.9998 vs CV F1 of
  0.9702 on RF alone — but the meta-learner mostly relegates RF to a
  tie-breaker between SVM and LR, which is presumably why the ensemble's
  CV F1 is closer to the SVM/LR numbers than to RF's.
- **Address verifier is mostly noise.** Out of the 6 original RAG verifiers,
  the geographic / Nominatim verifier produced "Conflict" results that, on
  inspection, were unreliable when other verifiers were already verified.
  The endpoint downgrades address conflicts to "Unknown" when at least one
  other verifier had a verified result — empirical, not theoretical.
