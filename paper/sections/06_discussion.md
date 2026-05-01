# 6. Discussion

## 6.1 What "novelty" means here

We do not claim a new model architecture. AEC (ESWA 2025) shipped a
stacking ensemble with LIME+SHAP on fake-news detection a year before us;
ENDEF shipped causal entity debiasing in 2022; VeraCT-Scan shipped
LLM-rationale RAG in 2024. What we claim novel is the **system-level
integration + reproducible audit protocol**: combining a vocab-coverage
abstention gate, a probe-driven debiasing *loop* (not a one-shot
mitigation), and a structured ML-vs-RAG conflict explainer in the same
pipeline.

This framing is appropriate for a project at the C403 mini-project scale.
It is honest about what is and isn't new. It also produces *measurable*
contributions — a bias-probe delta, a selective-risk plot, a conflict-rule
attribution table — that a peer reviewer can verify by running our
`--debias-loop` flag and our `selective_risk.py` script.

## 6.1.1 What the selective-risk plot actually showed

We hypothesised that a vocabulary-coverage gate would strictly dominate the
MaxProb baseline at high abstention rates on LIAR. The empirical result
(May 2026, 5K LIAR subset) was more nuanced: A1 abstains very gradually
because vocab overlap between ISOT and LIAR remains relatively high (LIAR
political claims share most words with ISOT news), so the texts most
filtered by low coverage are not the texts the model gets wrong most often.
MaxProb, in contrast, is the standard calibration signal and rises with
confidence threshold in the predictable way.

Re-framing in light of this finding: A1 is most valuable when the input is
**lexically** out of domain (technical jargon, OCR noise, transliteration,
non-English) — exactly the cases where MaxProb would output a confident
softmax distribution because the stylometric features alone happen to fall
into a familiar region. On the LIAR cross-domain pair specifically, A1's
contribution is mostly in *not lying* about texts the model can't read,
rather than in *higher* accuracy on those it does. We treat that as the
correct, honest contribution.

## 6.2 Limitations

- **In-domain accuracy is partly bias-driven.** The 91.45% source-name
  probe accuracy means a meaningful fraction of the 99.27% in-domain F1 is
  attributable to source leakage. After A2's debiasing loop, in-domain F1
  may drop a few tenths of a percent — that's the cost of honesty.
- **Cross-domain remains poor.** A1's abstention gate doesn't fix the
  underlying dataset-shift problem; it just refuses to lie about it. A
  transformer fine-tuned on combined ISOT+LIAR (or a model trained on
  larger, more diverse fake-news corpora) would likely close the gap.
- **The debiasing loop is greedy.** SHAP picks the top-k bias features and
  zeros them; we do not formally guarantee the resulting model is the
  optimal trade-off between in-domain accuracy and probe accuracy. A
  Pareto-frontier analysis would strengthen the contribution.
- **Google Fact Check API is rate-limited.** Without a key, the 7th
  verifier may return "unknown" frequently. The system handles this
  gracefully but the parity claim is weakened in practice.
- **No multilingual support.** ISOT and LIAR are English-only. Logically.ai
  supports 57 languages; we support 1.

## 6.3 Threats to validity

- **Probe-acc reduction may overshoot.** Aggressive masking could remove
  signal beyond just source leakage (e.g. legitimate proper noun mentions
  in real reporting). The before/after F1 column in `bias_audit.txt`
  guards against this — if F1 collapses, the loop has gone too far and
  we report the trade-off.
- **Conflict-explainer rule logic is hand-tuned.** The 6 override rules
  reflect engineering judgement rather than a learned policy. Reproducing
  them is straightforward (the rules are in `Truth/backend/main.py`),
  but a learned policy would be more elegant.
- **`vocab_coverage` is sensitive to entity masking.** Heavy masking
  reduces the number of "in-vocabulary" tokens, which could falsely
  trigger the abstention gate. We mitigate by computing coverage on the
  *processed* (post-masking) text against the *processed* TF-IDF
  vocabulary, but this is a known weak point.
