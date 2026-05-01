# 1. Introduction

Fake news detection is a problem with no shortage of ML systems but a chronic
shortage of *honest* ML systems. The standard story — "we trained a transformer
on ISOT and got 99% accuracy" — ignores three problems that surface the moment
the model meets text it wasn't trained on:

1. **Source-name leakage.** ISOT's "real" articles are scraped from Reuters,
   so 100% of their texts begin with `<city> (Reuters) –`. A linear classifier
   trained on the source name *alone* hits **91.45%** accuracy on ISOT
   (Section 4). The "99% model" is largely a Reuters detector.
2. **Cross-dataset collapse.** The same model evaluated on the LIAR dataset of
   short political claims drops to ~55% accuracy — barely better than chance.
3. **Confident wrong answers.** With no abstention mechanism, the model emits
   high-probability verdicts on out-of-domain text where it has no business
   classifying.

TruthLens addresses each of these by treating fake-news detection as a
**self-auditing system**, not a single classifier. We introduce three
contributions, each verified unpublished as of 2026-04-30 by a paper-search
audit against the seven closest peer systems (NewsGuard, Logically.ai, dEFEND,
FANG, ENDEF, AEC, VeraCT-Scan):

- **A1 — Vocabulary-Coverage Abstention Gate.** Selective prediction in NLP
  exists (Kamath et al. ACL 2021), but operationalizing it as TF-IDF
  dictionary overlap on a stacking ensemble is novel. When less than 30% of
  input tokens overlap the training vocabulary, TruthLens scales confidence
  proportionally and emits "Uncertain".

- **A2 — Bias-Probe-Driven Debiasing Loop.** A source-name-only probe
  measures entity-leakage bias, aggressive masking + SHAP-driven feature
  reweighting reduces it, then the probe runs again and the before/after
  delta ships with the model. ENDEF (SIGIR ’22) does causal entity
  debiasing — but does not iterate as an audit protocol with measured
  numbers on a stacking ensemble.

- **A4 — Structured ML-vs-RAG Conflict Explainer.** When the stylometric
  model and the 7-source RAG layer disagree, the API returns a structured
  `ConflictReport` field naming the override rule that fired and the
  verifiers that flagged the claim. AEC (ESWA 2025) has LIME+SHAP on a
  stacking ensemble but no RAG; VeraCT-Scan has LLM rationales but no
  conflict surfacing.

The three contributions are intentionally orthogonal: A1 protects against
out-of-domain confidence, A2 protects against in-domain shortcut learning,
A4 protects against unaccountable override behaviour. Together they form a
**system-level integration + reproducible audit protocol**.

The remainder of the paper is organized as: §2 surveys related work; §3
describes the method (feature engineering, stacking, A1, A2, A4); §4 reports
experiments on ISOT and LIAR including ablation, bias-audit, and selective-risk
plots; §5 reports results; §6 discusses limitations; §7 concludes.
