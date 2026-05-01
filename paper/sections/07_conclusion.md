# 7. Conclusion

TruthLens addresses fake-news detection as a **self-auditing system**, not a
single classifier. We combine TF-IDF + sentence-transformer + 17 stylometric
features in a stacking ensemble, augmented with seven retrieval-augmented
verifiers and three contributions verified unpublished as of 2026-04-30:

1. A **vocabulary-coverage abstention gate** that ties selective prediction
   to TF-IDF dictionary overlap on a stacking ensemble.
2. A **bias-probe-driven debiasing loop** that quantifies entity-leakage
   bias before and after training, with the audit ` results/bias_audit.txt`
   shipping alongside the model.
3. A **structured ML-vs-RAG conflict explainer** that returns a typed
   `ConflictReport` field naming the override rule that fired and the
   verifiers that flagged the claim, surfaced both in the API and in the
   React UI.

In-domain ensemble F1 is 0.992 on ISOT 5-fold CV. Cross-domain F1 on LIAR
collapses to 0.234 — exactly the failure mode A1's abstention gate
addresses. The source-name probe registers 91.45% accuracy before the
debiasing loop; with `--debias-loop` it drops below 70% while in-domain F1
stays above 0.98.

Future work: replace the hand-tuned override rules with a learned policy;
extend to multilingual (Logically.ai parity); explore a Pareto frontier
between in-domain F1 and bias-probe accuracy; integrate a transformer
fine-tune for the cross-domain case.

The entire pipeline, datasets, evaluation protocol, ablation tables,
bias-audit script, and reference-verified competitive landscape are public
under the project repository.
