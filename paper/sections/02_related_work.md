# 2. Related Work

The 2026-04-30 competitive landscape survey surveyed seven systems closest to
TruthLens — three commercial, four academic — verifying every claim by URL
fetch. Each addresses part of the fake-news detection problem; none addresses
the same combination TruthLens does.

**NewsGuard** [^newsguard] provides human-rated 0-100 trust scores for
~35,000 news domains, surfaced as a browser extension. The classifier is
implicit — humans rate the source, not the article — so the system is
source-level, not article-level.

**Logically.ai** [^logically] cross-references claims against a curated
fact-check database in 57 languages. The ML reasoning is opaque to end users.

**dEFEND** (Shu et al., KDD ’19) [^defend] introduced sentence-comment
co-attention to highlight which sentences and which user comments drove a
verdict. Strong on explainability, but no live RAG, no abstention, no bias
audit, and a closed evaluation on FakeNewsNet (PolitiFact + GossipCop).

**FANG** (Nguyen et al., CIKM ’20) [^fang] is a GraphSAGE-style inductive
graph neural network over a user-source-article social graph. Strong on
diffusion modelling, but requires social graph data.

**ENDEF** (Zhu et al., SIGIR ’22) [^endef] proposes a causal-graph framework
for entity debiasing — model the entity contribution separately and remove
its direct effect at inference time. ENDEF is the closest published peer to
A2: it diagnoses and mitigates entity bias. The difference: ENDEF reports
bias once and trains once. TruthLens treats bias measurement as an *iterative
audit protocol* — probe before, mitigate, probe after, ship the delta.

**AEC** (Khan et al., ESWA 2025) [^aec] is the closest published peer to
TruthLens' classical-ML half: a stacking ensemble (SVM, LR, RF) with both
LIME and SHAP explanations. The difference: AEC has no RAG layer, no
abstention, no conflict surfacing, no bias audit.

**VeraCT-Scan** (2024) [^veract] is the closest peer on the RAG side: an
LLM produces "justifiable reasoning" given retrieved evidence. The
difference: VeraCT-Scan generates text rationales; TruthLens emits a
*structured conflict report* that names the override rule that fired and
the verifiers that flagged the claim. VeraCT also has no classical-ML half
and no abstention gate.

Methodological references shape the framing:

- **Kamath et al.** ("The Art of Abstention", ACL 2021) [^kamath] survey
  selective prediction in NLP. Standard tools include softmax MaxProb,
  MC-dropout, ensembles, and temperature scaling. None tie abstention to
  TF-IDF dictionary overlap on a stacking ensemble — the operationalization
  in A1.
- **Schuster et al.** ("The Limitations of Stylometry", CL 2020) [^schuster]
  argue stylometric features are weak against sophisticated machine-generated
  misinformation. We treat stylometry as one of three orthogonal pipelines,
  not the sole signal.
- **Cross-Domain Failures of FND** (Janicka et al.) [^janicka] document the
  ~20% accuracy drop fake-news detectors suffer when evaluated out-of-domain.
  Our cross-dataset ISOT→LIAR results in §4 reproduce this finding and
  motivate the abstention gate.

What is *not* covered by any of the closely-related systems above:

| Capability | NewsGuard | Logically | dEFEND | FANG | ENDEF | AEC | VeraCT | TruthLens |
|---|---|---|---|---|---|---|---|---|
| Vocab-coverage abstention | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Probe-driven debiasing **loop** | ❌ | ❌ | ❌ | ❌ | partial | ❌ | ❌ | ✅ |
| Structured ML-vs-RAG conflict report | ❌ | partial | ❌ | ❌ | ❌ | ❌ | partial | ✅ |

The combination of all three is what TruthLens claims as novel.

[^newsguard]: NewsGuard. *How NewsGuard Works.* https://www.newsguardtech.com/how-it-works/
[^logically]: Logically. *Logically AI*. https://www.logically.ai/
[^defend]: Shu et al. *dEFEND: Explainable Fake News Detection.* KDD 2019. https://dl.acm.org/doi/10.1145/3292500.3330935
[^fang]: Nguyen et al. *FANG: Leveraging Social Context for Fake News Detection.* CIKM 2020. https://github.com/nguyenvanhoang7398/FANG
[^endef]: Zhu et al. *Generalizing to the Future: Mitigating Entity Bias in Fake News Detection.* SIGIR 2022. https://arxiv.org/abs/2204.09484
[^aec]: Khan et al. *Adaptive Ensemble Classifier with LIME+SHAP for FND.* Expert Systems with Applications, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0957417425013739
[^veract]: VeraCT Scan. arXiv 2024. https://arxiv.org/abs/2406.10289
[^kamath]: Kamath et al. *The Art of Abstention: Selective Prediction and Error Regularization for NLP.* ACL 2021. https://aclanthology.org/2021.acl-long.84/
[^schuster]: Schuster et al. *The Limitations of Stylometry for Detecting Machine-Generated Fake News.* CL 2020. https://aclanthology.org/2020.cl-2.8/
[^janicka]: Janicka et al. *Cross-Domain Failures of Fake News Detection.* https://www.cys.cic.ipn.mx/index.php/CyS/article/view/3281/0
