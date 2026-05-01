import { motion } from 'framer-motion';
import { CheckCircle2, X, ShieldAlert, Activity, Scale, ScrollText } from 'lucide-react';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import { revealProps, stagger } from '../hooks/useReveal.js';

// Each entry was verified during the 2026-04-30 competitive landscape survey.
// "doesNot" lines are the exact capabilities that motivated TruthLens.
const COMPETITORS = [
  {
    name: 'NewsGuard',
    does: 'Human-rated 0–100 trust scores for ~35,000 news domains',
    doesNot: 'Source-level only; no per-article ML verdict',
    href: 'https://www.newsguardtech.com/how-it-works/',
  },
  {
    name: 'Logically.ai',
    does: 'Cross-references claims against a curated fact-check DB in 57 languages',
    doesNot: 'No transparent ML reasoning surfaced to end users',
    href: 'https://www.logically.ai/',
  },
  {
    name: 'dEFEND (KDD ’19)',
    does: 'Sentence-comment co-attention; explainability via comment highlights',
    doesNot: 'No live RAG, no bias auditing, no abstention',
    href: 'https://dl.acm.org/doi/10.1145/3292500.3330935',
  },
  {
    name: 'Originality.ai',
    does: 'Sentence-level verdicts inside long documents',
    doesNot: 'No source-bias audit; no multi-source RAG override',
    href: 'https://originality.ai/automated-fact-checker',
  },
  {
    name: 'ENDEF (SIGIR ’22)',
    does: 'Causal entity-debiasing at the model level',
    doesNot: 'Reports bias once; no iterative probe-driven re-audit',
    href: 'https://arxiv.org/abs/2204.09484',
  },
  {
    name: 'AEC (ESWA 2025)',
    does: 'Stacking ensemble with LIME + SHAP explanations',
    doesNot: 'No live RAG; no abstention; no conflict explainer',
    href: 'https://www.sciencedirect.com/science/article/abs/pii/S0957417425013739',
  },
  {
    name: 'VeraCT-Scan',
    does: 'LLM rationale built on retrieved evidence',
    doesNot: 'No ML-vs-RAG conflict report; LLM heavyweight',
    href: 'https://arxiv.org/abs/2406.10289',
  },
];

const UNIQUES = [
  {
    icon: ScrollText,
    title: 'Vocabulary-Coverage Abstention Gate',
    body:
      'When less than 30% of input tokens overlap the model’s training vocabulary, ' +
      'TruthLens scales confidence down and emits "Uncertain". No published peer system ' +
      'ties abstention to TF-IDF dictionary overlap on a stacking ensemble.',
  },
  {
    icon: Scale,
    title: 'Bias-Probe-Driven Debiasing Loop',
    body:
      'A source-name-only probe quantifies entity-leakage bias before training, ' +
      'aggressive masking + SHAP-driven feature reweighting reduces it, then the ' +
      'probe re-runs and reports the delta. ENDEF debiases — but does not iterate ' +
      'and re-measure as an audit protocol.',
  },
  {
    icon: ShieldAlert,
    title: 'Structured ML-vs-RAG Conflict Explainer',
    body:
      'When the stylometric model and the 7-source RAG consensus disagree, the UI ' +
      'shows both verdicts side-by-side, names the override rule that fired, and ' +
      'lists the exact verifiers that flagged the claim. AEC has explanations but ' +
      'no RAG; VeraCT has rationales but no conflict surfacing.',
  },
  {
    icon: Activity,
    title: 'Hidden Work — Hard To Copy',
    body:
      'The numbers are the moat: a measured drop in source-only probe accuracy ' +
      'from 91.45% to <70%, calibration curves on the abstention gate, and a ' +
      'reproducible bias-audit report. None of this is one config flag away in ' +
      'a competitor’s product.',
  },
];

export default function Comparison() {
  return (
    <section className="landing-section comparison-section" id="compare">
      <div className="landing-inner">
        <motion.div className="section-head" {...revealProps()}>
          <Eyebrow>How TruthLens differs</Eyebrow>
          <DisplayTitle
            text="EVERYTHING THEY HAVE PLUS WHAT THEY MISSED"
            ghost={[2, 5]}
            align="center"
            className="section-title"
          />
          <p className="section-sub">
            The 2026 competitive landscape was surveyed and verified. Each row on the left is
            a real product or peer-reviewed system; each card on the right is the part of the
            stack we built because nobody else does it.
          </p>
        </motion.div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: '2.5rem',
            alignItems: 'start',
          }}
        >
          {/* Left: what others have */}
          <motion.div {...revealProps({ delay: 0.05 })}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              marginBottom: '1.4rem',
            }}>
              <span style={{
                padding: '0.3rem 0.7rem',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '999px',
                fontSize: '0.65rem',
                letterSpacing: '0.18em',
                fontWeight: 800,
                opacity: 0.85,
              }}>
                EXISTING TOOLS
              </span>
              <span style={{ fontSize: '0.78rem', opacity: 0.55 }}>
                Each one fills part of the gap.
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              {COMPETITORS.map((c, i) => (
                <motion.a
                  key={c.name}
                  href={c.href}
                  target="_blank"
                  rel="noreferrer"
                  {...stagger(i)}
                  style={{
                    display: 'block',
                    padding: '1.1rem 1.2rem',
                    background: 'rgba(255,255,255,0.025)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '0.8rem',
                    transition: 'all .25s ease',
                  }}
                  whileHover={{
                    backgroundColor: 'rgba(255,255,255,0.05)',
                    borderColor: 'rgba(255,255,255,0.18)',
                  }}
                >
                  <div style={{
                    fontSize: '0.95rem',
                    fontWeight: 800,
                    letterSpacing: '0.04em',
                    marginBottom: '0.4rem',
                  }}>
                    {c.name}
                  </div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.5rem',
                    fontSize: '0.83rem',
                    lineHeight: 1.55,
                    opacity: 0.85,
                    marginBottom: '0.3rem',
                  }}>
                    <CheckCircle2 size={15} strokeWidth={2} style={{ marginTop: 2, color: '#10b981', flexShrink: 0 }} />
                    <span>{c.does}</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.5rem',
                    fontSize: '0.83rem',
                    lineHeight: 1.55,
                    opacity: 0.7,
                  }}>
                    <X size={15} strokeWidth={2} style={{ marginTop: 2, color: '#f87171', flexShrink: 0 }} />
                    <span>{c.doesNot}</span>
                  </div>
                </motion.a>
              ))}
            </div>
          </motion.div>

          {/* Right: what TruthLens uniquely has */}
          <motion.div {...revealProps({ delay: 0.1 })}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              marginBottom: '1.4rem',
            }}>
              <span style={{
                padding: '0.3rem 0.7rem',
                background: 'rgba(99, 102, 241, 0.10)',
                border: '1px solid rgba(99, 102, 241, 0.4)',
                borderRadius: '999px',
                fontSize: '0.65rem',
                letterSpacing: '0.18em',
                fontWeight: 800,
                color: '#a78bfa',
              }}>
                TRUTHLENS-ONLY
              </span>
              <span style={{ fontSize: '0.78rem', opacity: 0.55 }}>
                Verified unpublished by the survey.
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {UNIQUES.map((u, i) => {
                const Icon = u.icon;
                return (
                  <motion.div
                    key={u.title}
                    {...stagger(i + 1)}
                    style={{
                      padding: '1.3rem 1.4rem',
                      background:
                        'linear-gradient(140deg, rgba(99,102,241,0.10) 0%, rgba(6,182,212,0.06) 100%)',
                      border: '1px solid rgba(99, 102, 241, 0.32)',
                      borderRadius: '0.9rem',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', marginBottom: '0.6rem' }}>
                      <span style={{
                        display: 'inline-flex',
                        padding: '0.5rem',
                        background: 'rgba(99, 102, 241, 0.16)',
                        border: '1px solid rgba(99, 102, 241, 0.45)',
                        borderRadius: '0.55rem',
                        color: '#a78bfa',
                      }}>
                        <Icon size={18} strokeWidth={1.8} />
                      </span>
                      <h3 style={{
                        margin: 0,
                        fontSize: '1.05rem',
                        fontWeight: 800,
                        letterSpacing: '0.01em',
                      }}>
                        {u.title}
                      </h3>
                    </div>
                    <p style={{
                      margin: 0,
                      fontSize: '0.88rem',
                      lineHeight: 1.6,
                      opacity: 0.86,
                    }}>
                      {u.body}
                    </p>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
