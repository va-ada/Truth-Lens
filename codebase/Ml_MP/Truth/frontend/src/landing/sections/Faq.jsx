import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus } from 'lucide-react';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import { revealProps } from '../hooks/useReveal.js';

const ITEMS = [
  {
    q: 'How accurate is TruthLens?',
    a: 'In-domain (ISOT 5-fold CV): F1 0.992 ± 0.001, AUC 0.9995. Cross-dataset (ISOT→LIAR) drops to ~55% — which is exactly why TruthLens has an abstention gate and a multi-source RAG layer to catch out-of-domain claims the model alone cannot judge.',
  },
  {
    q: 'What is the bias-probe debiasing loop?',
    a: 'Out of the box, an SVM trained on the ISOT source name alone (with no content) hits 91.45% accuracy — meaning ML models trained on ISOT are mostly source-name detectors. TruthLens runs that probe before training, masks Reuters/AP/CNN-style leakage, retrains, then re-runs the probe and reports the delta. The audit report ships with every model.',
  },
  {
    q: 'What is the vocabulary-coverage abstention gate?',
    a: 'Before issuing a verdict, TruthLens checks how many input tokens overlap its training vocabulary. Below 30% overlap (or fewer than 4 matched terms) the system scales confidence down and emits "Uncertain" instead of confidently misclassifying an out-of-domain text.',
  },
  {
    q: 'What is the ML-vs-RAG conflict explainer?',
    a: 'When the stylometric model and the 7-source RAG layer disagree, the UI surfaces both verdicts side-by-side, names which override rule fired (e.g. "Rule 1: Snopes flagged this as false"), and lists the verifiers that disagreed. No black-box "we ignored the model" — the conflict is shown.',
  },
  {
    q: 'What model powers the scanner?',
    a: 'TF-IDF→SVD (150d) + sentence-transformer MiniLM (384d) + 17 hand-crafted stylometric features → SVM, Logistic Regression, and Random Forest in a stacking ensemble with an LR meta-learner, calibrated for explainability via LIME and SHAP.',
  },
  {
    q: 'Is my article data stored?',
    a: 'No. Articles are processed in memory and discarded once the verdict is returned. The only persistent cache is for fact-checker lookups, with a 7-day TTL.',
  },
  {
    q: 'Can I use TruthLens in research?',
    a: 'Yes. Code, datasets, training protocol, ablation tables, bias-audit report, and verified references are all in the public repo and the research paper linked from the footer.',
  },
];

export default function Faq() {
  const [open, setOpen] = useState(0);

  return (
    <section className="landing-section faq" id="faq">
      <div className="landing-inner faq-split">
        <motion.div className="faq-head" {...revealProps()}>
          <Eyebrow>FAQ</Eyebrow>
          <DisplayTitle
            text="QUESTIONS? ANSWERS."
            ghost={[0]}
            align="left"
            className="faq-title"
          />
          <p className="faq-sub">
            Everything you might want to know about the model, the data, and how we
            keep your articles private.
          </p>
        </motion.div>

        <motion.ul className="faq-list" {...revealProps({ delay: 0.08 })}>
          {ITEMS.map((item, i) => {
            const isOpen = open === i;
            return (
              <li key={item.q} className={`faq-item ${isOpen ? 'open' : ''}`}>
                <button
                  type="button"
                  className="faq-btn"
                  aria-expanded={isOpen}
                  onClick={() => setOpen(isOpen ? -1 : i)}
                >
                  <span className="faq-q">{item.q}</span>
                  <motion.span
                    className="faq-icon"
                    animate={{ rotate: isOpen ? 45 : 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <Plus size={18} strokeWidth={1.75} />
                  </motion.span>
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      className="faq-answer"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <p>{item.a}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </li>
            );
          })}
        </motion.ul>
      </div>
    </section>
  );
}
