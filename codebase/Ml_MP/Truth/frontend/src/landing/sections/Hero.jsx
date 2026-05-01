import { motion } from 'framer-motion';
import DisplayTitle from '../components/DisplayTitle.jsx';
import Button from '../components/Button.jsx';
import IsoNewsGrid from '../components/IsoNewsGrid.jsx';
import { revealProps } from '../hooks/useReveal.js';

export default function Hero() {
  return (
    <section className="hero" id="top">
      <div className="landing-inner hero-inner">
        <motion.p className="hero-eyebrow" {...revealProps({ delay: 0.05 })}>
          <span className="hero-eyebrow-dot" /> TruthLens · Debiased Fake News Detection
        </motion.p>

        <motion.div {...revealProps({ delay: 0.12, y: 32 })}>
          <DisplayTitle
            text="MEET TRUTHLENS"
            ghost={[0]}
            as="h1"
            align="center"
            className="hero-title"
          />
        </motion.div>

        <motion.p className="hero-sub" {...revealProps({ delay: 0.22 })}>
          A self-auditing fake-news pipeline. TF-IDF + sentence embeddings + 17
          stylometric features feed a stacking ensemble; a 7-source RAG layer
          cross-checks every claim; and three things no other published system
          does — a vocab-coverage abstention gate, a probe-driven debiasing
          loop, and a structured ML-vs-RAG conflict explainer — make the
          verdict <em>traceable</em>, not just confident.
        </motion.p>

        <motion.div className="hero-cta-row" {...revealProps({ delay: 0.32 })}>
          <Button
            as="a"
            href="#/app"
            variant="primary"
            onClick={() => { window.location.hash = '#/app'; }}
          >
            Try The Scanner
          </Button>
          <Button as="a" href="#/paper" variant="ghost">
            Read The Paper
          </Button>
        </motion.div>

        <motion.div
          className="hero-grid-wrap"
          initial={{ opacity: 0, y: 48 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
        >
          <IsoNewsGrid />
        </motion.div>
      </div>
    </section>
  );
}
