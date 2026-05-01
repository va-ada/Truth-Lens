import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import StackedNewsLayers from '../components/StackedNewsLayers.jsx';
import { revealProps } from '../hooks/useReveal.js';

export default function UnderTheHood() {
  return (
    <section className="landing-section hood" id="model">
      <div className="landing-inner hood-split">
        <motion.div className="hood-copy" {...revealProps()}>
          <Eyebrow>Under The Hood</Eyebrow>
          <DisplayTitle
            text="A HYBRID PIPELINE BUILT FOR TRUTH"
            ghost={[0, 2, 4]}
            align="left"
            className="hood-title"
          />
          <p className="hood-p">
            A 551-dim feature vector — TF-IDF→SVD lexical (150d), MiniLM
            sentence embeddings (384d), 17 stylometric signals (burstiness,
            Zipf coefficient, readability, sentiment) — feeds an SVM + LR + RF
            stacking ensemble. On top, seven RAG verifiers (timeline,
            Wikipedia, web search, news consensus, Snopes/PolitiFact, Google
            Fact Check, geographic) cross-check live. Every stage emits LIME +
            SHAP explanations, and a published source-name probe audits bias
            at training time. Nothing is a black box.
          </p>
          <a href="#/paper" className="read-more">
            Read the research paper
            <ArrowUpRight size={16} strokeWidth={1.75} />
          </a>
        </motion.div>

        <motion.div
          className="hood-visual"
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <StackedNewsLayers />
        </motion.div>
      </div>
    </section>
  );
}
