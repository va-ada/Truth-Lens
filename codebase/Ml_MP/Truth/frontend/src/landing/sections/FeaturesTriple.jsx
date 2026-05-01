import { motion } from 'framer-motion';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import NewsCard from '../components/NewsCard.jsx';
import OrbitingSources from '../components/OrbitingSources.jsx';
import { revealProps, stagger } from '../hooks/useReveal.js';

const CONF = 0.86;
const CIRC = 2 * Math.PI * 46;

export default function FeaturesTriple() {
  return (
    <section className="landing-section features-triple" id="how">
      <div className="landing-inner">
        <motion.div className="section-head" {...revealProps()}>
          <Eyebrow>Why TruthLens</Eyebrow>
          <DisplayTitle
            text="EXPLAINABLE AND DEBIASED DETECTION"
            ghost={[0, 3]}
            align="center"
            className="section-title"
          />
          <p className="section-sub">
            Three signals combine to deliver a verdict you can trace: linguistic
            fingerprinting, live factual anchoring, and a calibrated confidence engine.
          </p>
        </motion.div>

        <div className="features-grid">
          <motion.div className="feature-col" {...stagger(0)}>
            <div className="feature-visual">
              <NewsCard
                variant="normal"
                source="DAILY POST"
                timestamp="3m ago"
                headline="Shocking unbelievable breakthrough stuns experts overnight"
                highlights={[0, 1, 5]}
                lines={4}
                size="md"
              />
            </div>
            <h3 className="feature-title">Stylometric Fingerprint</h3>
            <p className="feature-copy">
              Burstiness, Zipf coefficient, sentiment, readability, capitalization —
              17 hand-crafted features expose the writing-style fingerprint of fake
              news. LIME and SHAP make every word's contribution visible.
            </p>
          </motion.div>

          <motion.div className="feature-col" {...stagger(1)}>
            <div className="feature-visual feature-visual-orbit">
              <OrbitingSources />
            </div>
            <h3 className="feature-title">7-Source Factual Anchor</h3>
            <p className="feature-copy">
              Wikipedia, DuckDuckGo web, trusted-news consensus, Snopes /
              PolitiFact / FullFact, Google Fact Check Tools, calendar / temporal,
              and OpenStreetMap geographic — each claim is cross-checked, and
              when sources disagree we say which one won.
            </p>
          </motion.div>

          <motion.div className="feature-col" {...stagger(2)}>
            <div className="feature-visual feature-visual-conf">
              <NewsCard
                variant="normal"
                source="CITY TIMES"
                timestamp="just now"
                headline="Regional transport authority approves new metro line"
                lines={3}
                size="sm"
                className="conf-card"
              />
              <svg className="conf-ring" viewBox="0 0 120 120" width="120" height="120" aria-hidden="true">
                <circle cx="60" cy="60" r="46" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="6" />
                <circle
                  cx="60"
                  cy="60"
                  r="46"
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeDasharray={`${CONF * CIRC * 0.75} ${CIRC}`}
                  transform="rotate(-220 60 60)"
                />
                <text x="60" y="58" textAnchor="middle" className="conf-ring-num">{Math.round(CONF * 100)}</text>
                <text x="60" y="74" textAnchor="middle" className="conf-ring-label">CONFIDENCE</text>
              </svg>
            </div>
            <h3 className="feature-title">Self-Auditing Confidence</h3>
            <p className="feature-copy">
              A vocabulary-coverage gate forces "Uncertain" on out-of-domain
              input, a source-name bias probe audits training-time leakage, and
              a calibrated 270° ring shows the honest residual confidence —
              never a confidently wrong answer.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
