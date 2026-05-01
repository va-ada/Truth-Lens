import { motion } from 'framer-motion';
import { ShieldCheck, Zap } from 'lucide-react';
import { revealProps } from '../hooks/useReveal.js';

export default function DualCardsA() {
  return (
    <section className="landing-section dual-section" id="what">
      <div className="landing-inner">
        <div className="dual-grid">
          <motion.article className="dual-card dual-card-dark" {...revealProps({ delay: 0 })}>
            <div className="dual-card-index">01</div>
            <div className="dual-card-icon"><ShieldCheck size={22} strokeWidth={1.5} /></div>
            <h3 className="dual-card-title">Zero Data Kept</h3>
            <p className="dual-card-copy">
              Articles are processed in-memory and never persisted. Tokens are shredded
              the moment a verdict is returned.
            </p>
            <div className="shred-stage" aria-hidden="true">
              {['breaking', 'claims', 'overnight', 'sources', 'say'].map((w, i) => (
                <motion.span
                  key={w}
                  className="shred-token"
                  initial={{ opacity: 0.9, y: 0 }}
                  whileInView={{ opacity: [0.9, 0.9, 0], y: [0, 0, 18] }}
                  viewport={{ once: false, margin: '-80px' }}
                  transition={{ duration: 2.4, delay: i * 0.25, repeat: Infinity, repeatDelay: 1.2 }}
                >
                  {w}
                </motion.span>
              ))}
            </div>
          </motion.article>

          <motion.article className="dual-card dual-card-purple" {...revealProps({ delay: 0.12 })}>
            <div className="dual-card-index">02</div>
            <div className="dual-card-icon"><Zap size={22} strokeWidth={1.5} /></div>
            <h3 className="dual-card-title">Millisecond Verdicts</h3>
            <p className="dual-card-copy">
              A lean inference graph returns a verdict, confidence score, and top
              linguistic cues in under a tenth of a second on average.
            </p>
            <div className="latency-row" aria-hidden="true">
              <span className="latency-headline">Markets rally after central bank rate decision</span>
              <div className="latency-bar">
                <motion.div
                  className="latency-fill"
                  initial={{ width: 0 }}
                  whileInView={{ width: '78%' }}
                  viewport={{ once: false, margin: '-100px' }}
                  transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <div className="latency-meta">
                <span>98ms avg</span>
                <span className="verified-chip">VERIFIED</span>
              </div>
            </div>
          </motion.article>
        </div>
      </div>
    </section>
  );
}
