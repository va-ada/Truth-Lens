import { motion } from 'framer-motion';
import { Smartphone, FileText, Link2, Image as ImageIcon, Type } from 'lucide-react';
import { revealProps } from '../hooks/useReveal.js';

const INPUTS = [
  { label: 'Text', icon: Type, active: true },
  { label: 'Article URL', icon: Link2 },
  { label: 'PDF', icon: FileText },
  { label: 'Screenshot', icon: ImageIcon },
];

export default function DualCardsB() {
  return (
    <section className="landing-section dual-section dual-section-b">
      <div className="landing-inner">
        <div className="dual-grid">
          <motion.article className="dual-card dual-card-gradient" {...revealProps({ delay: 0 })}>
            <div className="dual-card-index">03</div>
            <div className="dual-card-icon"><Smartphone size={22} strokeWidth={1.5} /></div>
            <h3 className="dual-card-title">Read The News Safely</h3>
            <p className="dual-card-copy">
              Scroll any feed with a verdict badge on every headline. TruthLens rides
              alongside your reading — not between you and the article.
            </p>
            <div className="phone-silhouette" aria-hidden="true">
              <div className="phone-notch" />
              <div className="phone-feed">
                {[
                  { h: 'Scientists confirm ancient water on Mars', v: 'real' },
                  { h: 'Miracle overnight cure sweeps social feeds', v: 'fake' },
                  { h: 'Regional metro line approved by council', v: 'real' },
                  { h: 'Sports federation announces youth scheme', v: 'real' },
                ].map((row, i) => (
                  <div key={i} className={`phone-feed-row phone-feed-${row.v}`}>
                    <span className="phone-feed-dot" />
                    <span className="phone-feed-head">{row.h}</span>
                    <span className={`phone-feed-chip chip-${row.v}`}>
                      {row.v === 'fake' ? 'FAKE' : 'REAL'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </motion.article>

          <motion.article className="dual-card dual-card-dark" {...revealProps({ delay: 0.12 })}>
            <div className="dual-card-index">04</div>
            <div className="dual-card-icon"><FileText size={22} strokeWidth={1.5} /></div>
            <h3 className="dual-card-title">Text, URL, PDF, Or Image</h3>
            <p className="dual-card-copy">
              Drop in whatever you have. The scanner normalizes the input, extracts the
              article body, and runs the same pipeline end-to-end.
            </p>
            <div className="input-selector" aria-hidden="true">
              {INPUTS.map((inp) => {
                const Icon = inp.icon;
                return (
                  <span key={inp.label} className={`input-pill ${inp.active ? 'active' : ''}`}>
                    <Icon size={14} strokeWidth={1.75} />
                    {inp.label}
                  </span>
                );
              })}
            </div>
            <div className="input-preview">
              <span className="input-preview-label">PREVIEW</span>
              <div className="input-preview-lines">
                <span style={{ width: '88%' }} />
                <span style={{ width: '72%' }} />
                <span style={{ width: '56%' }} />
              </div>
            </div>
          </motion.article>
        </div>
      </div>
    </section>
  );
}
