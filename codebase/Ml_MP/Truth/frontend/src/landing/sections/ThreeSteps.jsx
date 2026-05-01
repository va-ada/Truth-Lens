import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check } from 'lucide-react';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import Button from '../components/Button.jsx';
import NewsCard from '../components/NewsCard.jsx';
import { revealProps } from '../hooks/useReveal.js';

const STEPS = [
  {
    label: 'Paste a headline',
    card: {
      source: 'INPUT',
      timestamp: 'pasted',
      headline: 'Overnight miracle cure stuns scientists worldwide',
      lines: 3,
    },
  },
  {
    label: 'Analyze',
    card: {
      source: 'SCANNER',
      timestamp: 'working',
      headline: 'Running stylometric and factual anchor checks',
      lines: 3,
      variant: 'glow',
    },
  },
  {
    label: 'Verdict',
    card: {
      source: 'RESULT',
      timestamp: '0.1s',
      headline: 'Overnight miracle cure stuns scientists worldwide',
      highlights: [0, 1, 2],
      verdict: 'fake',
      confidence: 0.94,
      lines: 3,
      variant: 'glow',
    },
  },
];

const BULLETS = [
  'Real-time inference — verdicts in under a second',
  'Explainable — every highlight is traceable',
  'Debiased — balanced across sources and regions',
  'Research-grade — reproducible, open evaluation',
];

export default function ThreeSteps() {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setIdx((i) => (i + 1) % STEPS.length), 2800);
    return () => clearInterval(id);
  }, []);

  const step = STEPS[idx];

  return (
    <section className="landing-section three-steps" id="three">
      <div className="landing-inner three-split">
        <motion.div
          className="three-phone-wrap"
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="phone-frame">
            <div className="phone-notch" />
            <div className="phone-screen">
              <div className="phone-progress">
                {STEPS.map((_, i) => (
                  <span key={i} className={`phone-progress-dot ${i === idx ? 'active' : ''}`} />
                ))}
              </div>
              <div className="phone-step-label">{step.label}</div>
              <AnimatePresence mode="wait">
                <motion.div
                  key={idx}
                  className="phone-card-slot"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                >
                  <NewsCard {...step.card} size="md" />
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        <motion.div className="three-copy" {...revealProps()}>
          <Eyebrow>How It Works</Eyebrow>
          <DisplayTitle
            text="SCAN IN THREE STEPS"
            ghost={[0, 2]}
            align="left"
            className="three-title"
          />
          <ul className="three-bullets">
            {BULLETS.map((b, i) => (
              <motion.li
                key={b}
                className="three-bullet"
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-100px' }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
              >
                <span className="three-bullet-check"><Check size={14} strokeWidth={2.25} /></span>
                {b}
              </motion.li>
            ))}
          </ul>
          <Button
            as="a"
            href="#/app"
            variant="primary"
            onClick={() => { window.location.hash = '#/app'; }}
          >
            Open The Scanner
          </Button>
        </motion.div>
      </div>
    </section>
  );
}
