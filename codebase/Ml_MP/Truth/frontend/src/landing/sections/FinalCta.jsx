import { motion } from 'framer-motion';
import Eyebrow from '../components/Eyebrow.jsx';
import DisplayTitle from '../components/DisplayTitle.jsx';
import Button from '../components/Button.jsx';
import { revealProps } from '../hooks/useReveal.js';

export default function FinalCta() {
  return (
    <section className="landing-section final-cta" id="cta">
      <div className="landing-inner final-cta-inner">
        <div className="final-cta-glow" aria-hidden="true" />
        <motion.div {...revealProps()}>
          <Eyebrow>Ready when you are</Eyebrow>
        </motion.div>
        <motion.div {...revealProps({ delay: 0.1, y: 32 })}>
          <DisplayTitle
            text="TRY TRUTHLENS NOW"
            ghost={[0, 2]}
            align="center"
            className="final-cta-title"
          />
        </motion.div>
        <motion.p className="final-cta-sub" {...revealProps({ delay: 0.18 })}>
          One click to the live scanner. Paste any article or headline and see an
          explainable, debiased verdict in milliseconds.
        </motion.p>
        <motion.div className="final-cta-row" {...revealProps({ delay: 0.26 })}>
          <Button
            as="a"
            href="#/app"
            variant="primary"
            onClick={() => { window.location.hash = '#/app'; }}
          >
            Get Started
          </Button>
          <Button as="a" href="#/paper" variant="ghost">
            Read The Paper
          </Button>
        </motion.div>
      </div>
    </section>
  );
}
