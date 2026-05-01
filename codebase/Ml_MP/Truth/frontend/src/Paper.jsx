import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ScrollText, Clock, ArrowLeft, ExternalLink } from 'lucide-react';
import './landing/landing.css';

/**
 * Paper "Coming Soon" stub.
 *
 * The full paper PDF + figures still need final layout. Until then this
 * route shows a placeholder with a link to the in-progress markdown
 * sections on GitHub so reviewers can still read the draft.
 */
export default function Paper() {
  useEffect(() => {
    document.body.classList.add('landing-active');
    return () => document.body.classList.remove('landing-active');
  }, []);

  return (
    <div className="landing-root">
      <main
        className="landing-main"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '4rem 1.5rem',
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          style={{
            maxWidth: '640px',
            width: '100%',
            textAlign: 'center',
            background: 'rgba(255,255,255,0.025)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '1.4rem',
            padding: '3.5rem 2.5rem',
            backdropFilter: 'blur(12px)',
          }}
        >
          <motion.div
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.15, duration: 0.6 }}
            style={{
              display: 'inline-flex',
              padding: '1.1rem',
              background: 'rgba(99, 102, 241, 0.12)',
              border: '1px solid rgba(99, 102, 241, 0.4)',
              borderRadius: '1.1rem',
              color: '#a78bfa',
              marginBottom: '2rem',
            }}
          >
            <ScrollText size={36} strokeWidth={1.6} />
          </motion.div>

          <p
            style={{
              fontSize: '0.7rem',
              letterSpacing: '0.32em',
              fontWeight: 800,
              color: '#a78bfa',
              marginBottom: '1.2rem',
            }}
          >
            <Clock size={11} style={{ display: 'inline', marginRight: '0.4rem', verticalAlign: 'middle' }} />
            COMING SOON
          </p>

          <h1
            style={{
              fontSize: 'clamp(2rem, 4vw, 3rem)',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              lineHeight: 1.1,
              marginBottom: '1.5rem',
            }}
          >
            The Paper Is On The Way
          </h1>

          <p
            style={{
              fontSize: '1rem',
              lineHeight: 1.7,
              opacity: 0.78,
              marginBottom: '2.5rem',
            }}
          >
            We're putting the final figures, ablation tables, and bias-audit
            numbers into camera-ready layout. Until that's published, the
            in-progress markdown sections — methods, experiments, results,
            references — are public on GitHub.
          </p>

          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.8rem',
              justifyContent: 'center',
            }}
          >
            <a
              href="#/"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.35rem',
                background: 'transparent',
                color: '#fff',
                border: '1px solid rgba(255,255,255,0.18)',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: 600,
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                textDecoration: 'none',
              }}
            >
              <ArrowLeft size={14} strokeWidth={2} />
              Back Home
            </a>
            <a
              href="https://github.com/va-ada/Truth-Lens/tree/main/report%28research%20paper%29"
              target="_blank"
              rel="noreferrer"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.35rem',
                background: 'linear-gradient(180deg, #ffffff 0%, #cfcfd4 100%)',
                color: '#0a0a0a',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: 700,
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                textDecoration: 'none',
                boxShadow: '0 1px 0 rgba(255,255,255,0.6) inset, 0 10px 30px -10px rgba(0,0,0,0.6)',
              }}
            >
              Read Drafts On GitHub
              <ExternalLink size={14} strokeWidth={2} />
            </a>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
