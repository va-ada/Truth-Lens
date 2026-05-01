import { motion } from 'framer-motion';
import NewsCard from './NewsCard.jsx';

const SOURCES = [
  { label: 'Wikipedia', short: 'W' },
  { label: 'Google', short: 'G' },
  { label: 'NewsAPI', short: 'N' },
  { label: 'Reuters', short: 'R' },
  { label: 'AP', short: 'AP' },
  { label: 'FactCheck', short: 'FC' },
];

export default function OrbitingSources() {
  const radius = 120;
  return (
    <div className="orbit-stage" aria-hidden="true">
      <motion.div
        className="orbit-ring"
        animate={{ rotate: 360 }}
        transition={{ duration: 38, repeat: Infinity, ease: 'linear' }}
      >
        {SOURCES.map((s, i) => {
          const angle = (i / SOURCES.length) * Math.PI * 2;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          return (
            <motion.span
              key={s.label}
              className="orbit-chip"
              style={{ transform: `translate(${x}px, ${y}px)` }}
              animate={{ rotate: -360 }}
              transition={{ duration: 38, repeat: Infinity, ease: 'linear' }}
            >
              <span className="orbit-chip-short">{s.short}</span>
              <span className="orbit-chip-label">{s.label}</span>
            </motion.span>
          );
        })}
      </motion.div>
      <div className="orbit-center">
        <NewsCard
          variant="normal"
          source="CLAIM"
          timestamp="checking"
          headline="Article cross-referenced across six fact anchors"
          lines={3}
          size="sm"
        />
      </div>
    </div>
  );
}
