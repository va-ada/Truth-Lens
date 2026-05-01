import { motion } from 'framer-motion';
import NewsCard from './NewsCard.jsx';

const STEPS = [
  { label: '01 · Preprocessing', source: 'INGEST', timestamp: 'step 1', headline: 'Tokenizing and cleaning raw article text' },
  { label: '02 · Embeddings', source: 'SVD + TF-IDF', timestamp: 'step 2', headline: 'Projecting text into debiased feature space' },
  { label: '03 · Classifier', source: 'RANDOM FOREST', timestamp: 'step 3', headline: 'Evaluating stylometric and semantic signals' },
  { label: '04 · Explainer', source: 'FACT ANCHOR', timestamp: 'step 4', headline: 'Cross-checking claims against live sources' },
];

export default function StackedNewsLayers() {
  return (
    <div className="stacked-stage" aria-hidden="true">
      {STEPS.map((s, i) => (
        <motion.div
          key={s.label}
          className="stacked-layer"
          style={{
            zIndex: STEPS.length - i,
            transform: `translate(${i * 18}px, ${i * 22}px)`,
            opacity: 1 - i * 0.14,
          }}
          initial={{ opacity: 0, y: 40 + i * 10 }}
          whileInView={{ opacity: 1 - i * 0.14, y: i * 22 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.8, delay: 0.12 * i, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="stacked-layer-tag">{s.label}</div>
          <NewsCard
            variant={i === 0 ? 'glow' : 'normal'}
            source={s.source}
            timestamp={s.timestamp}
            headline={s.headline}
            lines={3}
            size="md"
          />
        </motion.div>
      ))}
    </div>
  );
}
