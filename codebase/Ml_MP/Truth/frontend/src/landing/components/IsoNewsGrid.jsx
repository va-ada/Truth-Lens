import { motion } from 'framer-motion';
import NewsCard from './NewsCard.jsx';

const HEADLINES = [
  'Scientists confirm ancient water traces on Mars surface',
  'Markets rally after central bank rate decision',
  'New migratory pattern observed in arctic birds',
  'Regional transport authority approves new metro line',
  'University team wins national robotics championship',
  'Weather service issues coastal advisory for weekend',
  'Clean energy grid crosses milestone in three states',
  'Local library expands community reading program',
  'Space agency schedules crewed launch for autumn',
  'Education ministry releases revised curriculum draft',
  'City council debates updated zoning proposal',
  'Sports federation announces youth training scheme',
  'Researchers publish open dataset on urban traffic',
  'Coastal wetlands restoration enters final phase',
  'Startup raises funding for agriculture analytics',
];

const SOURCES = ['THE HERALD', 'DAILY POST', 'CITY TIMES', 'WORLD WIRE', 'THE LEDGER', 'NEWSLINE'];

const ROWS = 5;
const COLS = 7;
const CENTER_ROW = 2;
const CENTER_COL = 3;

export default function IsoNewsGrid() {
  const tiles = [];
  let h = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const isCenter = r === CENTER_ROW && c === CENTER_COL;
      tiles.push({
        r,
        c,
        isCenter,
        headline: HEADLINES[h % HEADLINES.length],
        source: SOURCES[(r + c) % SOURCES.length],
      });
      h++;
    }
  }

  return (
    <div className="iso-stage" aria-hidden="true">
      <div className="iso-glow iso-glow-a" />
      <div className="iso-glow iso-glow-b" />
      <div className="iso-grid">
        {tiles.map((t) => {
          if (t.isCenter) {
            return (
              <motion.div
                key={`${t.r}-${t.c}`}
                className="iso-tile iso-raised"
                style={{ gridRow: t.r + 1, gridColumn: t.c + 1 }}
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 4.2, repeat: Infinity, ease: 'easeInOut' }}
              >
                <NewsCard
                  variant="glow"
                  verdict="fake"
                  confidence={0.94}
                  source="THE HERALD"
                  timestamp="JUST NOW"
                  headline="Miracle cure claims spread across social feeds overnight"
                  highlights={[0, 1, 4]}
                  size="lg"
                />
              </motion.div>
            );
          }
          return (
            <div
              key={`${t.r}-${t.c}`}
              className="iso-tile"
              style={{ gridRow: t.r + 1, gridColumn: t.c + 1 }}
            >
              <NewsCard
                variant="dim"
                source={t.source}
                timestamp={`${((t.r + t.c) % 9) + 1}h ago`}
                headline={t.headline}
                lines={3}
                size="sm"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
