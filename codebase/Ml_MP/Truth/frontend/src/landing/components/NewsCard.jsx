/**
 * NewsCard — reusable mini news-article mockup.
 * Props:
 *  - variant: 'dim' | 'normal' | 'glow'
 *  - verdict: 'real' | 'fake' | 'uncertain' | null
 *  - confidence: 0..1 (renders a small 270deg ring when set)
 *  - source, timestamp, headline, lines (number of body bars), highlights (word indexes)
 *  - size: 'sm' | 'md' | 'lg'
 */
export default function NewsCard({
  variant = 'normal',
  verdict = null,
  confidence = null,
  source = 'THE HERALD',
  timestamp = '2h ago',
  headline = 'Breaking headline goes here in two lines of text',
  lines = 4,
  highlights = [],
  size = 'md',
  className = '',
  style,
}) {
  const words = headline.split(' ');
  const highlightSet = new Set(highlights);
  const pct = confidence != null ? Math.round(confidence * 100) : null;
  const circumference = 2 * Math.PI * 18;
  const dash = pct != null ? (pct / 100) * (circumference * 0.75) : 0;

  return (
    <article
      className={`news-card news-card-${variant} news-card-${size} ${className}`}
      style={style}
      aria-hidden="true"
    >
      <header className="news-card-head">
        <span className="news-card-source">{source}</span>
        <span className="news-card-time">{timestamp}</span>
      </header>

      <h4 className="news-card-headline">
        {words.map((w, i) => (
          <span key={i} className={highlightSet.has(i) ? 'hl' : ''}>
            {w}
            {i < words.length - 1 ? ' ' : ''}
          </span>
        ))}
      </h4>

      <div className="news-card-body">
        {Array.from({ length: lines }).map((_, i) => (
          <span
            key={i}
            className="news-card-body-bar"
            style={{ width: `${88 - i * 9}%` }}
          />
        ))}
      </div>

      {verdict && (
        <div className={`news-card-verdict news-card-verdict-${verdict}`}>
          <span className="verdict-dot" />
          <span className="verdict-label">
            {verdict === 'fake' ? 'FAKE NEWS' : verdict === 'real' ? 'VERIFIED' : 'UNCERTAIN'}
          </span>
          {pct != null && <span className="verdict-pct">{pct}%</span>}
        </div>
      )}

      {pct != null && !verdict && (
        <svg className="news-card-confidence" viewBox="0 0 44 44" width="36" height="36">
          <circle cx="22" cy="22" r="18" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="3" />
          <circle
            cx="22"
            cy="22"
            r="18"
            fill="none"
            stroke="var(--accent)"
            strokeWidth="3"
            strokeDasharray={`${dash} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(-220 22 22)"
          />
          <text x="22" y="26" textAnchor="middle" className="confidence-text">{pct}</text>
        </svg>
      )}
    </article>
  );
}
