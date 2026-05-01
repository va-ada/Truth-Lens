/**
 * Dashed design-system grid shown behind every section in the reference video.
 * Uses SVG pattern on a full-section absolute layer. pointer-events:none so it
 * never intercepts clicks.
 */
export default function GridOverlay({ cols = 3, className = '' }) {
  return (
    <div className={`grid-overlay grid-overlay-${cols} ${className}`} aria-hidden="true">
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="dashed-v" width="50%" height="100%" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="100%" stroke="rgba(255,255,255,0.08)" strokeDasharray="4 6" />
          </pattern>
          <pattern id="dashed-h" width="100%" height="80" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="100%" y2="0" stroke="rgba(255,255,255,0.06)" strokeDasharray="4 6" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dashed-v)" />
        <rect width="100%" height="100%" fill="url(#dashed-h)" />
      </svg>
    </div>
  );
}
