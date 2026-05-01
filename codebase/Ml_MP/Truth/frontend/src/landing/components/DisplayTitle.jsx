/**
 * DisplayTitle — two-tone word-by-word title.
 * Words whose index is listed in `ghost` render outlined silver (ghost);
 * other words render solid white. Matches the "TRANSPARENT / AND FAIR FINANCING"
 * treatment in the reference video.
 */
export default function DisplayTitle({ text, ghost = [], as = 'h2', className = '', align = 'center' }) {
  const Tag = as;
  const words = text.split(' ');
  const ghostSet = new Set(ghost);
  return (
    <Tag className={`display-title display-title-${align} ${className}`}>
      {words.map((w, i) => (
        <span key={i} className={ghostSet.has(i) ? 'word ghost' : 'word solid'}>
          {w}
          {i < words.length - 1 ? ' ' : ''}
        </span>
      ))}
    </Tag>
  );
}
