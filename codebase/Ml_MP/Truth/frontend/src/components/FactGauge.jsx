import { motion } from 'framer-motion';
import { ExternalLink } from 'lucide-react';

const FactGauge = ({ label, status, value, icon: Icon, url }) => {
  const getStatusColor = (s) => {
    if (s === 'verified') return '#10b981'; // Emerald
    if (s === 'conflict') return '#ef4444'; // Red
    return '#cbd5e1'; // Brighter Slate Gray for visibility
  };

  const color = getStatusColor(status);
  
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="fact-gauge-container"
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div className="gauge-svg-wrapper">
        <svg viewBox="0 0 100 100" className="gauge-svg">
          {/* Background Track */}
          <circle 
            cx="50" cy="50" r="45" 
            fill="none" 
            stroke="rgba(255,255,255,0.05)" 
            strokeWidth="8" 
          />
          {/* Progress Bar */}
          <motion.circle
            cx="50" cy="50" r="45"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray="283"
            initial={{ strokeDashoffset: 283 }}
            animate={{ strokeDashoffset: status === 'unknown' ? 141.5 : 0 }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div className="gauge-content">
          <Icon size={20} style={{ color }} />
        </div>
      </div>
      <div className="gauge-info" style={{ textAlign: 'center', marginTop: '0.8rem' }}>
        <span className="gauge-label" style={{ fontSize: '0.65rem', letterSpacing: '0.1em' }}>{label}</span>
        <span className="gauge-status" style={{ color, marginTop: '4px', display: 'block', fontWeight: '800' }}>{status.toUpperCase()}</span>
      </div>

      {url && (
        <motion.a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.1, color: '#fff' }}
          style={{ 
            marginTop: '0.8rem', 
            fontSize: '0.6rem', 
            color: 'rgba(255,255,255,0.4)', 
            textDecoration: 'none', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '4px',
            border: '1px solid rgba(255,255,255,0.1)',
            padding: '2px 8px',
            borderRadius: '4px'
          }}
        >
          <ExternalLink size={10} /> SOURCE
        </motion.a>
      )}
    </motion.div>
  );
};

export default FactGauge;
