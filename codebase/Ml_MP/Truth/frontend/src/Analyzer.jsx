import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Fingerprint, TextSearch, Sparkles, RefreshCcw, Image as ImageIcon, FileText, UploadCloud, X, AlertTriangle, CheckCircle, Globe, Clock, MapPin, Database, Activity, Newspaper, ShieldCheck } from 'lucide-react';
import FactGauge from './components/FactGauge';

function Counter({ value }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = Math.round(value * 100);
    if (start === end) return;

    let totalDuration = 2000;
    let incrementTime = totalDuration / end;

    let timer = setInterval(() => {
      start += 1;
      setDisplayValue(start);
      if (start === end) clearInterval(timer);
    }, incrementTime);

    return () => clearInterval(timer);
  }, [value]);

  return <span>{displayValue}%</span>;
}

export default function Analyzer() {
  const [inputType, setInputType] = useState('text');
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const [loading, setLoading] = useState(false);
  const [scanStep, setScanStep] = useState(0);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const scanSteps = [
    "Initializing Semantic Vector Engine...",
    "Extracting Stylometric Fingerprint...",
    "Cross-Referencing Global Fact Indexes...",
    "Mapping Geographic & Temporal Context...",
    "Finalizing Factual Consensus..."
  ];

  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setScanStep((prev) => (prev + 1) % scanSteps.length);
      }, 2300);
    } else {
      setScanStep(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const analyzeContent = async () => {
    if (inputType === 'text' && !text.trim()) return;
    if (inputType === 'file' && !file) return;

    setLoading(true);
    setResult(null);
    try {
      const formData = new FormData();
      if (inputType === 'text') formData.append('text', text);
      else if (inputType === 'file' && file) formData.append('file', file);

      // Backend URL is configurable so the same build can target localhost in
      // dev and a deployed backend (e.g. Hugging Face Space, Render) in prod.
      // Set VITE_API_URL in Netlify env vars; defaults to localhost:8000.
      const apiBase = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
      const response = await fetch(`${apiBase}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        alert("API Error: " + (data.detail || "Request Failed"));
        return;
      }
      setResult(data);
    } catch (error) {
      console.error("Error connecting to Truth backend", error);
      alert("Failed to connect to the Truth API.");
    } finally {
      setLoading(false);
    }
  };

  const getFactualStatus = (type) => {
    if (!result?.factual_analysis) return 'unknown';
    const check = result.factual_analysis.find(f => f.check_type === type);
    return check ? check.status : 'unknown';
  };

  // Map a per-sentence prob_fake into a background color (Originality.ai-style
  // gradient). Low risk = subtle green, medium = amber, high = red.
  const sentenceRiskColor = (probFake) => {
    if (probFake >= 0.70) return 'rgba(244, 63, 94, 0.18)';
    if (probFake >= 0.45) return 'rgba(245, 158, 11, 0.18)';
    return 'rgba(16, 185, 129, 0.10)';
  };

  const renderHighlightedText = () => {
    if (!result) return null;
    const rawText = result.analyzed_text || text || (file ? `Document: ${file.name}` : "");

    // Phase 1.3 / parity with Originality.ai — when the backend returns
    // per-sentence scores, render each sentence in its own span with a
    // risk-tinted background. Fall back to the word-level explainability
    // highlighter for short inputs (where sentence_scores is empty).
    const sentenceScores = Array.isArray(result.sentence_scores)
      ? result.sentence_scores
      : [];
    if (sentenceScores.length > 0) {
      // Build segments by walking char_start / char_end ranges.
      const segments = [];
      let cursor = 0;
      sentenceScores.forEach((s, i) => {
        if (s.char_start > cursor) {
          segments.push({ kind: 'gap', text: rawText.slice(cursor, s.char_start), key: `g${i}` });
        }
        segments.push({
          kind: 'sentence',
          text: rawText.slice(s.char_start, s.char_end),
          probFake: s.prob_fake,
          risk: s.risk,
          key: `s${i}`,
        });
        cursor = s.char_end;
      });
      if (cursor < rawText.length) {
        segments.push({ kind: 'gap', text: rawText.slice(cursor), key: 'g-tail' });
      }
      return segments.map((seg, idx) =>
        seg.kind === 'sentence' ? (
          <motion.span
            key={seg.key}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.03 }}
            title={`Sentence-level model fake-probability: ${(seg.probFake * 100).toFixed(0)}% (${seg.risk.toUpperCase()})`}
            style={{
              backgroundColor: sentenceRiskColor(seg.probFake),
              padding: '0.05em 0.2em',
              borderRadius: '0.35em',
              boxDecorationBreak: 'clone',
              WebkitBoxDecorationBreak: 'clone',
            }}
          >
            {seg.text}
          </motion.span>
        ) : (
          <span key={seg.key}>{seg.text}</span>
        )
      );
    }

    // Fallback: word-level highlighting from the global explainability list.
    const words = rawText.split(/(\s+)/);
    const featureMap = {};
    result.explainability.forEach(f => { featureMap[f.word.toLowerCase()] = f.color; });

    return words.map((word, index) => {
      const cleanWord = word.trim().toLowerCase().replace(/[.,!?;:]/g, '');
      if (featureMap[cleanWord]) {
        return (
          <motion.span
            key={index}
            initial={{ opacity: 0, filter: 'blur(5px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={{ delay: index * 0.01 }}
            className="highlight-token"
            style={{ backgroundColor: featureMap[cleanWord], color: 'white' }}
          >
            {word}
          </motion.span>
        );
      }
      return <span key={index}>{word}</span>;
    });
  };

  return (
    <div className="analyzer-root">
      <div className="app-container">
        <motion.header
          className="header"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="brand" style={{ position: 'relative' }}>
            <motion.div
              animate={{ scale: [1, 1.02, 1], filter: ["drop-shadow(0 0 0px rgba(59,130,246,0))", "drop-shadow(0 0 5px rgba(59,130,246,0.3))", "drop-shadow(0 0 0px rgba(59,130,246,0))"] }}
              transition={{ duration: 4, repeat: Infinity }}
              style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}
            >
              <ShieldAlert size={44} color="#3b82f6" style={{ filter: 'drop-shadow(0 0 10px rgba(59,130,246,0.2))' }} />
              <span style={{ color: '#ffffff', fontWeight: '900', textShadow: '0 0 20px rgba(59,130,246,0.5)', display: 'inline-block' }}>TruthLens</span>
            </motion.div>
          </h1>
          <p className="subtitle" style={{ fontSize: '0.9rem', letterSpacing: '0.2em', fontWeight: '900', opacity: 0.9 }}>
            NEURAL LINGUISTIC MAPPING • GLOBAL FACTUAL ANCHORING
          </p>
          <div style={{ marginTop: '1.5rem' }}>
            <a
              href="#/"
              style={{
                color: '#9a9aa2',
                fontSize: '0.75rem',
                letterSpacing: '0.25em',
                fontWeight: 700,
                textDecoration: 'none',
                padding: '0.5rem 1rem',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '999px',
              }}
            >
              ← BACK TO HOME
            </a>
          </div>
        </motion.header>

        <main>
          <AnimatePresence mode="wait">
            {!result && !loading && (
              <motion.section
                key="input"
                initial={{ opacity: 0, scale: 0.98, y: -10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 30, filter: 'blur(20px)' }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="glass-panel neon-border"
              >
                <div className="tabs-container">
                  <button className={`tab-btn ${inputType === 'text' ? 'active' : ''}`} onClick={() => setInputType('text')}>
                    <TextSearch size={18} /> Deep Text Scan
                  </button>
                  <button className={`tab-btn ${inputType === 'file' ? 'active' : ''}`} onClick={() => setInputType('file')}>
                    <ImageIcon size={18} /> Media Decryption
                  </button>
                </div>

                <div className="form-group" style={{ marginTop: '2rem' }}>
                  {inputType === 'text' ? (
                    <textarea
                      className="textarea"
                      placeholder="Enter article text for neural stylistic mapping..."
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                    />
                  ) : (
                    <motion.div
                      animate={isDragging ? { scale: 1.05, borderColor: 'var(--accent-blue)', backgroundColor: 'rgba(59, 130, 246, 0.05)' } : {}}
                      whileHover="hover"
                      className={`drop-zone ${isDragging ? 'drag-active' : ''}`}
                      onClick={() => fileInputRef.current.click()}
                      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                      onDragLeave={() => setIsDragging(false)}
                      onDrop={(e) => {
                        e.preventDefault();
                        setFile(e.dataTransfer.files[0]);
                        setIsDragging(false);
                      }}
                    >
                      <input type="file" hidden ref={fileInputRef} onChange={(e) => fileInputRef.current && setFile(e.target.files[0])} />

                      <motion.div
                          variants={{
                              hover: { y: -10, scale: 1.1, filter: "drop-shadow(0 0 15px rgba(59,130,246,0.6))" }
                          }}
                          animate={{
                              y: [0, -8, 0],
                              filter: ["drop-shadow(0 0 5px rgba(59,130,246,0.2))", "drop-shadow(0 0 15px rgba(59,130,246,0.4))", "drop-shadow(0 0 5px rgba(59,130,246,0.2))"]
                          }}
                          transition={{
                              duration: 3,
                              repeat: Infinity,
                              ease: "easeInOut"
                          }}
                          style={{ display: file ? 'none' : 'flex', justifyContent: 'center', width: '100%', marginBottom: '1rem' }}
                      >
                          <UploadCloud size={72} color="#3b82f6" />
                      </motion.div>

                      {file ? (
                        <div className="file-asset-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(59, 130, 246, 0.1)', padding: '1rem 1.5rem', borderRadius: '1rem', border: '1px solid rgba(59,130,246,0.3)', width: '100%', position: 'relative' }}>
                          <div style={{ background: 'var(--accent-blue)', padding: '0.8rem', borderRadius: '0.8rem', display: 'flex' }}>
                            <FileText size={24} color="#000" />
                          </div>
                          <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
                            <p className="drop-text" style={{ fontSize: '1rem', fontWeight: '800', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {file.name.toUpperCase()}
                            </p>
                            <p style={{ fontSize: '0.65rem', color: '#10b981', letterSpacing: '0.2em', fontWeight: '800', marginTop: '2px' }}>
                              <CheckCircle size={10} style={{ marginRight: '4px' }} /> FORENSIC LINK ESTABLISHED
                            </p>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); setFile(null); }}
                            className="btn-action"
                            style={{ padding: '0.5rem', background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
                            title="Remove Asset"
                          >
                            <X size={18} />
                          </button>
                        </div>
                      ) : (
                        <>
                          <p className="drop-text" style={{ fontSize: '1.1rem', fontWeight: '800', letterSpacing: '0.05em' }}>
                              DROP DOCUMENT OR IMAGE HERE
                          </p>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', letterSpacing: '0.2em', marginTop: '0.5rem', opacity: 0.6 }}>
                              SUPPORTED: PDF, DOCX, JPG, PNG
                          </p>
                        </>
                      )}
                    </motion.div>
                  )}

                  <motion.button
                    whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(59, 130, 246, 0.5)" }}
                    whileTap={{ scale: 0.95 }}
                    className="btn-ultimate"
                    onClick={analyzeContent}
                    disabled={loading}
                  >
                    <Sparkles size={24} />
                    Initiate Truth Analysis
                  </motion.button>
                </div>
              </motion.section>
            )}

            {loading && (
              <motion.section
                key="scan"
                initial={{ opacity: 0, scale: 1.1, filter: 'blur(10px)' }}
                animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                exit={{ opacity: 0, scale: 0.9, y: 50, filter: 'blur(30px)' }}
                className="glass-panel scan-container neon-border"
              >
                <div className="scanning-dna"></div>
                <motion.h2
                  key={scanStep}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  style={{ fontSize: '1.5rem', letterSpacing: '0.1em', fontWeight: '800', marginBottom: '0.5rem' }}
                >
                  {scanSteps[scanStep]}
                </motion.h2>
                <p style={{ color: 'var(--text-secondary)', letterSpacing: '0.2em', fontSize: '0.7rem', marginTop: '1rem' }}>PERFORMING GLOBAL VALIDATION...</p>
              </motion.section>
            )}

            {result && !loading && (
              <motion.section
                key="results"
                initial={{ opacity: 0, filter: 'blur(20px)', y: -20 }}
                animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 40, filter: 'blur(25px)' }}
                transition={{ type: 'spring', damping: 20 }}
                className="glass-panel glowing neon-border"
              >
                <div className="verdict-header" style={{ alignItems: 'flex-end' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                    <span className="subtitle" style={{ fontSize: '0.65rem', textAlign: 'left', fontWeight: '900', letterSpacing: '0.2em' }}>SYSTEM VERDICT</span>
                    <motion.div
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      className={`shield-badge ${result.prediction === 'Real News' ? 'real' : result.prediction === 'Uncertain' ? 'uncertain' : 'fake'}`}
                      style={{ fontSize: '1.5rem', padding: '0.75rem 2rem' }}
                    >
                      <Fingerprint size={24} />
                      {result.prediction.toUpperCase()}
                    </motion.div>
                    {/* Parity chips: claim type (Full Fact AI parity) + source credibility (NewsGuard parity) */}
                    {(result.claim_type || result.source_credibility) && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.2rem' }}>
                        {result.claim_type && (
                          <span style={{
                            padding: '0.3rem 0.7rem',
                            background: 'rgba(99, 102, 241, 0.08)',
                            border: '1px solid rgba(99, 102, 241, 0.3)',
                            borderRadius: '999px',
                            fontSize: '0.65rem',
                            letterSpacing: '0.15em',
                            fontWeight: 800,
                            color: '#a78bfa',
                          }}>
                            CLAIM TYPE: {String(result.claim_type).toUpperCase()}
                          </span>
                        )}
                        {result.source_credibility && (
                          <span style={{
                            padding: '0.3rem 0.7rem',
                            background:
                              result.source_credibility.tier === 'tier1' ? 'rgba(16, 185, 129, 0.10)'
                              : result.source_credibility.tier === 'factcheck' ? 'rgba(6, 182, 212, 0.10)'
                              : result.source_credibility.tier === 'tier2' ? 'rgba(59, 130, 246, 0.10)'
                              : 'rgba(245, 158, 11, 0.10)',
                            border:
                              result.source_credibility.tier === 'tier1' ? '1px solid rgba(16, 185, 129, 0.35)'
                              : result.source_credibility.tier === 'factcheck' ? '1px solid rgba(6, 182, 212, 0.35)'
                              : result.source_credibility.tier === 'tier2' ? '1px solid rgba(59, 130, 246, 0.35)'
                              : '1px solid rgba(245, 158, 11, 0.35)',
                            borderRadius: '999px',
                            fontSize: '0.65rem',
                            letterSpacing: '0.15em',
                            fontWeight: 800,
                          }}>
                            {result.source_credibility.domain.toUpperCase()} · {String(result.source_credibility.tier).toUpperCase()} ·{' '}
                            {Math.round(result.source_credibility.score * 100)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <div style={{ position: 'relative', width: '110px', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', position: 'absolute', transform: 'rotate(135deg)' }}>
                            <defs>
                                <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#3b82f6" />
                                    <stop offset="100%" stopColor="#06b6d4" />
                                </linearGradient>
                                <filter id="ringGlow">
                                    <feGaussianBlur stdDeviation="3" result="blur" />
                                    <feMerge>
                                        <feMergeNode in="blur" />
                                        <feMergeNode in="SourceGraphic" />
                                    </feMerge>
                                </filter>
                            </defs>
                            <circle
                                cx="50" cy="50" r="42"
                                fill="none"
                                stroke="rgba(255,255,255,0.05)"
                                strokeWidth="8"
                                strokeDasharray="198 264"
                                strokeLinecap="round"
                            />
                            <motion.circle
                                cx="50" cy="50" r="42"
                                fill="none"
                                stroke="url(#ringGradient)"
                                strokeWidth="8"
                                strokeLinecap="round"
                                filter="url(#ringGlow)"
                                strokeDasharray="198 264"
                                initial={{ strokeDashoffset: 198 }}
                                animate={{ strokeDashoffset: 198 - (198 * result.confidence) }}
                                transition={{ duration: 2, ease: "easeOut" }}
                            />
                        </svg>

                      <div style={{ zIndex: 10, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <h3 style={{ fontSize: '1.8rem', fontWeight: 900, color: '#06b6d4', fontFamily: 'monospace', textShadow: '0 0 15px rgba(6, 182, 212, 0.4)', margin: 0, padding: 0, lineHeight: 1 }}>
                              <Counter value={result.confidence} />
                          </h3>
                          <span className="subtitle" style={{ fontSize: '0.45rem', fontWeight: '900', letterSpacing: '0.15em', opacity: 0.7, marginTop: '4px' }}>
                              PREDICTION<br/>CONFIDENCE
                          </span>
                      </div>
                    </div>

                    <button onClick={() => setResult(null)} className="btn-action" style={{ padding: '0.8rem 1.5rem', fontSize: '0.8rem' }}>
                      <RefreshCcw size={16} /> RE-INITIATE SCAN
                    </button>
                  </div>
                </div>

                {result.prediction === 'Uncertain' && (
                  <div style={{ display: 'flex', justifyContent: 'center', width: '100%', marginBottom: '2rem' }}>
                      <motion.div
                          initial={{ y: 20, opacity: 0 }}
                          animate={{ y: 0, opacity: 1 }}
                          style={{
                              background: 'rgba(99, 102, 241, 0.08)',
                              border: '1px solid rgba(99, 102, 241, 0.4)',
                              padding: '0.8rem 2.5rem',
                              borderRadius: '100px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '1.2rem',
                              width: 'fit-content'
                          }}
                      >
                        <AlertTriangle size={20} color="#6366f1" />
                        <p style={{ margin: 0, fontSize: '0.85rem', letterSpacing: '0.08em', fontWeight: '800' }}>
                          <span style={{ color: '#6366f1', marginRight: '0.5rem' }}>INSUFFICIENT DATA:</span>
                          <span style={{ opacity: 0.9, color: '#fff' }}>Input does not contain enough recognizable news content to classify</span>
                        </p>
                      </motion.div>
                  </div>
                )}

                {result.conflict_detected && (
                  <div style={{ display: 'flex', justifyContent: 'center', width: '100%', marginBottom: '2rem' }}>
                      <motion.div
                          initial={{ y: 20, opacity: 0, scale: 0.95 }}
                          animate={{ y: 0, opacity: 1, scale: 1 }}
                          className="conflict-banner"
                          style={{
                              background: 'rgba(245, 158, 11, 0.08)',
                              border: '1px solid rgba(245, 158, 11, 0.4)',
                              padding: '0.8rem 2.5rem',
                              borderRadius: '100px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '1.2rem',
                              boxShadow: '0 0 30px rgba(245, 158, 11, 0.1)',
                              backdropFilter: 'blur(12px)',
                              width: 'fit-content'
                          }}
                      >
                        <motion.div
                          animate={{ scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] }}
                          transition={{ duration: 2, repeat: Infinity }}
                          style={{ display: 'flex', flexShrink: 0 }}
                        >
                          <AlertTriangle size={20} color="#f59e0b" />
                        </motion.div>
                        <p style={{ margin: 0, fontSize: '0.85rem', letterSpacing: '0.08em', fontWeight: '800', whiteSpace: 'nowrap' }}>
                          <span style={{ color: '#f59e0b', marginRight: '0.5rem' }}>FACTUAL CONFLICT:</span>
                          <span style={{ opacity: 0.9, color: '#fff' }}>RAG fact-checking found contradicting evidence</span>
                        </p>
                      </motion.div>
                  </div>
                )}

                <div className="dashboard-grid" style={{ gap: '1.5rem' }}>
                  <div className="fact-card" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <h3 className="metrics-title" style={{ fontSize: '0.75rem' }}><Activity size={16} /> Linguistic Anomaly Mapping</h3>
                    <div className="highlighted-text-container" style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.95rem' }}>
                      {renderHighlightedText()}
                    </div>
                  </div>

                  <div className="fact-card" style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <h3 className="metrics-title" style={{ fontSize: '0.75rem' }}><Database size={16} /> Factual Master Index</h3>
                    <div className="gauges-container" style={{ gap: '1rem' }}>
                      <FactGauge
                          label="WIKIPEDIA"
                          status={getFactualStatus('Wikipedia Link')}
                          icon={Database}
                          url={result.factual_analysis.find(c => c.check_type === 'Wikipedia Link')?.url}
                      />
                      <FactGauge
                          label="GEOGRAPHIC"
                          status={getFactualStatus('Spatial Validation')}
                          icon={MapPin}
                          url={result.factual_analysis.find(c => c.check_type === 'Spatial Validation')?.url}
                      />
                      <FactGauge
                          label="TEMPORAL"
                          status={getFactualStatus('Temporal Validation')}
                          icon={Clock}
                          url={result.factual_analysis.find(c => c.check_type === 'Temporal Validation')?.url}
                      />
                      <FactGauge
                          label="GLOBAL WEB"
                          status={getFactualStatus('Web Cross-Reference')}
                          icon={Globe}
                          url={result.factual_analysis.find(c => c.check_type === 'Web Cross-Reference')?.url}
                      />
                      <FactGauge
                          label="NEWS"
                          status={getFactualStatus('News Corroboration')}
                          icon={Newspaper}
                          url={result.factual_analysis.find(c => c.check_type === 'News Corroboration')?.url}
                      />
                      <FactGauge
                          label="FACT-CHECK"
                          status={getFactualStatus('Fact-Checker Verdict')}
                          icon={ShieldCheck}
                          url={result.factual_analysis.find(c => c.check_type === 'Fact-Checker Verdict')?.url}
                      />
                      <FactGauge
                          label="GOOGLE FACT CHECK"
                          status={getFactualStatus('Google Fact Check')}
                          icon={ShieldCheck}
                          url={result.factual_analysis.find(c => c.check_type === 'Google Fact Check')?.url}
                      />
                      <FactGauge
                          label="AI PLAUSIBILITY"
                          status={getFactualStatus('LLM Plausibility')}
                          icon={Sparkles}
                          url={result.factual_analysis.find(c => c.check_type === 'LLM Plausibility')?.url}
                      />
                    </div>
                  </div>
                </div>

                {/* CONFLICT EXPLAINER — Phase 4/5: structured ML-vs-RAG agreement breakdown.
                    Renders only when ML and RAG disagreed, OR when the vocab-coverage
                    abstention gate is active. Otherwise the verdict speaks for itself. */}
                {result.conflict_report && (result.conflict_report.disagreement || result.conflict_report.bias_gate_active) && (
                  <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="fact-card"
                    style={{
                      background: 'rgba(0,0,0,0.55)',
                      border: '1px solid rgba(99, 102, 241, 0.35)',
                      marginTop: '1.5rem',
                    }}
                  >
                    <h3 className="metrics-title" style={{ fontSize: '0.75rem' }}>
                      <ShieldAlert size={16} /> Conflict Explainer — How We Reached The Verdict
                    </h3>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'stretch', gap: '1rem', marginTop: '1rem' }}>
                      {/* ML side */}
                      <div style={{ padding: '1rem', background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '0.6rem' }}>
                        <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.18em', fontWeight: 900, color: '#3b82f6' }}>STYLE / LEXICAL MODEL</p>
                        <h4 style={{ margin: '0.4rem 0 0.2rem', fontSize: '1.05rem', fontWeight: 800 }}>{result.conflict_report.ml_verdict}</h4>
                        <p style={{ margin: 0, fontSize: '0.75rem', opacity: 0.75 }}>
                          confidence {Math.round(result.conflict_report.ml_confidence * 100)}%
                        </p>
                        {result.conflict_report.bias_gate_active && (
                          <p style={{ marginTop: '0.6rem', fontSize: '0.7rem', color: '#a78bfa' }}>
                            ⚠ vocab-coverage abstention gate engaged (input is out-of-domain)
                          </p>
                        )}
                      </div>

                      {/* Disagreement / agreement badge */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{
                          padding: '0.4rem 0.9rem',
                          borderRadius: '999px',
                          fontSize: '0.65rem',
                          letterSpacing: '0.18em',
                          fontWeight: 900,
                          background: result.conflict_report.disagreement
                            ? 'rgba(245, 158, 11, 0.12)'
                            : 'rgba(16, 185, 129, 0.10)',
                          color: result.conflict_report.disagreement ? '#f59e0b' : '#10b981',
                          border: `1px solid ${result.conflict_report.disagreement ? 'rgba(245,158,11,0.4)' : 'rgba(16,185,129,0.3)'}`,
                        }}>
                          {result.conflict_report.disagreement ? 'DISAGREE' : 'AGREE'}
                        </div>
                      </div>

                      {/* RAG side */}
                      <div style={{ padding: '1rem', background: 'rgba(6, 182, 212, 0.06)', border: '1px solid rgba(6, 182, 212, 0.2)', borderRadius: '0.6rem' }}>
                        <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.18em', fontWeight: 900, color: '#06b6d4' }}>RAG MULTI-SOURCE CONSENSUS</p>
                        <h4 style={{ margin: '0.4rem 0 0.2rem', fontSize: '1.05rem', fontWeight: 800 }}>{result.conflict_report.rag_verdict}</h4>
                        <p style={{ margin: 0, fontSize: '0.75rem', opacity: 0.75 }}>
                          consensus {Math.round(result.conflict_report.rag_confidence * 100)}%
                        </p>
                        {result.conflict_report.flagging_verifiers?.length > 0 && (
                          <p style={{ marginTop: '0.6rem', fontSize: '0.7rem', opacity: 0.8 }}>
                            flagged by: {result.conflict_report.flagging_verifiers.join(', ')}
                          </p>
                        )}
                      </div>
                    </div>

                    {result.conflict_report.triggered_rule && (
                      <p style={{
                        marginTop: '1rem',
                        padding: '0.7rem 1rem',
                        background: 'rgba(99, 102, 241, 0.08)',
                        border: '1px solid rgba(99, 102, 241, 0.25)',
                        borderRadius: '0.5rem',
                        fontSize: '0.78rem',
                        lineHeight: 1.4,
                      }}>
                        <span style={{ color: '#a78bfa', fontWeight: 800 }}>Rule applied: </span>
                        {result.conflict_report.triggered_rule}
                      </p>
                    )}

                    <p style={{ marginTop: '0.8rem', fontSize: '0.7rem', opacity: 0.6 }}>
                      Winning signal: <strong style={{ color: '#fff' }}>
                        {result.conflict_report.winning_signal === 'ml' ? 'Style/lexical model'
                          : result.conflict_report.winning_signal === 'rag' ? 'RAG multi-source consensus'
                          : 'Consensus (uncertain)'}
                      </strong>
                      {' · '}
                      Vocab coverage: <strong style={{ color: '#fff' }}>{Math.round((result.conflict_report.vocab_coverage ?? 1) * 100)}%</strong>
                    </p>
                  </motion.div>
                )}
              </motion.section>
            )}
          </AnimatePresence>
        </main>

        <footer style={{
          textAlign: 'center',
          padding: '4rem',
          fontSize: '0.85rem',
          letterSpacing: '0.8em',
          fontWeight: '900',
          color: '#fff',
          textShadow: '0 0 10px rgba(59, 130, 246, 0.8), 0 0 20px rgba(59, 130, 246, 0.4)',
          opacity: 0.6,
          background: 'linear-gradient(to top, rgba(9, 9, 11, 1), transparent)',
        }}>
          <motion.span
            animate={{ opacity: [0.4, 0.7, 0.4] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            TRUTHLENS
          </motion.span>
        </footer>
      </div>
    </div>
  );
}
