function GitHubMark({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.73.5.5 5.73.5 12a11.5 11.5 0 0 0 7.86 10.94c.57.11.78-.25.78-.55v-2c-3.19.69-3.86-1.37-3.86-1.37-.52-1.32-1.27-1.67-1.27-1.67-1.04-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.69 1.25 3.34.95.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.17 1.18a10.96 10.96 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.62 1.58.23 2.75.11 3.04.73.8 1.18 1.83 1.18 3.08 0 4.41-2.69 5.38-5.26 5.67.41.36.78 1.05.78 2.12v3.14c0 .3.21.67.79.55A11.5 11.5 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5z" />
    </svg>
  );
}

const TEAM = [
  { name: 'Team · SFIT C403', roll: 'Roll Nos. 52, 54, 57, 65' },
  { name: 'Guide', roll: 'Dr. Joanne Gomes' },
];

const COLS = [
  {
    title: 'Product',
    links: [
      { label: 'What is TruthLens', href: '#what' },
      { label: 'How it works', href: '#how' },
      { label: 'Compare to others', href: '#compare' },
      { label: 'Open the scanner', href: '#/app' },
    ],
  },
  {
    title: 'Research',
    links: [
      { label: 'Paper', href: '#/paper' },
      { label: 'Model card', href: '#model' },
      { label: 'Evaluation', href: '#faq' },
      { label: 'NewsGuard', href: 'https://www.newsguardtech.com/how-it-works/' },
      { label: 'Logically.ai', href: 'https://www.logically.ai/' },
      { label: 'dEFEND (KDD ’19)', href: 'https://dl.acm.org/doi/10.1145/3292500.3330935' },
      { label: 'FANG (CIKM ’20)', href: 'https://github.com/nguyenvanhoang7398/FANG' },
      { label: 'ENDEF (SIGIR ’22)', href: 'https://arxiv.org/abs/2204.09484' },
      { label: 'AEC (ESWA 2025)', href: 'https://www.sciencedirect.com/science/article/abs/pii/S0957417425013739' },
      { label: 'VeraCT-Scan (2024)', href: 'https://arxiv.org/abs/2406.10289' },
      { label: 'ISOT dataset', href: 'https://onlineacademiccommunity.uvic.ca/isot/' },
      { label: 'LIAR dataset', href: 'https://huggingface.co/datasets/liar' },
    ],
  },
  {
    title: 'Contact',
    links: [
      { label: 'sanjay.soni@nazara.com', href: 'mailto:sanjay.soni@nazara.com' },
      { label: 'GitHub', href: '#' },
    ],
  },
];

export default function Footer() {
  return (
    <footer className="landing-footer" role="contentinfo">
      <div className="landing-inner footer-inner">
        <div className="footer-brand">
          <div className="footer-logo">
            <span className="nav-logo"><span className="nav-logo-dot" /></span>
            <span className="nav-wordmark">TruthLens</span>
          </div>
          <p className="footer-tag">
            Neural linguistic mapping · Global factual anchoring.
            <br />
            A fake news detection system built at SFIT (C403).
          </p>
          <div className="footer-team">
            {TEAM.map((t) => (
              <div key={t.name} className="footer-team-row">
                <span className="footer-team-label">{t.name}</span>
                <span className="footer-team-value">{t.roll}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="footer-cols">
          {COLS.map((col) => (
            <div key={col.title} className="footer-col">
              <h5 className="footer-col-title">{col.title}</h5>
              <ul>
                {col.links.map((l) => (
                  <li key={l.label}>
                    <a href={l.href}>{l.label}</a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      <div className="landing-inner footer-bottom">
        <span>© {new Date().getFullYear()} TruthLens · SFIT C403</span>
        <a href="#" className="footer-social" aria-label="GitHub">
          <GitHubMark size={16} />
        </a>
      </div>
    </footer>
  );
}
