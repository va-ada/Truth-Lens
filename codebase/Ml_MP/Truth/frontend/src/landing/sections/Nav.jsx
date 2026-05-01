import { Search, HelpCircle, BookOpen } from 'lucide-react';
import Button from '../components/Button.jsx';

const LINKS = [
  { label: 'What is TruthLens', href: '#what' },
  { label: 'How it works', href: '#how' },
  { label: 'Compare', href: '#compare' },
  { label: 'Research', href: '#/paper' },
  { label: 'Model', href: '#model' },
];

export default function Nav() {
  return (
    <header className="nav" role="banner">
      <div className="nav-inner">
        <a href="#/" className="nav-brand" aria-label="TruthLens home">
          <span className="nav-logo" aria-hidden="true">
            <span className="nav-logo-dot" />
          </span>
          <span className="nav-wordmark">TruthLens</span>
        </a>

        <nav className="nav-links" aria-label="Primary">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="nav-link"
              {...(l.external ? { target: '_blank', rel: 'noreferrer' } : {})}
            >
              {l.label}
            </a>
          ))}
        </nav>

        <div className="nav-tools">
          <label className="nav-search" aria-label="Search">
            <Search size={14} strokeWidth={1.75} />
            <input type="text" placeholder="Search" />
            <kbd className="nav-kbd">⌘K</kbd>
          </label>
          <a href="#docs" className="nav-icon-link" aria-label="Docs">
            <BookOpen size={16} strokeWidth={1.75} />
          </a>
          <a href="#help" className="nav-icon-link" aria-label="Help">
            <HelpCircle size={16} strokeWidth={1.75} />
          </a>
          <Button
            as="a"
            href="#/app"
            variant="primary"
            className="nav-cta"
            onClick={() => {
              window.location.hash = '#/app';
            }}
          >
            Open App
          </Button>
        </div>
      </div>
    </header>
  );
}
