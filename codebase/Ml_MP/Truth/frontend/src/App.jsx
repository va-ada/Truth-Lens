import { useEffect, useState } from 'react';
import Landing from './landing/Landing.jsx';
import Analyzer from './Analyzer.jsx';
import Paper from './Paper.jsx';

function currentRoute() {
  const hash = window.location.hash.replace(/^#/, '');
  if (hash.startsWith('/app')) return 'app';
  if (hash.startsWith('/paper')) return 'paper';
  return 'home';
}

export default function App() {
  const [route, setRoute] = useState(currentRoute());

  useEffect(() => {
    const onHashChange = () => setRoute(currentRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }, [route]);

  if (route === 'app') return <Analyzer />;
  if (route === 'paper') return <Paper />;
  return <Landing />;
}
