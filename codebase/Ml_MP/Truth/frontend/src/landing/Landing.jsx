import { useEffect } from 'react';
import './landing.css';
import Nav from './sections/Nav.jsx';
import Hero from './sections/Hero.jsx';
import FeaturesTriple from './sections/FeaturesTriple.jsx';
import UnderTheHood from './sections/UnderTheHood.jsx';
import DualCardsA from './sections/DualCardsA.jsx';
import DualCardsB from './sections/DualCardsB.jsx';
import Comparison from './sections/Comparison.jsx';
import ThreeSteps from './sections/ThreeSteps.jsx';
import Faq from './sections/Faq.jsx';
import FinalCta from './sections/FinalCta.jsx';
import Footer from './sections/Footer.jsx';
import GridOverlay from './components/GridOverlay.jsx';
import useSmoothScroll from './hooks/useSmoothScroll.js';

export default function Landing() {
  useSmoothScroll();

  useEffect(() => {
    document.body.classList.add('landing-active');
    return () => document.body.classList.remove('landing-active');
  }, []);

  return (
    <div className="landing-root">
      <GridOverlay cols={3} className="landing-backdrop-grid" />
      <Nav />
      <main className="landing-main">
        <Hero />
        <FeaturesTriple />
        <UnderTheHood />
        <DualCardsA />
        <DualCardsB />
        <Comparison />
        <ThreeSteps />
        <Faq />
        <FinalCta />
      </main>
      <Footer />
    </div>
  );
}
