export function revealProps({ delay = 0, y = 24, once = true } = {}) {
  return {
    initial: { opacity: 0, y },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once, margin: '-80px' },
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1], delay },
  };
}

export function stagger(index, step = 0.06) {
  return revealProps({ delay: index * step });
}
