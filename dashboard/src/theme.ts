import { useEffect, useState } from 'react';

import type { Mode } from './chartTheme';

export type ThemeChoice = 'auto' | Mode;

const KEY = 'quant-dev-theme';
const dark = () => window.matchMedia('(prefers-color-scheme: dark)');

function stored(): ThemeChoice {
  try {
    const value = localStorage.getItem(KEY);
    return value === 'light' || value === 'dark' ? value : 'auto';
  } catch {
    return 'auto';
  }
}

/** The theme choice (auto follows the OS), stamped on <html data-theme>, and the mode in effect. */
export function useTheme(): [ThemeChoice, (choice: ThemeChoice) => void, Mode] {
  const [choice, setChoice] = useState<ThemeChoice>(stored);
  const [osDark, setOsDark] = useState(() => dark().matches);

  useEffect(() => {
    const query = dark();
    const listener = (event: MediaQueryListEvent) => setOsDark(event.matches);
    query.addEventListener('change', listener);
    return () => query.removeEventListener('change', listener);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (choice === 'auto') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', choice);
    try {
      if (choice === 'auto') localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, choice);
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  }, [choice]);

  const mode: Mode = choice === 'auto' ? (osDark ? 'dark' : 'light') : choice;
  return [choice, setChoice, mode];
}
