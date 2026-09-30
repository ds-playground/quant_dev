import { useEffect, useState } from 'react';

const KEY = 'quant-dev-added-tickers';

/** Valid Yahoo symbols, as the API accepts them: letters, digits and ^ = . _ - */
export const SYMBOL = /^[A-Za-z0-9^=._-]{1,20}$/;

/** A symbol as typed, tidied: trimmed and upper-cased (Yahoo's symbols are upper case). */
export const tidySymbol = (text: string) => text.trim().toUpperCase();

function stored(): string[] {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(list) ? list.filter((s) => typeof s === 'string' && SYMBOL.test(s)) : [];
  } catch {
    return [];
  }
}

/** Yahoo symbols added beyond the configured list, kept in this browser between visits. */
export function useAddedTickers() {
  const [symbols, setSymbols] = useState<string[]>(stored);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(symbols));
    } catch {
      // storage unavailable: the list lasts for this page only
    }
  }, [symbols]);
  return {
    symbols,
    add: (symbol: string) => setSymbols((list) => (list.includes(symbol) ? list : [...list, symbol])),
    remove: (symbol: string) => setSymbols((list) => list.filter((s) => s !== symbol)),
  };
}
