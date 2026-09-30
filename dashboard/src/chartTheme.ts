// The charts come from viz.py styled for a light surface. In dark mode their colours are
// re-stepped here, by role, to the dark values of the same palette (src/styles.css). The data
// and every trace are untouched; only colours change.

export type Mode = 'light' | 'dark';

const DARK = {
  surface: '#1a1a19',
  ink: '#ffffff',
  ink2: '#c3c2b7',
  grid: '#2c2c2a',
  baseline: '#383835',
};

// The palette's series steps, light -> dark.
const SERIES: Record<string, string> = {
  '#2a78d6': '#3987e5',
  '#eb6834': '#d95926',
  '#1baf7a': '#199e70',
};

/** Relative luminance of a #rgb / #rrggbb / rgb(a)() / named white or black colour, else null. */
export function luminance(colour: string): number | null {
  const c = colour.trim().toLowerCase();
  let rgb: number[] | null = null;
  if (c === 'white') rgb = [255, 255, 255];
  else if (c === 'black') rgb = [0, 0, 0];
  else if (/^#[0-9a-f]{3}$/.test(c)) rgb = [1, 2, 3].map((i) => parseInt(c[i] + c[i], 16));
  else if (/^#[0-9a-f]{6}$/.test(c)) rgb = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  else {
    const m = c.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
    if (m) rgb = [m[1], m[2], m[3]].map(Number);
  }
  if (!rgb) return null;
  const [r, g, b] = rgb.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** The dark-mode colour for a colour found under `key`, or the colour unchanged. */
export function darkColour(key: string, colour: string): string {
  const series = SERIES[colour.toLowerCase()];
  if (series) return series;
  const l = luminance(colour);
  if (l === null) return colour;
  const k = key.toLowerCase();
  if (k.endsWith('bgcolor')) return l > 0.6 ? DARK.surface : colour;
  if (k === 'gridcolor' || k === 'zerolinecolor') return l > 0.6 ? DARK.grid : colour;
  if (k === 'linecolor') return l > 0.3 ? DARK.baseline : colour;
  if (l < 0.02) return DARK.ink;          // near-black ink: text, reference lines
  if (l < 0.2) return DARK.ink2;          // dark greys and slates: secondary text
  return colour;
}

// Colour scales map values to colours; re-stepping them would change what they encode.
const SKIP = new Set(['colorscale', 'colorway', 'sequential', 'sequentialminus', 'diverging']);

function walk(value: unknown, key: string): unknown {
  if (typeof value === 'string') return key.toLowerCase().includes('color') ? darkColour(key, value) : value;
  if (Array.isArray(value)) return value.map((v) => walk(v, key));
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = SKIP.has(k.toLowerCase()) ? v : walk(v, k);
    return out;
  }
  return value;
}

/** A figure's data and layout for the given mode: unchanged in light, re-stepped in dark. */
export function themed<T extends { data: object[]; layout: Record<string, unknown> }>(
  figure: T, mode: Mode,
): T {
  if (mode === 'light') return figure;
  const layout = walk(figure.layout, 'layout') as Record<string, unknown>;
  // Plotly's default template leaves some text colours implicit; set the base font explicitly.
  layout.font = { ...(layout.font as object), color: DARK.ink2 };
  return { ...figure, data: walk(figure.data, 'data') as object[], layout };
}
