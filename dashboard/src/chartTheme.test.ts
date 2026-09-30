import { describe, expect, it } from 'vitest';

import { darkColour, luminance, themed } from './chartTheme';

const figure = {
  data: [
    { type: 'scatter', x: [1, 2], y: [3, 4], line: { color: '#2a78d6', width: 2 } },
    { type: 'scatter', line: { color: '#0b0b0b', dash: 'dash' }, name: 'Annualized' },
    { type: 'bar', marker: { color: ['#1D9E75', '#D85A30'] } },
    { type: 'heatmap', z: [[1]], colorscale: [[0, '#ffffff'], [1, '#08306b']] },
  ],
  layout: {
    height: 380,
    paper_bgcolor: 'white',
    plot_bgcolor: '#fcfcfb',
    xaxis: { gridcolor: '#e1e0d9', linecolor: '#c3c2b7', title: { text: 'Date' } },
    yaxis2: { gridcolor: '#f0f0f0', zerolinecolor: 'white' },
    font: { family: 'system-ui', color: '#52514e' },
    annotations: [{ text: 'Normal model', font: { color: '#0b0b0b' } }],
    template: { layout: { colorway: ['#636efa', '#EF553B'] } },
  },
};

describe('luminance', () => {
  it('reads hex, rgb and named colours', () => {
    expect(luminance('white')).toBeCloseTo(1);
    expect(luminance('#000')).toBe(0);
    expect(luminance('rgba(255, 255, 255, 0.5)')).toBeCloseTo(1);
    expect(luminance('not a colour')).toBeNull();
  });
});

describe('themed', () => {
  it('leaves the light figure untouched', () => {
    expect(themed(figure, 'light')).toBe(figure);
  });

  it('re-steps colours by role in dark mode', () => {
    const dark = themed(figure, 'dark');
    const layout = dark.layout as typeof figure.layout;
    expect(layout.paper_bgcolor).toBe('#1a1a19');
    expect(layout.plot_bgcolor).toBe('#1a1a19');
    expect(layout.xaxis.gridcolor).toBe('#2c2c2a');
    expect(layout.xaxis.linecolor).toBe('#383835');
    expect(layout.yaxis2).toEqual({ gridcolor: '#2c2c2a', zerolinecolor: '#2c2c2a' });
    expect(layout.annotations[0].font.color).toBe('#ffffff');
    expect(layout.font).toEqual({ family: 'system-ui', color: '#c3c2b7' });
    const [series, ink, bars, heatmap] = dark.data as typeof figure.data;
    expect(series.line!.color).toBe('#3987e5');            // the series' dark step
    expect(ink.line!.color).toBe('#ffffff');               // black ink turns white
    expect(bars.marker!.color).toEqual(['#1D9E75', '#D85A30']);   // mid-tone fills keep their hue
    expect(heatmap.colorscale).toEqual(figure.data[3].colorscale); // scales encode values: untouched
    expect(layout.template).toEqual(figure.layout.template);       // colorway untouched
  });

  it('changes nothing but colours', () => {
    const dark = themed(figure, 'dark');
    const strip = (v: unknown): unknown =>
      Array.isArray(v) ? v.map(strip)
        : v && typeof v === 'object'
          ? Object.fromEntries(Object.entries(v).filter(([k]) => !k.toLowerCase().includes('color'))
            .map(([k, x]) => [k, strip(x)]))
          : v;
    expect(strip(dark)).toEqual(strip(figure));
  });

  it('keeps unknown values under colour keys', () => {
    expect(darkColour('color', '#1D9E75')).toBe('#1D9E75');
    expect(darkColour('fillcolor', 'rgba(29,158,117,0.15)')).toBe('rgba(29,158,117,0.15)');
  });
});
