import { expect, test, type Page } from '@playwright/test';

// Every tab of the dashboard against the real API, on the demo data (offline). Each test starts
// from a fresh page on SPX. Charts count as drawn once plotly.js has rendered their SVG.

const drawn = (page: Page) => page.locator('.plot:has(.main-svg)');
const settled = (page: Page) =>
  expect(page.locator('.stale, .loading, [aria-busy="true"]')).toHaveCount(0);

let errors: string[];

test.beforeEach(async ({ page }) => {
  errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('/');
  await expect(page.locator('label:has-text("Ticker") select')).toHaveValue('SPX');
});

test.afterEach(() => {
  expect(errors, 'browser errors').toEqual([]);
});

test('overview: tiles, four charts, the demo notice', async ({ page }) => {
  await expect(page.locator('.tile')).toHaveCount(5);
  await expect(drawn(page)).toHaveCount(4);
  await expect(page.getByRole('note')).toContainText(/not market data/i);
  await expect(page.locator('.tile').first()).toContainText('Last price');
});

test('changing the ticker and the thresholds reloads the views', async ({ page }) => {
  await settled(page);
  const spxPrice = await page.locator('.tile .value').first().textContent();
  await page.locator('label:has-text("Ticker") select').selectOption('CL');
  await expect(page.locator('.tile .value').first()).not.toHaveText(spxPrice!);
  await page.locator('label:has-text("Win above") input').fill('1');
  await page.locator('label:has-text("Loss below") input').fill('-1');
  await page.getByRole('button', { name: 'Apply' }).click();
  await expect(page.locator('.card .note').first()).toContainText('win (1%)');
  await page.locator('label:has-text("Start") input').fill('2030-01-01');
  await expect(page.getByRole('alert')).toContainText('Start date must be before end date');
  await expect(page.getByRole('button', { name: 'Apply' })).toBeDisabled();
});

test('streaks & cumulative: tables, charts, the timeline window', async ({ page }) => {
  await page.getByRole('tab', { name: 'Streaks & cumulative' }).click();
  await expect(page.locator('main table')).toHaveCount(2);
  await expect(drawn(page)).toHaveCount(5);
  await page.locator('label:has-text("Timeline window") select').selectOption('3');
  await expect(page.locator('.plot .gtitle', { hasText: '3-day streak timeline' })).toHaveCount(1);
});

test('rare events: the filters change the table', async ({ page }) => {
  await page.getByRole('tab', { name: 'Rare events' }).click();
  const heading = page.locator('main h2').first();
  await expect(heading).toContainText('rare events over 3 days');
  const count = async () => Number((await heading.textContent())!.split(' ')[0]);
  await settled(page);
  const all = await count();
  await page.locator('label:has-text("Rarer than") input').fill('2');
  await expect.poll(count).toBeLessThan(all);
  await page.locator('label:has-text("Event type") select').selectOption('cumulative');
  await settled(page);
  const types = await page.locator('main tbody tr td:first-child').allTextContents();
  expect(types.length).toBeGreaterThan(0);
  expect(new Set(types)).toEqual(new Set(['cumulative']));
});

test('statistics: four sections and the bootstrap on request', async ({ page }) => {
  await page.getByRole('tab', { name: 'Statistics' }).click();
  await expect(page.locator('main section.card h2')).toHaveText([
    /1\. Distribution/, /2\. Dependence/, /3\. Drawdowns/, /4\. How precise/]);
  await expect(drawn(page)).toHaveCount(4);
  await page.getByRole('button', { name: 'Run the bootstrap' }).click();
  await expect(page.locator('main h3', { hasText: '1,000 resamples' })).toBeVisible();
  await expect(drawn(page)).toHaveCount(6);                  // plus the two event charts
});

test('multi-ticker: the chosen tickers on request', async ({ page }) => {
  await page.getByRole('tab', { name: 'Multi-ticker' }).click();
  const demo = page.locator('fieldset', { hasText: 'Demo data (offline)' });
  // The selection starts as the Data control's set: every demo ticker.
  await expect(demo.getByRole('checkbox', { checked: true })).toHaveCount(10);
  await page.getByRole('button', { name: 'Compare 10 tickers' }).click();
  await expect(page.locator('main details')).toHaveCount(10);
  // A smaller choice: none, then two.
  await demo.getByRole('button', { name: 'None' }).click();
  await expect(page.getByRole('button', { name: 'Compare 0 tickers' })).toBeDisabled();
  await demo.getByRole('checkbox', { name: /Apple/ }).check();
  await demo.getByRole('checkbox', { name: /Coca-Cola/ }).check();
  await page.getByRole('button', { name: 'Compare 2 tickers' }).click();
  await expect(page.locator('main details')).toHaveCount(2);
  await expect(page.locator('main details summary')).toContainText([/Apple/, /Coca-Cola/]);
});

test('the saved-CSV choice loads without an error', async ({ page }) => {
  await page.locator('label:has-text("Data") select').selectOption('local');
  // Nothing saved gives the explanation; saved data gives the overview.
  await expect(page.getByText('Nothing saved yet').or(page.locator('.tile').first())).toBeVisible();
});

test('dark theme re-colours the page and the charts', async ({ page }) => {
  await expect(drawn(page)).toHaveCount(4);
  await page.locator('label:has-text("Theme") select').selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(13, 13, 13)');
  // Plotly paints the figure's paper colour on its root SVG.
  await expect(page.locator('.plot .main-svg').first()).toHaveCSS('background-color', 'rgb(26, 26, 25)');
});

test('phone width: no sideways scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(drawn(page)).toHaveCount(4);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBe(0);
});

test('the API and its docs are served alongside', async ({ request }) => {
  expect((await (await request.get('/api/health')).json()).dashboard_built).toBe(true);
  // /docs is FastAPI's page, whose Swagger UI script comes from a CDN: check the page and the
  // schema it reads, not the CDN.
  const docs = await request.get('/docs');
  expect(docs.status()).toBe(200);
  expect(await docs.text()).toContain('swagger-ui');
  const schema = await (await request.get('/openapi.json')).json();
  expect(Object.keys(schema.paths)).toContain('/api/rare-events');
});
