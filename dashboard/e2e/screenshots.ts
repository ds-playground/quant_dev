import { expect, test, type Page } from '@playwright/test';

// The screenshots in docs/dashboard.md: every tab on the synthetic data (SYN-INDEX unless noted), at desktop
// width, plus the dark theme and a phone. Not part of `npm run e2e`; run `npm run screenshots`
// after a change to the UI, then look at each image before committing it.

const OUT = '../docs/images/dashboard';
const drawn = (page: Page) => page.locator('.plot:has(.main-svg)');
const settled = (page: Page) =>
  expect(page.locator('.stale, .loading, [aria-busy="true"]')).toHaveCount(0);

// `from`: crop to the page below this element's top (the tab bar, so the header and parameter
// panel, shown on the overview image, are not repeated), down to `to`'s bottom (default: the end).
async function shoot(page: Page, name: string, charts: number,
                     { fullPage = true, from, to }: { fullPage?: boolean; from?: string; to?: string } = {}) {
  await expect(drawn(page)).toHaveCount(charts);
  await settled(page);
  await page.waitForTimeout(500);                    // let plotly finish its transitions
  const path = `${OUT}/${name}.png`;
  if (!from) return void await page.screenshot({ path, fullPage });
  const top = (await page.locator(from).first().boundingBox())!;
  const scroll = await page.evaluate(() => window.scrollY);
  const end = to ? (await page.locator(to).last().boundingBox())! : null;
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  const y = top.y + scroll - 8;
  const bottom = end ? end.y + scroll + end.height + 16 : height;
  await page.screenshot({ path, fullPage: true,
                          clip: { x: 0, y, width: page.viewportSize()!.width, height: bottom - y } });
}

const TABS = 'nav.tabs';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('label:has-text("Ticker") select')).toHaveValue('SYN-INDEX');
});

test('overview', async ({ page }) => {
  await shoot(page, 'overview', 4);
});

test('streaks and cumulative', async ({ page }) => {
  await page.getByRole('tab', { name: 'Streaks & cumulative' }).click();
  await shoot(page, 'streaks', 5, { from: TABS });
});

test('rare events', async ({ page }) => {
  await page.getByRole('tab', { name: 'Rare events' }).click();
  await page.locator('label:has-text("Rarer than") input').fill('5');
  await page.locator('label:has-text("Event type") select').selectOption('cumulative');
  await expect(page.locator('main h2').first()).toContainText('rare events over 3 days');
  await page.locator('label:has-text("Rarer than") input').blur();
  await shoot(page, 'rare-events', 0, { from: TABS });
});

test('statistics', async ({ page }) => {
  await page.getByRole('tab', { name: 'Statistics' }).click();
  // Sections 1 to 3, then section 4 after the bootstrap, as two images.
  await shoot(page, 'statistics', 4, { from: TABS, to: 'main .grid-2' });
  await page.getByRole('button', { name: 'Run the bootstrap' }).click();
  await expect(page.locator('main h3', { hasText: '1,000 resamples' })).toBeVisible();
  await shoot(page, 'statistics-bootstrap', 6,
              { from: 'main section.card:has(h2:text("4. How precise"))', to: 'main .plot' });
});

test('multi-ticker', async ({ page }) => {
  await page.getByRole('tab', { name: 'Multi-ticker' }).click();
  const synthetic = page.locator('fieldset', { hasText: 'Synthetic data (offline)' });
  await synthetic.getByRole('button', { name: 'All' }).click();
  await page.getByRole('button', { name: 'Compare 6 tickers' }).click();
  await expect(page.locator('main details')).toHaveCount(6);
  await shoot(page, 'multi-ticker', 0, { from: TABS });
});

test('dark theme', async ({ page }) => {
  await page.locator('label:has-text("Theme") select').selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await shoot(page, 'dark', 4, { fullPage: false });
});

test('phone', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await shoot(page, 'phone', 4, { fullPage: false });
});
