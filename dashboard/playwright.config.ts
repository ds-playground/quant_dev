import { defineConfig, devices } from '@playwright/test';

// The end-to-end check: the built dashboard served by `python -m src.api`, driven in Chromium.
// Run with `npm run e2e` (builds first). PYTHON picks the interpreter (default: `python`).
// The first time on a new machine: `npx playwright install chromium`.
const PORT = 8765;

export default defineConfig({
  testDir: 'e2e',
  timeout: 120_000,
  expect: { timeout: 60_000 },
  workers: 1,
  reporter: [['list']],
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 900 } } }],
  webServer: {
    command: `${process.env.PYTHON ?? 'python'} -m src.api --port ${PORT}`,
    cwd: '..',
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
