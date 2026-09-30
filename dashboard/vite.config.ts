import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// In development the API runs separately (uvicorn on :8000); Vite forwards /api to it, so the
// browser sees one origin and no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 5000 },
  // Unit tests only; the browser tests in e2e/ run under Playwright (npm run e2e).
  test: { include: ['src/**/*.test.ts'] },
});
