/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served from https://blaine-hiers.github.io/headroom/
  base: process.env.VITE_BASE ?? '/headroom/',
  test: {
    projects: [
      // lib/ is framework-free: plain node.
      { extends: true, test: { name: 'lib', include: ['src/**/*.test.ts'], environment: 'node' } },
      // UI tests need a DOM.
      {
        extends: true,
        test: {
          name: 'ui',
          include: ['src/**/*.test.tsx'],
          environment: 'jsdom',
          setupFiles: ['src/ui/test-setup.ts'],
        },
      },
    ],
  },
})
