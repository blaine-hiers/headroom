/// <reference types="vitest/config" />
import { createHash } from 'node:crypto'
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { serviceWorkerSource } from './src/sw/serviceWorkerSource.ts'

function listFilesRecursive(dir: string, base: string = dir): string[] {
  let out: string[] = []
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry)
    out = statSync(full).isDirectory()
      ? out.concat(listFilesRecursive(full, base))
      : out.concat([path.relative(base, full).split(path.sep).join('/')])
  }
  return out
}

/**
 * Writes dist/sw.js after the production build, precaching every file the build actually
 * produced (JS/CSS chunks plus everything copied from public/ — icons, manifest, favicon).
 * The cache name is a hash of every file's own contents, so any change to any built file
 * (including a public/ asset whose filename doesn't change) rolls the cache version and the
 * old cache is dropped on activate. See src/main.tsx for the registration side, and
 * src/sw/serviceWorkerSource.ts for the worker itself.
 */
function serviceWorkerPlugin(): Plugin {
  let outDir = 'dist'
  let base = '/headroom/'
  return {
    name: 'headroom-service-worker',
    apply: 'build',
    configResolved(config) {
      outDir = config.build.outDir
      base = config.base
    },
    closeBundle() {
      const absOutDir = path.resolve(outDir)
      const files = listFilesRecursive(absOutDir).filter((f) => f !== 'sw.js').sort()
      const hash = createHash('sha1')
      for (const f of files) {
        hash.update(f)
        hash.update(readFileSync(path.join(absOutDir, f)))
      }
      const version = hash.digest('hex').slice(0, 12)
      const urls = files.map((f) => `${base}${f}`)
      const source = serviceWorkerSource({ version, base, urls })
      writeFileSync(path.join(absOutDir, 'sw.js'), source)
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), serviceWorkerPlugin()],
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
