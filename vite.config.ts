import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served from https://blaine-hiers.github.io/headroom/
  base: process.env.VITE_BASE ?? '/headroom/',
})
