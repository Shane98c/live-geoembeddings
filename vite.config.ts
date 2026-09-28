import { defineConfig } from 'vite'

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site under /live-geoembeddings/.
  base: command === 'build' ? '/live-geoembeddings/' : '/',
  server: {
    watch: { ignored: ['**/.venv/**'] },
  },
}))
