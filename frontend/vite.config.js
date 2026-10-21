import { defineConfig } from 'vite';

export default defineConfig({
  server: { proxy: { '/api': 'http://localhost:8000' } },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-recharts': ['recharts'],
          'vendor-icons': ['lucide-react']
        }
      }
    }
  }
});
