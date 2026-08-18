import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Helper to parse comma-separated allowed hosts or default to 'all' / true
const parseAllowedHosts = (envVal) => {
  if (!envVal) return true; // Default to allowing all hosts behind reverse proxy
  if (envVal === 'all' || envVal === 'true') return true;
  return envVal.split(',').map((h) => h.trim());
};

export default defineConfig({
  plugins: [react()],
  // Production / Container Preview Configuration (`vite preview`)
  preview: {
    allowedHosts: parseAllowedHosts(process.env.ALLOWED_HOSTS),
    host: process.env.HOST || '0.0.0.0',
    port: parseInt(process.env.PORT, 10) || 8787,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BACKEND || 'http://127.0.0.1:8787',
        changeOrigin: true
      }
    }
  },
  // Development Server Configuration (`vite`)
  server: {
    allowedHosts: parseAllowedHosts(process.env.ALLOWED_HOSTS),
    host: process.env.HOST || '0.0.0.0',
    port: parseInt(process.env.DEV_PORT, 10) || 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BACKEND || 'http://127.0.0.1:8787',
        changeOrigin: true,
      },
      '/health': process.env.VITE_API_BACKEND || 'http://127.0.0.1:8787'
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test-setup.js',
    globals: true,
    css: true
  }
});
