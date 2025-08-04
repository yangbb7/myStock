/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/socket.io': {
        target: 'http://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunks - more granular splitting
          if (id.includes('node_modules')) {
            // Core React
            if (id.includes('react') || id.includes('react-dom')) {
              return 'react-vendor';
            }
            // UI Framework
            if (id.includes('antd') || id.includes('@ant-design')) {
              return 'antd-vendor';
            }
            // Charts and visualization
            if (id.includes('echarts') || id.includes('recharts') || id.includes('@antv')) {
              return 'charts-vendor';
            }
            // Network and data fetching
            if (id.includes('axios') || id.includes('socket.io') || id.includes('@tanstack/react-query')) {
              return 'network-vendor';
            }
            // State management
            if (id.includes('zustand') || id.includes('redux')) {
              return 'state-vendor';
            }
            // Utilities
            if (id.includes('lodash') || id.includes('dayjs') || id.includes('date-fns')) {
              return 'utils-vendor';
            }
            // Router
            if (id.includes('react-router')) {
              return 'router-vendor';
            }
            // Testing utilities (should not be in production)
            if (id.includes('@testing-library') || id.includes('vitest')) {
              return 'test-vendor';
            }
            // Other vendor libraries
            return 'vendor';
          }
          
          // Feature-based chunks with more granular splitting
          if (id.includes('/pages/Dashboard/') || id.includes('/components/Dashboard/')) {
            return 'dashboard';
          }
          if (id.includes('/pages/Strategy/') || id.includes('/components/Strategy/')) {
            return 'strategy';
          }
          if (id.includes('/pages/Data/') || id.includes('/components/Data/')) {
            return 'data';
          }
          if (id.includes('/pages/Orders/') || id.includes('/components/Orders/')) {
            return 'orders';
          }
          if (id.includes('/pages/Portfolio/') || id.includes('/components/Portfolio/')) {
            return 'portfolio';
          }
          if (id.includes('/pages/Risk/') || id.includes('/components/Risk/')) {
            return 'risk';
          }
          if (id.includes('/pages/Backtest/') || id.includes('/components/Backtest/')) {
            return 'backtest';
          }
          if (id.includes('/pages/System/') || id.includes('/components/System/')) {
            return 'system';
          }
          // Chart components as separate chunk
          if (id.includes('/components/Charts/')) {
            return 'charts-components';
          }
          // Common components
          if (id.includes('/components/Common/')) {
            return 'common-components';
          }
          // Forms and layout
          if (id.includes('/components/Forms/') || id.includes('/components/Layout/')) {
            return 'ui-components';
          }
          // Services and utilities
          if (id.includes('/services/') || id.includes('/utils/')) {
            return 'services-utils';
          }
          // Hooks
          if (id.includes('/hooks/')) {
            return 'hooks';
          }
        },
        chunkFileNames: () => {
          return `js/[name]-[hash].js`;
        },
        entryFileNames: 'js/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          if (!assetInfo.name) return `assets/[name]-[hash][extname]`;
          
          const info = assetInfo.name.split('.');
          const ext = info[info.length - 1];
          if (/\.(mp4|webm|ogg|mp3|wav|flac|aac)(\?.*)?$/i.test(assetInfo.name)) {
            return `media/[name]-[hash].${ext}`;
          }
          if (/\.(png|jpe?g|gif|svg)(\?.*)?$/i.test(assetInfo.name)) {
            return `img/[name]-[hash].${ext}`;
          }
          if (/\.(woff2?|eot|ttf|otf)(\?.*)?$/i.test(assetInfo.name)) {
            return `fonts/[name]-[hash].${ext}`;
          }
          return `assets/[name]-[hash].${ext}`;
        },
      },
      external: [],
    },
    chunkSizeWarningLimit: 1000,
    assetsInlineLimit: 4096,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
});
