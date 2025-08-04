import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';

// Initialize performance monitoring as early as possible
if (typeof window !== 'undefined') {
  // Mark app start time
  performance.mark('app-start');
  
  // Setup performance observer for monitoring
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.entryType === 'measure' || entry.entryType === 'navigation') {
          console.log(`Performance: ${entry.name} took ${entry.duration}ms`);
        }
      });
    });
    
    observer.observe({ entryTypes: ['measure', 'navigation'] });
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

// Mark app initialization complete
if (typeof window !== 'undefined') {
  performance.mark('app-initialized');
  performance.measure('app-initialization', 'app-start', 'app-initialized');
}
