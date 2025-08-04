import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface SystemStatus {
  isRunning: boolean;
  uptime: number;
  modules: Record<string, any>;
}

interface SystemStore {
  theme: 'light' | 'dark';
  systemStatus: SystemStatus | null;
  setTheme: (theme: 'light' | 'dark') => void;
  setSystemStatus: (status: SystemStatus) => void;
}

export const useSystemStore = create<SystemStore>()(
  persist(
    (set) => ({
      theme: 'light',
      systemStatus: null,
      setTheme: (theme) => set({ theme }),
      setSystemStatus: (systemStatus) => set({ systemStatus }),
    }),
    {
      name: 'system-store',
      partialize: (state) => ({ theme: state.theme }),
    }
  )
);