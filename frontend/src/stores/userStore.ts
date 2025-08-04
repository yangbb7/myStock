import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, AuthTokens, LoginCredentials, RegisterData } from '@/types/auth';
import { authApi } from '@/api/auth';
import { tokenManager } from '@/utils/auth';
import { message } from 'antd';

interface UserStore {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  setUser: (user: User) => void;
  clearError: () => void;
  checkAuth: () => Promise<void>;
}

export const useUserStore = create<UserStore>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      loading: false,
      error: null,

      login: async (credentials) => {
        set({ loading: true, error: null });
        try {
          const response = await authApi.login(credentials);
          
          // 保存tokens
          tokenManager.saveTokens({
            access_token: response.access_token,
            refresh_token: response.refresh_token,
            token_type: response.token_type
          });
          
          // 设置用户信息
          set({ 
            user: response.user, 
            isAuthenticated: true, 
            loading: false 
          });
          
          message.success('登录成功');
        } catch (error: any) {
          set({ 
            loading: false, 
            error: error.response?.data?.detail || '登录失败' 
          });
          message.error(error.response?.data?.detail || '登录失败');
          throw error;
        }
      },

      register: async (data) => {
        set({ loading: true, error: null });
        try {
          const response = await authApi.register(data);
          
          // 保存tokens
          tokenManager.saveTokens({
            access_token: response.access_token,
            refresh_token: response.refresh_token,
            token_type: response.token_type
          });
          
          // 设置用户信息
          set({ 
            user: response.user, 
            isAuthenticated: true, 
            loading: false 
          });
          
          message.success('注册成功');
        } catch (error: any) {
          set({ 
            loading: false, 
            error: error.response?.data?.detail || '注册失败' 
          });
          message.error(error.response?.data?.detail || '注册失败');
          throw error;
        }
      },

      logout: async () => {
        set({ loading: true });
        try {
          const refreshToken = tokenManager.getRefreshToken();
          if (refreshToken) {
            await authApi.logout(refreshToken);
          }
        } catch (error) {
          console.error('Logout error:', error);
        } finally {
          // 清除本地数据
          tokenManager.clearTokens();
          set({ 
            user: null, 
            isAuthenticated: false, 
            loading: false 
          });
          message.success('已退出登录');
        }
      },

      refreshUser: async () => {
        try {
          const user = await authApi.getCurrentUser();
          set({ user, isAuthenticated: true });
        } catch (error) {
          console.error('Failed to refresh user:', error);
          set({ user: null, isAuthenticated: false });
        }
      },

      setUser: (user) => {
        set({ user, isAuthenticated: true });
      },

      clearError: () => {
        set({ error: null });
      },

      checkAuth: async () => {
        const token = tokenManager.getAccessToken();
        if (!token) {
          set({ user: null, isAuthenticated: false });
          return;
        }

        try {
          const user = await authApi.getCurrentUser();
          set({ user, isAuthenticated: true });
        } catch (error) {
          console.error('Auth check failed:', error);
          tokenManager.clearTokens();
          set({ user: null, isAuthenticated: false });
        }
      }
    }),
    {
      name: 'user-store',
      partialize: (state) => ({ 
        // 只持久化用户信息，不持久化loading和error状态
        user: state.user,
        isAuthenticated: state.isAuthenticated
      }),
    }
  )
);