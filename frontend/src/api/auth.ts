import axios from 'axios';
import request from '@/utils/request';
import type { 
  LoginCredentials, 
  RegisterData, 
  AuthResponse, 
  User, 
  PasswordResetRequest, 
  PasswordResetConfirm,
  PasswordChange 
} from '@/types/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const authApi = {
  // 用户注册
  register: async (data: RegisterData): Promise<AuthResponse> => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/register`, data);
    return response.data;
  },

  // 用户登录
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/login`, credentials);
    return response.data;
  },

  // 用户登出
  logout: async (refreshToken: string): Promise<void> => {
    await request.post('/api/v1/auth/logout', { refresh_token: refreshToken });
  },

  // 刷新访问令牌
  refreshToken: async (refreshToken: string): Promise<AuthResponse> => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/refresh`, { 
      refresh_token: refreshToken 
    });
    return response.data;
  },

  // 获取当前用户信息
  getCurrentUser: async (): Promise<User> => {
    const response = await request.get('/api/v1/auth/me');
    return response.data;
  },

  // 更新用户信息
  updateProfile: async (data: Partial<User>): Promise<User> => {
    const response = await request.put('/api/v1/auth/me', data);
    return response.data;
  },

  // 请求密码重置
  requestPasswordReset: async (data: PasswordResetRequest): Promise<{ message: string }> => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/password-reset`, data);
    return response.data;
  },

  // 确认密码重置
  confirmPasswordReset: async (data: PasswordResetConfirm): Promise<{ message: string }> => {
    const response = await axios.post(`${API_BASE_URL}/api/v1/auth/password-reset/confirm`, data);
    return response.data;
  },

  // 修改密码
  changePassword: async (data: PasswordChange): Promise<{ message: string }> => {
    const response = await request.post('/api/v1/auth/change-password', data);
    return response.data;
  },

  // 获取用户会话列表
  getSessions: async (): Promise<any[]> => {
    const response = await request.get('/api/v1/auth/sessions');
    return response.data;
  },

  // 撤销特定会话
  revokeSession: async (sessionId: string): Promise<void> => {
    await request.delete(`/api/v1/auth/sessions/${sessionId}`);
  }
};