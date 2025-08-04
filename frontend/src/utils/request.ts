import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import { message } from 'antd';
import { tokenManager, isTokenExpired } from './auth';
import { authApi } from '@/api/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// 创建axios实例
const request = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 正在刷新token的标志
let isRefreshing = false;
// 等待token刷新的请求队列
let refreshSubscribers: ((token: string) => void)[] = [];

// 通知所有等待的请求
const notifyRefreshSubscribers = (token: string) => {
  refreshSubscribers.forEach(callback => callback(token));
  refreshSubscribers = [];
};

// 添加请求到等待队列
const addRefreshSubscriber = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

// 请求拦截器
request.interceptors.request.use(
  (config) => {
    console.log('[Request] Intercepting request:', config.url);
    
    // 公共API endpoints不需要认证
    const publicEndpoints = ['/health', '/data/realtime', '/data/market', '/data/symbols'];
    const isPublicEndpoint = publicEndpoints.some(endpoint => config.url?.includes(endpoint));
    
    if (isPublicEndpoint) {
      console.log('[Request] Public endpoint - skipping auth:', config.url);
      return config;
    }
    
    const token = tokenManager.getAccessToken();
    
    if (token && !config.url?.includes('/auth/refresh')) {
      // 检查token是否过期
      if (isTokenExpired(token)) {
        // Token过期，需要刷新
        return new Promise((resolve) => {
          addRefreshSubscriber((newToken: string) => {
            config.headers.Authorization = `Bearer ${newToken}`;
            resolve(config);
          });
        });
      }
      
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
request.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log('[Request] Response received:', {
      url: response.config.url,
      status: response.status,
      data: response.data
    });
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    // 处理401错误（未授权）
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (originalRequest.url?.includes('/auth/refresh')) {
        // 刷新token也失败了，需要重新登录
        tokenManager.clearTokens();
        window.location.href = '/login';
        return Promise.reject(error);
      }
      
      originalRequest._retry = true;
      
      if (!isRefreshing) {
        isRefreshing = true;
        
        try {
          const refreshToken = tokenManager.getRefreshToken();
          if (!refreshToken) {
            throw new Error('No refresh token');
          }
          
          const response = await authApi.refreshToken(refreshToken);
          tokenManager.saveTokens({
            access_token: response.access_token,
            refresh_token: response.refresh_token,
            token_type: response.token_type
          });
          
          notifyRefreshSubscribers(response.access_token);
          isRefreshing = false;
          
          // 重试原始请求
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${response.access_token}`;
          }
          return request(originalRequest);
        } catch (refreshError) {
          isRefreshing = false;
          tokenManager.clearTokens();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      }
      
      // 等待token刷新完成
      return new Promise((resolve) => {
        addRefreshSubscriber((token: string) => {
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${token}`;
          }
          resolve(request(originalRequest));
        });
      });
    }
    
    // 处理403错误（权限不足）
    if (error.response?.status === 403) {
      message.error('您没有权限执行此操作');
    } else if (error.response?.status === 500) {
      message.error('服务器错误，请稍后重试');
    } else if (!error.response) {
      message.error('网络错误，请检查网络连接');
    }
    
    return Promise.reject(error);
  }
);

export default request;