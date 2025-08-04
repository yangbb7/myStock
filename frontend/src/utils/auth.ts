import type { AuthTokens } from '@/types/auth';

const TOKEN_KEY = 'myquant_auth_tokens';

export const tokenManager = {
  // 保存tokens到localStorage
  saveTokens: (tokens: AuthTokens): void => {
    localStorage.setItem(TOKEN_KEY, JSON.stringify(tokens));
  },

  // 获取tokens
  getTokens: (): AuthTokens | null => {
    const tokensStr = localStorage.getItem(TOKEN_KEY);
    if (!tokensStr) return null;
    
    try {
      return JSON.parse(tokensStr);
    } catch {
      return null;
    }
  },

  // 获取访问令牌
  getAccessToken: (): string | null => {
    const tokens = tokenManager.getTokens();
    return tokens?.access_token || null;
  },

  // 获取刷新令牌
  getRefreshToken: (): string | null => {
    const tokens = tokenManager.getTokens();
    return tokens?.refresh_token || null;
  },

  // 清除tokens
  clearTokens: (): void => {
    localStorage.removeItem(TOKEN_KEY);
  },

  // 更新访问令牌
  updateAccessToken: (accessToken: string): void => {
    const tokens = tokenManager.getTokens();
    if (tokens) {
      tokens.access_token = accessToken;
      tokenManager.saveTokens(tokens);
    }
  }
};

// JWT解码函数
export const decodeJWT = (token: string): any => {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
};

// 检查token是否过期
export const isTokenExpired = (token: string): boolean => {
  const decoded = decodeJWT(token);
  if (!decoded || !decoded.exp) return true;
  
  const currentTime = Date.now() / 1000;
  return decoded.exp < currentTime;
};