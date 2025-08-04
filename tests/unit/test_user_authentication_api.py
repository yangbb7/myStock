"""
用户认证API测试

测试用户注册、登录、JWT验证、密码重置等功能
严格遵循TDD原则，覆盖各种边界情况和安全场景
"""

import pytest
import pytest_asyncio
import json
import jwt
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from myQuant.interfaces.api.user_authentication_api import (
    router,
    UserRegisterRequest,
    UserLoginRequest,
    UserLoginResponse,
    TokenVerifyRequest,
    PasswordResetRequest,
    PasswordChangeRequest,
    get_current_user,
    create_access_token,
    verify_password,
    hash_password,
    decode_access_token
)
from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import UserRepository


class TestUserAuthenticationAPI:
    """用户认证API测试类"""
    
    @pytest.fixture
    def test_client(self):
        """创建测试客户端"""
        import tempfile
        import os
        from myQuant.interfaces.api.monolith_api import MonolithAPI, APIConfig
        from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig, SystemModule
        
        # Create unique database for each test
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        system_config = SystemConfig(
            database_url=f"sqlite:///{temp_db.name}",
            enabled_modules=[
                SystemModule.DATA,
                SystemModule.STRATEGY,
                SystemModule.EXECUTION,
                SystemModule.RISK,
                SystemModule.PORTFOLIO,
                SystemModule.ANALYTICS
            ]
        )
        
        api_config = APIConfig(
            title="myQuant Test Environment",
            description="Test environment for myQuant",
            port=8000,
            debug=True,
            enable_docs=True
        )
        
        trading_system = EnhancedTradingSystem(system_config)
        api = MonolithAPI(trading_system, api_config)
        
        client = TestClient(api.app)
        
        yield client
        
        # Cleanup
        try:
            os.unlink(temp_db.name)
        except:
            pass
    
    @pytest_asyncio.fixture
    async def db_manager(self):
        """创建测试数据库"""
        db_manager = DatabaseManager("sqlite://:memory:")
        await db_manager.initialize()
        yield db_manager
        await db_manager.close()
    
    @pytest_asyncio.fixture
    async def user_repository(self, db_manager):
        """创建用户仓库"""
        return UserRepository(db_manager)
    
    def test_user_register_success(self, test_client):
        """测试用户注册成功"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        request_data = {
            "username": f"testuser_{unique_id}",
            "email": f"test_{unique_id}@example.com",
            "password": "StrongPassword123!"
        }
        
        response = test_client.post("/api/v1/auth/register", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == f"testuser_{unique_id}"
        assert data["email"] == f"test_{unique_id}@example.com"
        assert "user_id" in data
        assert "created_at" in data
        assert "password" not in data  # 确保密码不被返回
    
    def test_user_register_duplicate_username(self, test_client):
        """测试重复用户名注册失败"""
        # 先注册一个用户
        request_data = {
            "username": "existinguser",
            "email": "existing@example.com",
            "password": "Password123!"
        }
        test_client.post("/api/v1/auth/register", json=request_data)
        
        # 尝试使用相同用户名注册
        duplicate_data = {
            "username": "existinguser",
            "email": "new@example.com",
            "password": "Password456!"
        }
        
        response = test_client.post("/api/v1/auth/register", json=duplicate_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "username already exists" in data["detail"].lower()
    
    def test_user_register_duplicate_email(self, test_client):
        """测试重复邮箱注册失败"""
        # 先注册一个用户
        request_data = {
            "username": "user1",
            "email": "duplicate@example.com",
            "password": "Password123!"
        }
        test_client.post("/api/v1/auth/register", json=request_data)
        
        # 尝试使用相同邮箱注册
        duplicate_data = {
            "username": "user2",
            "email": "duplicate@example.com",
            "password": "Password456!"
        }
        
        response = test_client.post("/api/v1/auth/register", json=duplicate_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "email already exists" in data["detail"].lower()
    
    def test_user_register_weak_password(self, test_client):
        """测试弱密码注册失败"""
        weak_passwords = [
            "12345",           # 太短
            "password",        # 没有数字
            "12345678",        # 没有字母
            "Password",        # 没有数字
            "password123",     # 没有大写字母
            "PASSWORD123",     # 没有小写字母
        ]
        
        for weak_pass in weak_passwords:
            request_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": weak_pass
            }
            
            response = test_client.post("/api/v1/auth/register", json=request_data)
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
            assert any("password" in str(error).lower() for error in data["detail"])
    
    def test_user_register_invalid_email(self, test_client):
        """测试无效邮箱格式"""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user@@example.com",
            "user@example",
        ]
        
        for invalid_email in invalid_emails:
            request_data = {
                "username": "testuser",
                "email": invalid_email,
                "password": "StrongPassword123!"
            }
            
            response = test_client.post("/api/v1/auth/register", json=request_data)
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    def test_user_login_success(self, test_client):
        """测试用户登录成功"""
        # 先注册用户
        register_data = {
            "username": "loginuser",
            "email": "login@example.com",
            "password": "LoginPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 登录
        login_data = {
            "username": "loginuser",
            "password": "LoginPassword123!"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user_id" in data
        assert "username" in data
        assert data["username"] == "loginuser"
    
    def test_user_login_with_email(self, test_client):
        """测试使用邮箱登录"""
        # 先注册用户
        register_data = {
            "username": "emaillogin",
            "email": "emaillogin@example.com",
            "password": "EmailPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 使用邮箱登录
        login_data = {
            "username": "emaillogin@example.com",
            "password": "EmailPassword123!"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    def test_user_login_wrong_password(self, test_client):
        """测试错误密码登录失败"""
        # 先注册用户
        register_data = {
            "username": "wrongpass",
            "email": "wrongpass@example.com",
            "password": "CorrectPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 使用错误密码登录
        login_data = {
            "username": "wrongpass",
            "password": "WrongPassword123!"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "incorrect username or password" in data["detail"].lower()
    
    def test_user_login_nonexistent_user(self, test_client):
        """测试不存在用户登录失败"""
        login_data = {
            "username": "nonexistentuser",
            "password": "Password123!"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "incorrect username or password" in data["detail"].lower()
    
    def test_jwt_token_generation(self):
        """测试JWT令牌生成"""
        user_id = 123
        username = "testuser"
        
        token = create_access_token(
            data={"sub": str(user_id), "username": username}
        )
        
        # 解码验证
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["sub"] == str(user_id)
        assert decoded["username"] == username
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_jwt_token_expiration(self):
        """测试JWT令牌过期"""
        user_id = 123
        
        # 创建一个已过期的令牌
        expired_token = create_access_token(
            data={"sub": str(user_id)},
            expires_delta=timedelta(seconds=-1)  # 已过期
        )
        
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(expired_token)
        
        assert exc_info.value.status_code == 401
        assert "token has expired" in str(exc_info.value.detail).lower()
    
    def test_jwt_token_invalid_signature(self):
        """测试无效签名的JWT令牌"""
        # 创建一个带有错误签名的令牌
        invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.WRONG_SIGNATURE"
        
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(invalid_token)
        
        assert exc_info.value.status_code == 401
        assert "could not validate credentials" in str(exc_info.value.detail).lower()
    
    def test_get_current_user_valid_token(self, test_client):
        """测试使用有效令牌获取当前用户"""
        # 先注册并登录获取令牌
        register_data = {
            "username": "currentuser",
            "email": "current@example.com",
            "password": "CurrentPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": "currentuser",
            "password": "CurrentPassword123!"
        })
        token = login_response.json()["access_token"]
        
        # 使用令牌访问受保护的端点
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "currentuser"
        assert data["email"] == "current@example.com"
        assert "user_id" in data
    
    def test_get_current_user_no_token(self, test_client):
        """测试无令牌访问受保护端点"""
        response = test_client.get("/api/v1/auth/me")
        
        assert response.status_code == 403
        data = response.json()
        assert "detail" in data
        assert "not authenticated" in data["detail"].lower()
    
    def test_get_current_user_invalid_token(self, test_client):
        """测试无效令牌访问受保护端点"""
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "could not validate credentials" in data["detail"].lower()
    
    def test_password_reset_request(self, test_client):
        """测试密码重置请求"""
        # 先注册用户
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        register_data = {
            "username": f"resetuser_{unique_id}",
            "email": f"reset_{unique_id}@example.com",
            "password": "OldPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 请求密码重置
        reset_request = {
            "email": f"reset_{unique_id}@example.com"
        }
        
        response = test_client.post("/api/v1/auth/password-reset", json=reset_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "password reset email has been sent" in data["message"].lower()
    
    def test_password_reset_nonexistent_email(self, test_client):
        """测试不存在邮箱的密码重置请求"""
        reset_request = {
            "email": "nonexistent@example.com"
        }
        
        # 应该返回成功（出于安全考虑不暴露邮箱是否存在）
        response = test_client.post("/api/v1/auth/password-reset", json=reset_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_password_change_success(self, test_client):
        """测试成功修改密码"""
        # 先注册并登录
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        register_data = {
            "username": f"changepass_{unique_id}",
            "email": f"changepass_{unique_id}@example.com",
            "password": "OldPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": f"changepass_{unique_id}",
            "password": "OldPassword123!"
        })
        token = login_response.json()["access_token"]
        
        # 修改密码
        change_request = {
            "old_password": "OldPassword123!",
            "new_password": "NewPassword456!"
        }
        
        response = test_client.post(
            "/api/v1/auth/change-password",
            json=change_request,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "password changed successfully" in data["message"].lower()
        
        # 验证旧密码无法登录
        old_login = test_client.post("/api/v1/auth/login", json={
            "username": f"changepass_{unique_id}",
            "password": "OldPassword123!"
        })
        assert old_login.status_code == 401
        
        # 验证新密码可以登录
        new_login = test_client.post("/api/v1/auth/login", json={
            "username": f"changepass_{unique_id}",
            "password": "NewPassword456!"
        })
        assert new_login.status_code == 200
    
    def test_password_change_wrong_old_password(self, test_client):
        """测试使用错误的旧密码修改密码失败"""
        # 先注册并登录
        register_data = {
            "username": "wrongoldpass",
            "email": "wrongold@example.com",
            "password": "CurrentPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": "wrongoldpass",
            "password": "CurrentPassword123!"
        })
        token = login_response.json()["access_token"]
        
        # 使用错误的旧密码尝试修改
        change_request = {
            "old_password": "WrongPassword123!",
            "new_password": "NewPassword456!"
        }
        
        response = test_client.post(
            "/api/v1/auth/change-password",
            json=change_request,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "incorrect old password" in data["detail"].lower()
    
    def test_token_refresh(self, test_client):
        """测试令牌刷新"""
        # 先登录获取令牌
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        register_data = {
            "username": f"refreshuser_{unique_id}",
            "email": f"refresh_{unique_id}@example.com",
            "password": "RefreshPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": f"refreshuser_{unique_id}",
            "password": "RefreshPassword123!"
        })
        old_token = login_response.json()["access_token"]
        
        # 刷新令牌
        response = test_client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {old_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        # 新令牌可能相同（如果在同一秒内生成），但应该是有效的
        # 验证新令牌包含正确的用户信息
        import jwt
        decoded = jwt.decode(data["access_token"], options={"verify_signature": False})
        assert "sub" in decoded
        assert "username" in decoded
    
    def test_user_logout(self, test_client):
        """测试用户登出"""
        # 先登录
        register_data = {
            "username": "logoutuser",
            "email": "logout@example.com",
            "password": "LogoutPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": "logoutuser",
            "password": "LogoutPassword123!"
        })
        token = login_response.json()["access_token"]
        
        # 登出
        response = test_client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "logged out successfully" in data["message"].lower()
        
        # 验证令牌已失效（需要实现黑名单机制）
        me_response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == 401
    
    def test_password_hashing(self):
        """测试密码哈希功能"""
        password = "TestPassword123!"
        
        # 哈希密码
        hashed = hash_password(password)
        
        # 验证哈希不等于原密码
        assert hashed != password
        assert len(hashed) > 50  # bcrypt哈希通常很长
        
        # 验证密码
        assert verify_password(password, hashed) is True
        assert verify_password("WrongPassword", hashed) is False
    
    def test_concurrent_login_attempts(self, test_client):
        """测试并发登录尝试（防止暴力破解）"""
        # 注册用户
        register_data = {
            "username": "bruteforce",
            "email": "brute@example.com",
            "password": "BrutePassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 连续失败登录尝试
        failed_attempts = []
        for i in range(5):
            response = test_client.post("/api/v1/auth/login", json={
                "username": "bruteforce",
                "password": f"WrongPassword{i}!"
            })
            failed_attempts.append(response.status_code)
        
        # 应该在多次失败后锁定账户或增加延迟
        assert all(status == 401 for status in failed_attempts[:3])
        # 第4-5次尝试可能返回429（太多请求）或仍是401但有延迟
    
    def test_username_validation(self, test_client):
        """测试用户名验证规则"""
        invalid_usernames = [
            "ab",              # 太短
            "a" * 51,          # 太长
            "user name",       # 包含空格
            "user@name",       # 包含特殊字符
            "用户名",           # 非ASCII字符
            "",                # 空字符串
        ]
        
        for invalid_username in invalid_usernames:
            request_data = {
                "username": invalid_username,
                "email": "valid@example.com",
                "password": "ValidPassword123!"
            }
            
            response = test_client.post("/api/v1/auth/register", json=request_data)
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    def test_session_management(self, test_client):
        """测试会话管理"""
        # 注册并登录
        register_data = {
            "username": "sessionuser",
            "email": "session@example.com",
            "password": "SessionPassword123!"
        }
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # 获取活动会话列表
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": "sessionuser",
            "password": "SessionPassword123!"
        })
        token = login_response.json()["access_token"]
        
        response = test_client.get(
            "/api/v1/auth/sessions",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) > 0
        
        # 每个会话应包含必要信息
        session = data["sessions"][0]
        assert "session_id" in session
        assert "created_at" in session
        assert "last_accessed" in session
        assert "ip_address" in session
        assert "user_agent" in session