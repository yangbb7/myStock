"""
用户认证API

提供用户注册、登录、JWT令牌管理、密码重置等功能
"""

import re
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
import jwt

from myQuant.infrastructure.database.database_manager import DatabaseManager
from myQuant.infrastructure.database.repositories import UserRepository
from myQuant.infrastructure.container import get_container


router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
SECRET_KEY = "your-secret-key-here"  # 生产环境应从环境变量读取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer认证
security = HTTPBearer()

# 用于存储已登出的令牌（生产环境应使用Redis）
blacklisted_tokens = set()

# 用于跟踪登录失败次数（生产环境应使用Redis）
login_attempts = {}


async def ensure_database_initialized():
    """确保数据库已初始化"""
    container = get_container()
    db_manager = container.db_manager()
    
    if not db_manager.is_connected():
        await db_manager.initialize()
    
    return db_manager


class UserRegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        """验证密码强度"""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserRegisterResponse(BaseModel):
    """用户注册响应"""
    user_id: int
    username: str
    email: str
    created_at: datetime


class UserLoginRequest(BaseModel):
    """用户登录请求"""
    username: str  # 可以是用户名或邮箱
    password: str


class UserLoginResponse(BaseModel):
    """用户登录响应"""
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class TokenVerifyRequest(BaseModel):
    """令牌验证请求"""
    token: str


class PasswordResetRequest(BaseModel):
    """密码重置请求"""
    email: EmailStr


class PasswordChangeRequest(BaseModel):
    """密码修改请求"""
    old_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_password(cls, v):
        """验证密码强度"""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserInfo(BaseModel):
    """用户信息"""
    user_id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str


def hash_password(password: str) -> str:
    """哈希密码"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    """解码访问令牌"""
    try:
        # 检查令牌是否在黑名单中
        if token in blacklisted_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户"""
    token = credentials.credentials
    payload = decode_access_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    # 获取用户信息
    db_manager = await ensure_database_initialized()
    user_repo = UserRepository(db_manager)
    user = await user_repo.get_by_id(int(user_id))
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


def check_login_attempts(username: str):
    """检查登录尝试次数"""
    if username in login_attempts:
        attempts_info = login_attempts[username]
        if attempts_info["count"] >= 5:
            # 检查是否在锁定期内
            if datetime.utcnow() < attempts_info["locked_until"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed login attempts. Please try again later."
                )
            else:
                # 重置计数
                login_attempts[username] = {"count": 0, "locked_until": None}


def record_failed_login(username: str):
    """记录失败的登录尝试"""
    if username not in login_attempts:
        login_attempts[username] = {"count": 0, "locked_until": None}
    
    login_attempts[username]["count"] += 1
    
    if login_attempts[username]["count"] >= 5:
        # 锁定15分钟
        login_attempts[username]["locked_until"] = datetime.utcnow() + timedelta(minutes=15)


@router.post("/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest):
    """用户注册"""
    db_manager = await ensure_database_initialized()
    user_repo = UserRepository(db_manager)
    
    # 检查用户名是否已存在
    existing_user = await user_repo.get_by_username(request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # 检查邮箱是否已存在
    existing_email = await user_repo.get_by_email(request.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    # 创建用户
    hashed_password = hash_password(request.password)
    user_id = await user_repo.create(
        username=request.username,
        email=request.email,
        password_hash=hashed_password
    )
    
    # 获取创建的用户信息
    user = await user_repo.get_by_id(user_id)
    
    return UserRegisterResponse(
        user_id=user_id,
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"]
    )


@router.post("/login", response_model=UserLoginResponse)
async def login(request: UserLoginRequest):
    """用户登录"""
    # 检查登录尝试次数
    check_login_attempts(request.username)
    
    db_manager = await ensure_database_initialized()
    user_repo = UserRepository(db_manager)
    
    # 支持用户名或邮箱登录
    if "@" in request.username:
        user = await user_repo.get_by_email(request.username)
    else:
        user = await user_repo.get_by_username(request.username)
    
    if not user or not verify_password(request.password, user["password_hash"]):
        record_failed_login(request.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # 检查用户是否激活
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # 清除失败登录记录
    if request.username in login_attempts:
        del login_attempts[request.username]
    
    # 创建访问令牌
    access_token = create_access_token(
        data={"sub": str(user["id"]), "username": user["username"]}
    )
    
    return UserLoginResponse(
        access_token=access_token,
        user_id=user["id"],
        username=user["username"]
    )


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserInfo(
        user_id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        is_active=current_user.get("is_active", True),
        created_at=current_user["created_at"]
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """用户登出"""
    # 将令牌加入黑名单
    blacklisted_tokens.add(credentials.credentials)
    
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=UserLoginResponse)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """刷新访问令牌"""
    # 创建新的访问令牌
    access_token = create_access_token(
        data={"sub": str(current_user["id"]), "username": current_user["username"]}
    )
    
    return UserLoginResponse(
        access_token=access_token,
        user_id=current_user["id"],
        username=current_user["username"]
    )


@router.post("/password-reset")
async def request_password_reset(request: PasswordResetRequest):
    """请求密码重置"""
    db_manager = await ensure_database_initialized()
    user_repo = UserRepository(db_manager)
    
    # 查找用户（出于安全考虑，无论用户是否存在都返回相同响应）
    user = await user_repo.get_by_email(request.email)
    
    if user:
        # 生成重置令牌
        reset_token = secrets.token_urlsafe(32)
        
        # TODO: 存储重置令牌并发送邮件
        # 这里应该实现：
        # 1. 将reset_token存储到数据库，关联到用户
        # 2. 发送包含重置链接的邮件
        pass
    
    return {"message": "If the email exists, a password reset email has been sent"}


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user)
):
    """修改密码"""
    db_manager = await ensure_database_initialized()
    user_repo = UserRepository(db_manager)
    
    # 验证旧密码
    if not verify_password(request.old_password, current_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect old password"
        )
    
    # 更新密码
    hashed_password = hash_password(request.new_password)
    await user_repo.update_password(current_user["id"], hashed_password)
    
    return {"message": "Password changed successfully"}


@router.get("/sessions", response_model=Dict[str, List[SessionInfo]])
async def get_sessions(
    current_user: dict = Depends(get_current_user),
    user_agent: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None),
    x_real_ip: Optional[str] = Header(None)
):
    """获取用户的活动会话"""
    # TODO: 实现真实的会话管理
    # 这里返回模拟数据
    
    # 获取客户端IP
    ip_address = x_forwarded_for or x_real_ip or "127.0.0.1"
    if "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
    
    sessions = [
        SessionInfo(
            session_id=secrets.token_hex(16),
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent or "Unknown"
        )
    ]
    
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """撤销指定会话"""
    # TODO: 实现真实的会话撤销
    return {"message": f"Session {session_id} has been revoked"}