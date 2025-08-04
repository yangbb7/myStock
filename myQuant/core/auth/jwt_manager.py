# -*- coding: utf-8 -*-
"""
JWT Token Management for myQuant platform
"""

import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Default secret key (should be overridden by environment variable)
DEFAULT_SECRET_KEY = "myQuant_default_secret_key_change_in_production"

def get_secret_key() -> str:
    """Get JWT secret key from environment or use default"""
    return os.getenv('JWT_SECRET_KEY', DEFAULT_SECRET_KEY)

def create_token(user_data: Dict[str, Any], expires_in_hours: int = 24) -> str:
    """
    Create a JWT token
    
    Args:
        user_data: User information to encode in token
        expires_in_hours: Token expiration time in hours
        
    Returns:
        JWT token string
    """
    payload = {
        'user_data': user_data,
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow(),
    }
    
    return jwt.encode(payload, get_secret_key(), algorithm='HS256')

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded user data if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=['HS256'])
        return payload.get('user_data')
    except jwt.ExpiredSignatureError:
        # Token has expired
        return None
    except jwt.InvalidTokenError:
        # Token is invalid
        return None

def refresh_token(token: str, new_expires_in_hours: int = 24) -> Optional[str]:
    """
    Refresh an existing token with new expiration
    
    Args:
        token: Current JWT token
        new_expires_in_hours: New expiration time in hours
        
    Returns:
        New JWT token if current token is valid, None otherwise
    """
    user_data = verify_token(token)
    if user_data:
        return create_token(user_data, new_expires_in_hours)
    return None