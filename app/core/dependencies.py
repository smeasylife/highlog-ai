"""JWT Authentication Dependencies

This module provides JWT authentication functionality similar to highLog's JwtAuthenticationFilter.
It extracts and validates JWT tokens from the Authorization header.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends, status
from pydantic import BaseModel
import jwt
from jwt import PyJWTError
import os

logger = logging.getLogger(__name__)


# ========== JWT Configuration ==========
# These must match the values used in highLog
# JWT_SECRET is loaded from environment variable
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError(
        "JWT_SECRET environment variable is not set. "
        "Please set JWT_SECRET in your .env file to match highLog's JWT secret."
    )
JWT_ALGORITHM = "HS256"


# ========== Schemas ==========
class CurrentUser(BaseModel):
    """Authenticated user information extracted from JWT token"""
    user_id: int
    email: str
    role: str


# ========== JWT Token Validation ==========
def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload as dictionary

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error decoding JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def extract_token(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract Bearer token from Authorization header

    Args:
        authorization: Authorization header value

    Returns:
        Extracted token string

    Raises:
        HTTPException: If authorization header is missing or malformed
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: 'Bearer <token>'"
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is empty"
        )

    return token


async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> CurrentUser:
    """
    FastAPI dependency to get current authenticated user from JWT token

    This function should be used as a dependency in protected endpoints:
        @router.get("/protected")
        async def protected_endpoint(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.user_id, "email": user.email}

    Args:
        authorization: Authorization header value

    Returns:
        CurrentUser object with user information

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # 1. Extract token from Authorization header
        token = extract_token(authorization)

        # 2. Decode and validate token
        payload = decode_jwt_token(token)

        # 3. Extract user information from payload
        user_id = payload.get("sub")  # subject contains user_id
        email = payload.get("email")
        role = payload.get("role")

        if not user_id or not email or not role:
            logger.error("JWT token payload missing required fields")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload is invalid"
            )

        # Convert user_id to int (it's stored as string in JWT subject)
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid user_id in token: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID in token"
            )

        logger.debug(f"Authenticated user: {user_id} ({email}), role: {role}")

        return CurrentUser(
            user_id=user_id,
            email=email,
            role=role
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user_optional(
    authorization: Optional[str] = Header(None)
) -> Optional[CurrentUser]:
    """
    Optional JWT authentication dependency.

    Returns None if authentication fails instead of raising an exception.
    Useful for endpoints that have both authenticated and anonymous access.

    Args:
        authorization: Authorization header value

    Returns:
        CurrentUser object if authentication succeeds, None otherwise
    """
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


# ========== Role-based Authorization ==========


def require_role(*allowed_roles: str):
    """
    Factory function to create a dependency that requires specific roles

    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(
            user: CurrentUser = Depends(require_role("ADMIN", "SUPERADMIN"))
        ):
            return {"message": "Welcome admin"}

    Args:
        *allowed_roles: List of allowed role names

    Returns:
        Dependency function that checks user role
    """
    async def role_checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_roles:
            logger.warning(f"User {user.user_id} with role {user.role} attempted to access endpoint requiring roles: {allowed_roles}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: one of {allowed_roles}"
            )
        return user

    return role_checker
