import os
import json
import logging
from functools import wraps
from typing import Optional, Dict, Any
from flask import request, jsonify, g
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://iayecqndmobjswtzoldb.supabase.co')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlheWVjcW5kbW9ianN3dHpvbGRiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMjU1MjUsImV4cCI6MjA2NzgwMTUyNX0.FpndmbB-t9dvJsUFUX8l4VdLlbP4BZ1a425116UF10Q')

# Initialize Supabase client
def initialize_supabase() -> Optional[Any]:
    """Initialize Supabase client."""
    try:
        # Try to import Supabase with error handling
        try:
            from supabase._sync.client import create_client  # type: ignore
        except ImportError as e:
            logger.error(f"Failed to import Supabase client: {e}")
            return None
        
        # Create client with proper configuration
        supabase = create_client(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_ANON_KEY
        )
        logger.info("Supabase client initialized successfully")
        return supabase
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        return None

# Global Supabase client
supabase_client = initialize_supabase()

# Simple local authentication for development when Supabase is not available
LOCAL_USERS = {
    "test@example.com": {
        "password": "password123",
        "user_id": "local_user_1",
        "email": "test@example.com",
        "name": "Test User"
    }
}

# JWT Secret for session management
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def create_jwt_token(user_id: str, email: str) -> str:
    """Create a JWT token for the user session."""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Get user information from JWT token."""
    payload = verify_jwt_token(token)
    if payload:
        return {
            'user_id': payload['user_id'],
            'email': payload['email']
        }
    return None

def verify_supabase_token(access_token: str) -> Optional[Dict[str, Any]]:
    """Verify Supabase access token and return user information."""
    try:
        if not supabase_client:
            logger.error("Supabase client not initialized")
            return None
        
        # Get user from Supabase using the access token
        user = supabase_client.auth.get_user(access_token)
        
        if user and user.user:
            return {
                'user_id': user.user.id,
                'email': user.user.email,
                'name': user.user.user_metadata.get('name', ''),
                'picture': user.user.user_metadata.get('avatar_url', '')
            }
        return None
    except Exception as e:
        logger.error(f"Supabase token verification failed: {e}")
        return None

def require_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for JWT token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            user = get_user_from_token(token)
            if user:
                g.current_user = user
                return f(*args, **kwargs)
        
        # Check for Supabase access token in request body or headers
        access_token = request.json.get('accessToken') if request.is_json and request.json else None
        if not access_token:
            access_token = request.headers.get('X-Supabase-Access-Token')
        
        if access_token:
            user = verify_supabase_token(access_token)
            if user:
                g.current_user = user
                return f(*args, **kwargs)
        
        return jsonify({'error': 'Authentication required'}), 401
    
    return decorated_function

def optional_auth(f):
    """Decorator for optional authentication - sets g.current_user if authenticated."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.current_user = None
        
        # Check for JWT token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            user = get_user_from_token(token)
            if user:
                g.current_user = user
        
        # Check for Supabase access token in request body or headers
        access_token = request.json.get('accessToken') if request.is_json and request.json else None
        if not access_token:
            access_token = request.headers.get('X-Supabase-Access-Token')
        
        if access_token and not g.current_user:
            user = verify_supabase_token(access_token)
            if user:
                g.current_user = user
        
        return f(*args, **kwargs)
    
    return decorated_function

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get the current authenticated user from Flask's g object."""
    return getattr(g, 'current_user', None)

def create_user_session(user_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a user session with JWT token."""
    jwt_token = create_jwt_token(user_info['user_id'], user_info['email'])
    return {
        'user': user_info,
        'token': jwt_token,
        'expires_in': JWT_EXPIRATION_HOURS * 3600  # seconds
    }

def sign_in_with_supabase(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Sign in user with Supabase email/password or local authentication."""
    try:
        # Try Supabase first if available
        if supabase_client:
            response = supabase_client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                return {
                    'user_id': response.user.id,
                    'email': response.user.email,
                    'name': response.user.user_metadata.get('name', ''),
                    'picture': response.user.user_metadata.get('avatar_url', ''),
                    'access_token': response.session.access_token if response.session else None
                }
        
        # Fallback to local authentication
        logger.info(f"Trying local authentication for {email}")
        logger.info(f"Available local users: {list(LOCAL_USERS.keys())}")
        if email in LOCAL_USERS:
            logger.info(f"User {email} found in LOCAL_USERS")
            if LOCAL_USERS[email]['password'] == password:
                user_info = LOCAL_USERS[email]
                logger.info(f"Local authentication successful for {email}")
                return {
                    'user_id': user_info['user_id'],
                    'email': user_info['email'],
                    'name': user_info['name'],
                    'picture': '',
                    'access_token': None
                }
            else:
                logger.warning(f"Password mismatch for {email}")
        else:
            logger.warning(f"User {email} not found in LOCAL_USERS")
        
        return None
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None

def sign_up_with_supabase(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Sign up user with Supabase email/password or local authentication."""
    try:
        # Try Supabase first if available
        if supabase_client:
            response = supabase_client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if response.user:
                return {
                    'user_id': response.user.id,
                    'email': response.user.email,
                    'name': response.user.user_metadata.get('name', ''),
                    'picture': response.user.user_metadata.get('avatar_url', ''),
                    'access_token': response.session.access_token if response.session else None
                }
        
        # Fallback to local authentication (simple registration)
        if email not in LOCAL_USERS:
            # In a real app, you'd hash the password and store it properly
            LOCAL_USERS[email] = {
                "password": password,
                "user_id": f"local_user_{len(LOCAL_USERS) + 1}",
                "email": email,
                "name": email.split('@')[0]
            }
            logger.info(f"Local user registered: {email}")
            return {
                'user_id': LOCAL_USERS[email]['user_id'],
                'email': email,
                'name': LOCAL_USERS[email]['name'],
                'picture': '',
                'access_token': None
            }
        
        return None
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return None 