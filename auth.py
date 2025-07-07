import os
import json
import logging
from functools import wraps
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth
from flask import request, jsonify, g
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials."""
    try:
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            # Try to load from environment variable first
            firebase_config = os.getenv('FIREBASE_CONFIG')
            if firebase_config:
                cred_dict = json.loads(firebase_config)
                cred = credentials.Certificate(cred_dict)
            else:
                # Try to load from file
                config_path = 'firebase_config.json'
                if os.path.exists(config_path):
                    cred = credentials.Certificate(config_path)
                else:
                    logger.warning("Firebase config not found. Authentication will be disabled.")
                    return False
            
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False

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

def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verify Firebase ID token and return user information."""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return {
            'user_id': decoded_token['uid'],
            'email': decoded_token.get('email', ''),
            'name': decoded_token.get('name', ''),
            'picture': decoded_token.get('picture', '')
        }
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
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
        
        # Check for Firebase ID token in request body or headers
        id_token = request.json.get('idToken') if request.is_json else None
        if not id_token:
            id_token = request.headers.get('X-Firebase-ID-Token')
        
        if id_token:
            user = verify_firebase_token(id_token)
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
        
        # Check for Firebase ID token in request body or headers
        id_token = request.json.get('idToken') if request.is_json else None
        if not id_token:
            id_token = request.headers.get('X-Firebase-ID-Token')
        
        if id_token and not g.current_user:
            user = verify_firebase_token(id_token)
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