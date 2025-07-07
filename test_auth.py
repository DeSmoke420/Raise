#!/usr/bin/env python3
"""
Test script for Firebase Authentication setup
"""

import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_firebase_import():
    """Test if Firebase Admin SDK can be imported."""
    try:
        import firebase_admin
        from firebase_admin import credentials, auth
        logger.info("✅ Firebase Admin SDK imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Firebase Admin SDK import failed: {e}")
        return False

def test_firebase_config():
    """Test if Firebase configuration is available."""
    # Check for environment variable
    firebase_config = os.getenv('FIREBASE_CONFIG')
    if firebase_config:
        try:
            config_dict = json.loads(firebase_config)
            logger.info("✅ Firebase config found in environment variable")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in FIREBASE_CONFIG: {e}")
            return False
    
    # Check for config file
    config_files = ['firebase_config.json', 'firebase-config.json']
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                logger.info(f"✅ Firebase config found in {config_file}")
                return True
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"❌ Error reading {config_file}: {e}")
                return False
    
    logger.warning("⚠️  No Firebase configuration found")
    logger.info("   Set FIREBASE_CONFIG environment variable or create firebase_config.json")
    return False

def test_jwt_secret():
    """Test if JWT secret is configured."""
    jwt_secret = os.getenv('JWT_SECRET')
    if jwt_secret:
        if len(jwt_secret) >= 32:
            logger.info("✅ JWT secret found and appears secure")
            return True
        else:
            logger.warning("⚠️  JWT secret is too short (should be at least 32 characters)")
            return False
    else:
        logger.warning("⚠️  JWT_SECRET environment variable not set")
        logger.info("   Using default secret (not recommended for production)")
        return False

def test_auth_module():
    """Test if the auth module can be imported."""
    try:
        from auth import (
            initialize_firebase,
            create_jwt_token,
            verify_jwt_token,
            get_user_from_token
        )
        logger.info("✅ Auth module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Auth module import failed: {e}")
        return False

def test_firebase_initialization():
    """Test Firebase initialization (without actually connecting)."""
    try:
        from auth import initialize_firebase
        # This will fail gracefully if no config is found
        result = initialize_firebase()
        if result:
            logger.info("✅ Firebase initialized successfully")
        else:
            logger.warning("⚠️  Firebase initialization failed (no config)")
        return True
    except Exception as e:
        logger.error(f"❌ Firebase initialization error: {e}")
        return False

def main():
    """Run all authentication tests."""
    logger.info("🔍 Testing Firebase Authentication Setup")
    logger.info("=" * 50)
    
    tests = [
        ("Firebase Admin SDK Import", test_firebase_import),
        ("Firebase Configuration", test_firebase_config),
        ("JWT Secret Configuration", test_jwt_secret),
        ("Auth Module Import", test_auth_module),
        ("Firebase Initialization", test_firebase_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Authentication setup looks good.")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Please check the setup guide.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 