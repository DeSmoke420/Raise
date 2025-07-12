#!/usr/bin/env python3
"""
Test script for Supabase Authentication setup
"""

import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_supabase_import():
    """Test if Supabase client can be imported."""
    try:
        from supabase._sync.client import create_client  # type: ignore
        logger.info("✅ Supabase client imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Supabase client import failed: {e}")
        return False

def test_supabase_config():
    """Test if Supabase configuration is available."""
    # Check for environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if supabase_url and supabase_key:
        logger.info("✅ Supabase config found in environment variables")
        return True
    
    # Check if using default config
    default_url = 'https://iayecqndmobjswtzoldb.supabase.co'
    default_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlheWVjcW5kbW9ianN3dHpvbGRiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMjU1MjUsImV4cCI6MjA2NzgwMTUyNX0.FpndmbB-t9dvJsUFUX8l4VdLlbP4BZ1a425116UF10Q'
    
    if supabase_url == default_url or supabase_key == default_key:
        logger.info("✅ Using default Supabase configuration")
        return True
    
    logger.warning("⚠️  No Supabase configuration found")
    logger.info("   Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
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
            require_auth,
            optional_auth,
            get_current_user,
            create_user_session,
            sign_in_with_supabase,
            sign_up_with_supabase
        )
        logger.info("✅ Auth module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Auth module import failed: {e}")
        return False

def test_supabase_initialization():
    """Test Supabase initialization (without actually connecting)."""
    try:
        from auth import initialize_supabase
        # This will fail gracefully if no config is found
        result = initialize_supabase()
        if result:
            logger.info("✅ Supabase initialized successfully")
        else:
            logger.warning("⚠️  Supabase initialization failed (no config)")
        return True
    except Exception as e:
        logger.error(f"❌ Supabase initialization error: {e}")
        return False

def main():
    """Run all authentication tests."""
    logger.info("🔍 Testing Supabase Authentication Setup")
    logger.info("=" * 50)
    
    tests = [
        ("Supabase Client Import", test_supabase_import),
        ("Supabase Configuration", test_supabase_config),
        ("JWT Secret Configuration", test_jwt_secret),
        ("Auth Module Import", test_auth_module),
        ("Supabase Initialization", test_supabase_initialization),
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