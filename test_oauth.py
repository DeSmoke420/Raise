#!/usr/bin/env python3
"""
OAuth Configuration Test Script
Tests the OAuth setup and helps identify redirect issues.
"""

import os
import requests
import json
from urllib.parse import urlparse

def test_oauth_config():
    """Test OAuth configuration and identify potential issues."""
    print("🔍 Testing OAuth Configuration...")
    print("=" * 50)
    
    # Test 1: Check Supabase configuration
    print("\n1. Testing Supabase Configuration:")
    supabase_url = 'https://iayecqndmobjswtzoldb.supabase.co'
    supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlheWVjcW5kbW9ianN3dHpvbGRiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMjU1MjUsImV4cCI6MjA2NzgwMTUyNX0.FpndmbB-t9dvJsUFUX8l4VdLlbP4BZ1a425116UF10Q'
    
    try:
        # Test Supabase connectivity
        response = requests.get(f"{supabase_url}/auth/v1/settings", 
                              headers={'apikey': supabase_key})
        if response.status_code == 200:
            print("✅ Supabase connection successful")
        else:
            print(f"❌ Supabase connection failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Supabase connection error: {e}")
    
    # Test 2: Check deployment URL
    print("\n2. Testing Deployment URL:")
    deployment_url = 'https://raiseproject-production.up.railway.app'
    try:
        response = requests.get(deployment_url, timeout=10)
        if response.status_code == 200:
            print("✅ Deployment URL accessible")
        else:
            print(f"⚠️  Deployment URL returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Deployment URL error: {e}")
    
    # Test 3: Check OAuth callback URLs
    print("\n3. OAuth Callback URL Analysis:")
    callback_urls = [
        f"{supabase_url}/auth/v1/callback",
        f"{deployment_url}/auth/callback",
        f"{deployment_url}/api/auth/google/callback"
    ]
    
    for url in callback_urls:
        try:
            response = requests.head(url, timeout=5)
            print(f"✅ {url} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - Error: {e}")
    
    # Test 4: Check for localhost references
    print("\n4. Checking for Localhost References:")
    localhost_patterns = [
        'localhost',
        '127.0.0.1',
        ':3000',
        ':5000',
        ':8000'
    ]
    
    files_to_check = ['index.html', 'app.py', 'auth.py']
    for file in files_to_check:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in localhost_patterns:
                    if pattern in content:
                        print(f"⚠️  Found '{pattern}' in {file}")
    
    # Test 5: Environment Variables
    print("\n5. Environment Variables Check:")
    env_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'JWT_SECRET']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set")
    
    print("\n" + "=" * 50)
    print("🔧 RECOMMENDED FIXES:")
    print("1. Go to Google Cloud Console > APIs & Services > Credentials")
    print("2. Find your OAuth 2.0 Client ID")
    print("3. Remove any localhost redirect URIs")
    print("4. Add: https://iayecqndmobjswtzoldb.supabase.co/auth/v1/callback")
    print("5. Or add: https://raiseproject-production.up.railway.app/auth/callback")
    print("6. Save changes and test again")

if __name__ == "__main__":
    test_oauth_config() 