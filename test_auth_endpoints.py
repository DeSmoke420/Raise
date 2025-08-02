#!/usr/bin/env python3
"""
Test script to verify authentication endpoints are working
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:5000"

def test_auth_endpoints():
    print("🔍 Testing Authentication Endpoints")
    print("=" * 50)
    
    # Test 1: Sign up with a new user
    print("\n📋 Test 1: User Registration")
    signup_data = {
        "email": "testuser@example.com",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/signup", 
                               json=signup_data,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registration successful: {result.get('user', {}).get('email')}")
            token = result.get('token')
        else:
            print(f"❌ Registration failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return
    
    # Test 2: Sign in with the same user
    print("\n📋 Test 2: User Sign In")
    signin_data = {
        "email": "testuser@example.com",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/signin", 
                               json=signin_data,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sign in successful: {result.get('user', {}).get('email')}")
            token = result.get('token')
        else:
            print(f"❌ Sign in failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Sign in error: {e}")
        return
    
    # Test 3: Get user info with token
    print("\n📋 Test 3: Get User Info")
    try:
        response = requests.get(f"{BASE_URL}/api/auth/user",
                              headers={'Authorization': f'Bearer {token}'})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ User info retrieved: {result}")
        else:
            print(f"❌ Get user info failed: {response.text}")
    except Exception as e:
        print(f"❌ Get user info error: {e}")
    
    # Test 4: Verify token
    print("\n📋 Test 4: Verify Token")
    try:
        response = requests.post(f"{BASE_URL}/api/auth/verify",
                               json={"token": token},
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Token verification successful: {result}")
        else:
            print(f"❌ Token verification failed: {response.text}")
    except Exception as e:
        print(f"❌ Token verification error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Authentication endpoints test completed!")

if __name__ == "__main__":
    test_auth_endpoints() 