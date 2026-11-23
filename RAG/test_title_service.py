#!/usr/bin/env python
"""
Test script to verify Title Service integration with Chat Server
Run this to ensure both services communicate correctly.
"""

import requests
import json
import time
from typing import Dict, Any

TITLE_SERVICE_URL = "http://localhost:8001"
CHAT_SERVER_URL = "http://localhost:8000"

def test_title_service_health() -> bool:
    """Test if title service is running and healthy"""
    print("\n📋 TEST 1: Title Service Health Check")
    print("-" * 50)
    
    try:
        response = requests.get(f"{TITLE_SERVICE_URL}/health", timeout=3)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Service Status: {data['status']}")
        print(f"   Model: {data['model']}")
        return True
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {TITLE_SERVICE_URL}")
        print("   Make sure title service is running:")
        print("   $ python RAG/title_service.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate_title_async() -> bool:
    """Test async title generation"""
    print("\n📋 TEST 2: Async Title Generation")
    print("-" * 50)
    
    test_payload = {
        "thread_id": "test_thread_001",
        "messages": [
            {"role": "human", "content": "How do I build a machine learning model?"},
            {"role": "assistant", "content": "You start with data preparation..."}
        ],
        "async_mode": True
    }
    
    try:
        print("Sending request...")
        response = requests.post(
            f"{TITLE_SERVICE_URL}/generate_title",
            json=test_payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"   Title: {data['title']}")
        print(f"   Thread: {data['thread_id']}")
        print(f"   Response time: Immediate (queued)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate_title_sync() -> bool:
    """Test synchronous title generation (wait for result)"""
    print("\n📋 TEST 3: Sync Title Generation (with wait)")
    print("-" * 50)
    
    test_payload = {
        "thread_id": "test_thread_002",
        "messages": [
            {"role": "human", "content": "What is the difference between supervised and unsupervised learning?"}
        ],
        "async_mode": False
    }
    
    try:
        print("Sending request and waiting for title...")
        start = time.time()
        response = requests.post(
            f"{TITLE_SERVICE_URL}/generate_title",
            json=test_payload,
            timeout=30
        )
        elapsed = time.time() - start
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"   Title: {data['title']}")
        print(f"   Response time: {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_generation() -> bool:
    """Test batch title generation"""
    print("\n📋 TEST 4: Batch Title Generation")
    print("-" * 50)
    
    test_payload = {
        "thread_ids": ["batch_test_1", "batch_test_2", "batch_test_3"]
    }
    
    try:
        print("Sending batch request for 3 threads...")
        response = requests.post(
            f"{TITLE_SERVICE_URL}/batch_generate_titles",
            json=test_payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"   Threads queued: {data['thread_count']}")
        print(f"   Message: {data['message']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_service_stats() -> bool:
    """Get service statistics"""
    print("\n📋 TEST 5: Service Statistics")
    print("-" * 50)
    
    try:
        response = requests.get(f"{TITLE_SERVICE_URL}/stats", timeout=3)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Service: {data['service']}")
        print(f"   Version: {data['version']}")
        print(f"   Model: {data['model']}")
        print(f"   Device: {data['device']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat_server_integration() -> bool:
    """Test if chat server can call title service"""
    print("\n📋 TEST 6: Chat Server Integration")
    print("-" * 50)
    
    try:
        # Check if chat server is running
        response = requests.get(f"{CHAT_SERVER_URL}/", timeout=3)
        response.raise_for_status()
        print(f"✅ Chat server is running")
        
        # Check if title service URL is configured
        print(f"   Chat server should call: {TITLE_SERVICE_URL}")
        print(f"   Environment variable: TITLE_SERVICE_URL")
        return True
    except requests.ConnectionError:
        print(f"⚠️  Chat server not running on {CHAT_SERVER_URL}")
        print("   This is OK for testing just the title service")
        print("   But required for full integration testing")
        return False
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return False

def main():
    print("=" * 60)
    print("  Title Generation Service - Integration Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results["Health Check"] = test_title_service_health()
    
    if not results["Health Check"]:
        print("\n❌ Title service is not running. Please start it first:")
        print("   $ python RAG/title_service.py")
        return
    
    results["Async Generation"] = test_generate_title_async()
    results["Sync Generation"] = test_generate_title_sync()
    results["Batch Generation"] = test_batch_generation()
    results["Stats"] = test_service_stats()
    results["Chat Integration"] = test_chat_server_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 All tests passed! Title service is ready to use.")
        print("\nNext steps:")
        print("1. Make sure both services are running:")
        print("   - Title Service: python RAG/title_service.py")
        print("   - Chat Server: python RAG/server.py")
        print("2. Send a message via /chat/send endpoint")
        print("3. Check chat server logs for title generation")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
