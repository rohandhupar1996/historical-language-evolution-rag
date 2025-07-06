#!/usr/bin/env python3
"""
RAG Server Debug Script
Identifies and fixes the 500 errors in semantic search and language evolution
"""

import requests
import json

def debug_semantic_search():
    """Debug semantic search endpoint"""
    print("🔍 Debugging semantic search...")
    
    try:
        # Simple test query
        response = requests.post(
            "http://127.0.0.1:8001/semantic_search",
            json={"query": "test", "k": 1},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 500:
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error text: {response.text}")
        else:
            data = response.json()
            print(f"Success: {data}")
            
    except Exception as e:
        print(f"Request error: {e}")

def debug_language_evolution():
    """Debug language evolution endpoint"""
    print("\n🔍 Debugging language evolution...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/language_evolution",
            json={"word": "test"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 500:
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error text: {response.text}")
        else:
            data = response.json()
            print(f"Success: {data}")
            
    except Exception as e:
        print(f"Request error: {e}")

def test_working_endpoint():
    """Test the working question endpoint for comparison"""
    print("\n✅ Testing working question endpoint...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/ask_question",
            json={"question": "test question"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Answer length: {len(data.get('answer', ''))}")
            print(f"Sources: {len(data.get('source_documents', []))}")
        
    except Exception as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    print("🚀 RAG Server Debug Session")
    print("=" * 40)
    
    debug_semantic_search()
    debug_language_evolution()
    test_working_endpoint()
    
    print("\n💡 Common fixes:")
    print("1. Check vector database collection exists")
    print("2. Verify embedding manager is working")  
    print("3. Check ChromaDB client connection")
    print("4. Restart RAG server after fixes")