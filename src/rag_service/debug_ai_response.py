#!/usr/bin/env python3
"""
Debug script to test AI responses and fix Streamlit display issue
"""

import requests
import json

def test_ai_question():
    """Test AI question and see exact response"""
    print("🧪 Testing AI Question Response...")
    
    try:
        # Test the exact same call Streamlit makes
        response = requests.post(
            "http://127.0.0.1:8001/ask_question",
            json={"question": "How did German spelling change over time?"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse Keys: {list(data.keys())}")
            
            # Check each field
            for key, value in data.items():
                if isinstance(value, str):
                    print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
                elif isinstance(value, list):
                    print(f"{key}: List with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        print(f"  First item keys: {list(value[0].keys())}")
                else:
                    print(f"{key}: {type(value)} - {value}")
            
            # Check if answer is empty
            answer = data.get('answer', '')
            if not answer or answer.strip() == '':
                print("\n❌ PROBLEM: Answer is empty!")
            else:
                print(f"\n✅ Answer length: {len(answer)} characters")
                print(f"Answer preview: {answer[:200]}...")
            
            # Check sources
            sources = data.get('source_documents', [])
            print(f"\n📚 Sources: {len(sources)} documents")
            if sources:
                for i, source in enumerate(sources[:2]):
                    content = source.get('content', '')
                    metadata = source.get('metadata', {})
                    print(f"  Source {i+1}: {len(content)} chars, metadata: {list(metadata.keys())}")
            
            return data
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_streamlit_simulation():
    """Simulate exactly what Streamlit does"""
    print("\n🎭 Simulating Streamlit AI Call...")
    
    # This is exactly what the SimpleRAGClient does
    try:
        data = {"question": "Test question about German language"}
        response = requests.post("http://127.0.0.1:8001/ask_question", json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Simulate Streamlit logic
            if result:
                print("✅ Result exists")
                answer = result.get('answer', '')
                sources = result.get('source_documents', [])
                
                print(f"Answer: {bool(answer)}")
                print(f"Sources: {len(sources)}")
                
                # This is the exact check Streamlit does
                if result.get('source_documents'):
                    print("✅ Sources check passed")
                else:
                    print("❌ Sources check failed")
                
                return True
            else:
                print("❌ Result is None/False")
                return False
        else:
            print(f"❌ Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging AI Response Issue")
    print("=" * 50)
    
    # Test 1: Direct API call
    result = test_ai_question()
    
    # Test 2: Streamlit simulation
    streamlit_works = test_streamlit_simulation()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS:")
    
    if result:
        if not result.get('answer'):
            print("❌ ISSUE: AI returns empty answers")
            print("💡 FIX: Check RAG QA chain initialization")
        elif not result.get('source_documents'):
            print("❌ ISSUE: No source documents returned")
            print("💡 FIX: Check vectorstore setup")
        else:
            print("✅ API works fine")
            if not streamlit_works:
                print("❌ ISSUE: Streamlit response handling")
                print("💡 FIX: Check Streamlit result processing")
    else:
        print("❌ ISSUE: API completely broken")
        print("💡 FIX: Check RAG server logs")

