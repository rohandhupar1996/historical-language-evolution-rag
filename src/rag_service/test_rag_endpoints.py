#!/usr/bin/env python3
"""
RAG Server Endpoint Testing Script
Tests all endpoints to ensure they're working before web app integration
"""

import requests
import json
import time
from typing import Dict, Any

class RAGEndpointTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip('/')
        self.test_results = {}
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test /health endpoint"""
        print("🔍 Testing /health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health: {data}")
                return {"status": "success", "data": data}
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return {"status": "error", "message": f"Status {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Health endpoint error: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_semantic_search_endpoint(self) -> Dict[str, Any]:
        """Test /semantic_search endpoint"""
        print("\n🔍 Testing /semantic_search endpoint...")
        
        test_queries = [
            {"query": "deutsche sprache", "k": 3},
            {"query": "religious texts medieval", "k": 5, "period_filter": "1650-1700"},
            {"query": "archaic spelling patterns", "k": 4}
        ]
        
        results = []
        
        for i, test_query in enumerate(test_queries):
            try:
                print(f"   Testing query {i+1}: '{test_query['query']}'")
                
                response = requests.post(
                    f"{self.base_url}/semantic_search",
                    json=test_query,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Query {i+1}: Found {data.get('total_results', 0)} results")
                    results.append({"query": test_query['query'], "results": data.get('total_results', 0)})
                else:
                    print(f"   ❌ Query {i+1} failed: {response.status_code}")
                    results.append({"query": test_query['query'], "error": f"Status {response.status_code}"})
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Query {i+1} error: {e}")
                results.append({"query": test_query['query'], "error": str(e)})
        
        return {"status": "tested", "results": results}
    
    def test_question_answering_endpoint(self) -> Dict[str, Any]:
        """Test /ask_question endpoint"""
        print("\n🔍 Testing /ask_question endpoint...")
        
        test_questions = [
            {"question": "How did German language evolve?"},
            {"question": "What are common archaic spelling patterns?", "period_filter": "1650-1700"},
            {"question": "Describe religious language in historical texts"}
        ]
        
        results = []
        
        for i, test_question in enumerate(test_questions):
            try:
                print(f"   Testing question {i+1}: '{test_question['question'][:40]}...'")
                
                response = requests.post(
                    f"{self.base_url}/ask_question",
                    json=test_question,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer_length = len(data.get('answer', ''))
                    sources = len(data.get('source_documents', []))
                    print(f"   ✅ Question {i+1}: Answer length {answer_length}, {sources} sources")
                    results.append({
                        "question": test_question['question'][:40],
                        "answer_length": answer_length,
                        "sources": sources
                    })
                else:
                    print(f"   ❌ Question {i+1} failed: {response.status_code}")
                    results.append({"question": test_question['question'][:40], "error": f"Status {response.status_code}"})
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Question {i+1} error: {e}")
                results.append({"question": test_question['question'][:40], "error": str(e)})
        
        return {"status": "tested", "results": results}
    
    def test_language_evolution_endpoint(self) -> Dict[str, Any]:
        """Test /language_evolution endpoint"""
        print("\n🔍 Testing /language_evolution endpoint...")
        
        test_words = [
            {"word": "deutsch"},
            {"word": "gott", "periods": ["1650-1700", "1700-1750"]},
            {"word": "recht"}
        ]
        
        results = []
        
        for i, test_word in enumerate(test_words):
            try:
                print(f"   Testing word {i+1}: '{test_word['word']}'")
                
                response = requests.post(
                    f"{self.base_url}/language_evolution",
                    json=test_word,
                    timeout=45
                )
                
                if response.status_code == 200:
                    data = response.json()
                    periods = len(data.get('periods', {}))
                    summary_length = len(data.get('summary', ''))
                    print(f"   ✅ Word {i+1}: {periods} periods analyzed, summary length {summary_length}")
                    results.append({
                        "word": test_word['word'],
                        "periods_analyzed": periods,
                        "summary_length": summary_length
                    })
                else:
                    print(f"   ❌ Word {i+1} failed: {response.status_code}")
                    results.append({"word": test_word['word'], "error": f"Status {response.status_code}"})
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Word {i+1} error: {e}")
                results.append({"word": test_word['word'], "error": str(e)})
        
        return {"status": "tested", "results": results}
    
    def test_statistics_endpoint(self) -> Dict[str, Any]:
        """Test /statistics endpoint"""
        print("\n🔍 Testing /statistics endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/statistics", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                db_stats = data.get('database_stats', {})
                vector_stats = data.get('vector_stats', {})
                
                total_chunks = db_stats.get('chunks', {}).get('total_chunks', 0)
                total_embeddings = vector_stats.get('total_embeddings', 0)
                
                print(f"✅ Statistics: {total_chunks} chunks, {total_embeddings} embeddings")
                return {
                    "status": "success", 
                    "total_chunks": total_chunks,
                    "total_embeddings": total_embeddings,
                    "data": data
                }
            else:
                print(f"❌ Statistics failed: {response.status_code}")
                return {"status": "error", "message": f"Status {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Statistics error: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run all endpoint tests"""
        print("🚀 RAG Server Endpoint Testing Suite")
        print("=" * 50)
        
        # Test server availability first
        health_result = self.test_health_endpoint()
        self.test_results['health'] = health_result
        
        if health_result['status'] != 'success':
            print("\n❌ Server not healthy, skipping other tests")
            return self.test_results
        
        if not health_result['data'].get('is_initialized'):
            print("\n❌ RAG system not initialized, skipping functional tests")
            return self.test_results
        
        # Test all endpoints
        self.test_results['semantic_search'] = self.test_semantic_search_endpoint()
        self.test_results['question_answering'] = self.test_question_answering_endpoint()
        self.test_results['language_evolution'] = self.test_language_evolution_endpoint()
        self.test_results['statistics'] = self.test_statistics_endpoint()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        for endpoint, result in self.test_results.items():
            if endpoint == 'health':
                total_tests += 1
                if result['status'] == 'success' and result['data'].get('is_initialized'):
                    passed_tests += 1
                    print(f"✅ {endpoint.upper()}: Ready")
                else:
                    print(f"❌ {endpoint.upper()}: Not ready")
            
            elif endpoint == 'statistics':
                total_tests += 1
                if result['status'] == 'success':
                    passed_tests += 1
                    print(f"✅ {endpoint.upper()}: {result['total_chunks']} chunks, {result['total_embeddings']} embeddings")
                else:
                    print(f"❌ {endpoint.upper()}: Failed")
            
            elif 'results' in result:
                for test in result['results']:
                    total_tests += 1
                    if 'error' not in test:
                        passed_tests += 1
                
                success_rate = len([t for t in result['results'] if 'error' not in t]) / len(result['results'])
                if success_rate > 0.5:
                    print(f"✅ {endpoint.upper()}: {int(success_rate * 100)}% success rate")
                else:
                    print(f"❌ {endpoint.upper()}: {int(success_rate * 100)}% success rate")
        
        print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed ({int(passed_tests/total_tests*100)}%)")
        
        if passed_tests / total_tests >= 0.8:
            print("🎉 RAG Server is READY for web app integration!")
        else:
            print("⚠️ Fix issues before web app integration")
        
        return passed_tests / total_tests >= 0.8


def main():
    """Run the test suite"""
    print("🧪 Starting RAG Server Endpoint Tests...")
    print("Make sure your RAG server is running on http://127.0.0.1:8001")
    print()
    
    # Wait a moment for user to confirm
    input("Press Enter to start testing (or Ctrl+C to cancel)...")
    
    tester = RAGEndpointTester()
    results = tester.run_full_test_suite()
    
    # Save results
    with open('rag_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: rag_test_results.json")
    
    return results


if __name__ == "__main__":
    main()