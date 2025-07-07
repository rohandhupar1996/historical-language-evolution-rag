# ==========================================
# FILE: src/rag_system/qa_chain.py (FIXED)
# ==========================================
"""Question-answering chain management - FIXED for proper LLM support."""

from typing import Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pathlib import Path
from .config import EMBEDDING_MODEL


class QAChainManager:
    """Manages LangChain QA chains with proper LLM provider support."""
    
    def __init__(self, vector_db_path: str, collection_name: str):
        self.vector_db_path = Path(vector_db_path)
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.qa_chain = None
        self.current_provider = None
    
    def setup_vectorstore(self):
        """Setup LangChain vectorstore."""
        self.vectorstore = Chroma(
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print("✅ LangChain vectorstore ready")
    
    def setup_qa_chain(self, llm_provider: str = "simple"):
        """Setup QA chain with specified LLM - FIXED."""
        self.current_provider = llm_provider
        
        if llm_provider == "simple":
            print("📝 Using simple retrieval mode (no LLM generation)")
            self.qa_chain = None
            return
        
        elif llm_provider == "openai":
            print("🚀 Initializing OpenAI GPT model...")
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required")
                
                # Use the new OpenAI integration
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=api_key
                )
                print("✅ OpenAI GPT model initialized successfully!")
                
            except ImportError:
                print("❌ langchain_openai not installed. Installing...")
                print("💡 Run: pip install langchain-openai")
                self.qa_chain = None
                return
            except Exception as e:
                print(f"❌ OpenAI initialization failed: {e}")
                print("🔄 Falling back to simple mode...")
                self.qa_chain = None
                return
        
        elif llm_provider == "huggingface":
            print("🤗 Initializing HuggingFace model...")
            try:
                from langchain_community.llms import HuggingFacePipeline
                from transformers import pipeline
                
                # Use a small, fast model for better performance
                hf_pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",  # Smaller, faster model
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=50256,
                    truncation=True,
                    return_full_text=False
                )
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
                print("✅ HuggingFace model initialized successfully!")
                
            except Exception as e:
                print(f"❌ HuggingFace model loading failed: {e}")
                print("💡 Try: pip install transformers accelerate")
                print("🔄 Falling back to simple mode...")
                self.qa_chain = None
                return
        
        else:
            print(f"❌ Unknown LLM provider: {llm_provider}")
            print("🔄 Falling back to simple mode...")
            self.qa_chain = None
            return
        
        # Create the QA chain with the initialized LLM
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
            print(f"✅ QA chain ready with {llm_provider} provider!")
            
        except Exception as e:
            print(f"❌ QA chain creation failed: {e}")
            self.qa_chain = None
    
    def ask_question(self, question: str, period_filter: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question using the QA chain."""
        
        if self.qa_chain is None:
            # Simple retrieval mode
            print(f"📝 Using simple retrieval for: {question[:50]}...")
            
            if period_filter:
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": {"period": period_filter}}
                )
            else:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
            docs = retriever.get_relevant_documents(question)
            
            answer_parts = []
            for i, doc in enumerate(docs[:3]):
                answer_parts.append(f"[Chunk {i+1}]: {doc.page_content[:200]}...")
            
            return {
                'question': question,
                'answer': "\n\n".join(answer_parts),
                'source_documents': [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in docs
                ],
                'mode': 'simple_retrieval'
            }
        
        # LLM generation mode
        print(f"🤖 Using {self.current_provider} LLM for: {question[:50]}...")
        
        try:
            if period_filter:
                filtered_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": {"period": period_filter}}
                )
                temp_qa_chain = RetrievalQA.from_chain_type(
                    llm=self.qa_chain.combine_documents_chain.llm_chain.llm,
                    chain_type="stuff",
                    retriever=filtered_retriever,
                    return_source_documents=True
                )
                result = temp_qa_chain.invoke({"query": question})
            else:
                result = self.qa_chain.invoke({"query": question})
            
            return {
                'question': question,
                'answer': result['result'],
                'source_documents': [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in result['source_documents']
                ],
                'mode': f'{self.current_provider}_generation'
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            print("🔄 Falling back to simple retrieval...")
            
            # Fallback to simple mode
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(question)
            return {
                'question': question,
                'answer': f"LLM failed, showing retrieved content: {docs[0].page_content[:300]}..." if docs else "No results found",
                'source_documents': [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in docs
                ],
                'mode': 'fallback_retrieval'
            }