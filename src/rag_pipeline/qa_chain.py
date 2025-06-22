# ==========================================
# FILE: rag_system/qa_chain.py
# ==========================================
"""Question-answering chain management."""

from typing import Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pathlib import Path
from .config import EMBEDDING_MODEL


class QAChainManager:
    """Manages LangChain QA chains."""
    
    def __init__(self, vector_db_path: str, collection_name: str):
        self.vector_db_path = Path(vector_db_path)
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.qa_chain = None
    
    def setup_vectorstore(self):
        """Setup LangChain vectorstore."""
        self.vectorstore = Chroma(
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print("✅ LangChain vectorstore ready")
    
    def setup_qa_chain(self, llm_provider: str = "simple"):
        """Setup QA chain with specified LLM."""
        if llm_provider == "simple":
            print("⚠️ Using simple retrieval mode (no LLM generation)")
            self.qa_chain = None
            return
        
        if llm_provider == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable required")
            from langchain_community.llms import OpenAI
            llm = OpenAI(temperature=0.1, max_tokens=1000)
        
        elif llm_provider == "huggingface":
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline
            
            try:
                hf_pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    max_length=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=50256
                )
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
            except Exception as e:
                print(f"⚠️ HuggingFace model loading failed: {e}")
                self.qa_chain = None
                return
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        print("✅ QA chain ready")
    
    def ask_question(self, question: str, period_filter: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question using the QA chain."""
        if self.qa_chain is None:
            # Simple retrieval mode
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
            result = temp_qa_chain({"query": question})
        else:
            result = self.qa_chain({"query": question})
        
        return {
            'question': question,
            'answer': result['result'],
            'source_documents': [
                {'content': doc.page_content, 'metadata': doc.metadata}
                for doc in result['source_documents']
            ],
            'mode': 'llm_generation'
        }
