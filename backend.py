import os
# 1. FIX CRASH: Disable parallelism before importing HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import pickle
import streamlit as st  # Import Streamlit for caching
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

load_dotenv()

COURSE_VECTOR_PATH = "./faiss_course_store"
FAQ_VECTOR_PATH = "./faiss_faq_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# 2. PERFORMANCE FIX: Cache the Embeddings Model
# This prevents the app from reloading the heavy 400MB model on every click.
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# 3. PERFORMANCE FIX: Cache the Reranker
@st.cache_resource
def get_reranker():
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")

def load_vectorstore(folder):
    if not os.path.exists(folder):
        return None
    
    # Use the cached embeddings instead of creating new ones
    embeddings = get_embeddings()
    
    # Allow dangerous deserialization is required for local pickle files
    vectorstore = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def rerank_docs(query, docs):
    if not docs:
        return []
    
    ranker = get_reranker()
    
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(docs)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    sorted_docs = []
    for res in results:
        if res['score'] > 0.001: 
            doc = docs[int(res['id'])]
            doc.metadata["score"] = res['score']
            sorted_docs.append(doc)
            
    return sorted_docs[:5]

def get_qa_chain():
    # Load stores (Streamlit cache will make this fast)
    faq_store = load_vectorstore(FAQ_VECTOR_PATH)
    course_store = load_vectorstore(COURSE_VECTOR_PATH)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    prompt = PromptTemplate.from_template("""
    You are an expert academic advisor for the University of Bristol.
    Use the provided context to answer the student's question accurately.
    
    Guidelines:
    1. If the exact answer is in the context, use it.
    2. If the user asks about a specific course (e.g., MSc AI) and it is NOT in the context, explicitly say: "I currently do not have information on that specific course." DO NOT hallucinate fees or details.
    3. Be friendly but professional.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """)

    def run_hybrid(inputs):
        question = inputs["question"]
        topics = inputs.get("topics", [])

        faq_docs = []
        if faq_store:
            search_kwargs = {"k": 10}
            if topics:
                search_kwargs["filter"] = {"topics": {"$in": topics}}
            faq_docs = faq_store.similarity_search(question, **search_kwargs)

        course_docs = []
        if course_store:
            course_docs = course_store.similarity_search(question, k=10)

        all_retrieved = faq_docs + course_docs
        best_docs = rerank_docs(question, all_retrieved)

        context_text = "\n\n".join([d.page_content for d in best_docs])
        response = llm.invoke(prompt.format(context=context_text, question=question))
        
        # Build source list
        source_list = []
        seen = set()
        for doc in best_docs:
            # Create a unique key for the source
            title = doc.metadata.get("title", "FAQ")
            if title not in seen:
                source_list.append({
                    "source": doc.metadata.get("source", "University Database"),
                    "title": title
                })
                seen.add(title)

        source_data = []
        for doc in best_docs:
            source_data.append({
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "University Database"),
                "content": doc.page_content,  # <--- THIS IS NEW (The actual text)
                "score": doc.metadata.get("score", 0)
            })

        return response.content, source_data

    return RunnableLambda(run_hybrid)