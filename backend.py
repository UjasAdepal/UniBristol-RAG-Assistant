import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from flashrank import Ranker, RerankRequest

# Config & Constants
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

COURSE_VECTOR_PATH = "./faiss_course_store"
FAQ_VECTOR_PATH = "./faiss_faq_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# Resource Caching

@st.cache_resource
def get_embeddings():
    """Load and cache the embedding model to prevent reloading."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_reranker():
    """Load and cache the FlashRank reranker."""
    return Ranker(model_name=RERANK_MODEL, cache_dir="./opt")

# Core Logic

def load_vectorstore(folder_path):
    """Safely load a FAISS index if the folder exists."""
    if not os.path.exists(folder_path):
        return None
    return FAISS.load_local(
        folder_path, 
        get_embeddings(), 
        allow_dangerous_deserialization=True
    )

def rerank_docs(query, docs):
    """Rerank retrieved documents based on relevance score."""
    if not docs:
        return []
    
    ranker = get_reranker()
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(docs)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Filter by score threshold (0.001) and take top 5
    sorted_docs = []
    for res in results:
        if res['score'] > 0.001: 
            doc = docs[int(res['id'])]
            doc.metadata["score"] = res['score']
            sorted_docs.append(doc)
            
    return sorted_docs[:5]

def get_qa_chain():
    """Initialize the RAG pipeline with hybrid retrieval."""
    faq_store = load_vectorstore(FAQ_VECTOR_PATH)
    course_store = load_vectorstore(COURSE_VECTOR_PATH)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    template = """
    You are an expert academic advisor for the University of Bristol.
    Use the provided context to answer the student's question accurately.
    
    Guidelines:
    1. If the exact answer is in the context, use it.
    2. If the user asks about a specific course (e.g., MSc AI) and it is NOT in the context, explicitly say: "I currently do not have information on that specific course."
    3. Be friendly but professional.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    def run_hybrid_retrieval(inputs):
        question = inputs["question"]
        topics = inputs.get("topics", [])

        # 1. Retrieve from FAQ Store
        docs = []
        if faq_store:
            kwargs = {"k": 10}
            if topics:
                kwargs["filter"] = {"topics": {"$in": topics}}
            docs.extend(faq_store.similarity_search(question, **kwargs))

        # 2. Retrieve from Course Store
        if course_store:
            docs.extend(course_store.similarity_search(question, k=10))

        # 3. Rerank Results
        best_docs = rerank_docs(question, docs)

        # 4. Generate Answer
        context_str = "\n\n".join([d.page_content for d in best_docs])
        response = llm.invoke(prompt.format(context=context_str, question=question))
        
        # 5. Extract Source Data
        source_data = [{
            "title": doc.metadata.get("title", "Unknown"),
            "source": doc.metadata.get("source", "University Database"),
            "content": doc.page_content,
            "score": doc.metadata.get("score", 0)
        } for doc in best_docs]

        return response.content, source_data

    return RunnableLambda(run_hybrid_retrieval)