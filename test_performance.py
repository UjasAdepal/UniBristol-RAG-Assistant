"""
Performance comparison: Old vs New approach
Run: python test_performance.py
"""

import time
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from flashrank import Ranker, RerankRequest

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test questions
TEST_QUESTIONS = [
    "How much is the Cratchley Scholarship worth?",
    "Can I pay tuition fees in installments?",
    "What are the accommodation fee dates?",
    "What is the pass mark for a Masters dissertation?",
    "When does Welcome Week start?"
]

# Configuration
VECTOR_STORE_PATH = "./faiss_course_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

print("=" * 70)
print("🔬 PERFORMANCE COMPARISON: Old vs New Approach")
print("=" * 70)

# ==================== OLD APPROACH (Load on every query) ====================

def old_approach_query(question):
    """Simulates old approach: load everything on each query"""
    start = time.time()
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Load vector store
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load reranker
    ranker = Ranker(model_name=RERANK_MODEL, cache_dir="./opt")
    
    # Create LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    
    # Retrieval
    docs = vectorstore.similarity_search(question, k=10)
    
    # Rerank (simplified for demo)
    passages = [{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Generation (simplified - just check we can call it)
    # In real app, would generate full answer
    
    elapsed = time.time() - start
    return elapsed

print("\n🐌 OLD APPROACH (Load everything on each query):")
print("-" * 70)

old_times = []
for i, q in enumerate(TEST_QUESTIONS, 1):
    print(f"  Query {i}: {q[:50]}...", end=" ")
    t = old_approach_query(q)
    old_times.append(t)
    print(f"⏱️  {t:.2f}s")

old_avg = sum(old_times) / len(old_times)
print(f"\n  📊 Average: {old_avg:.2f}s")

# ==================== NEW APPROACH (Load once, reuse) ====================

def initialize_system():
    """Load everything once"""
    print("\n  🔄 Initializing system (one-time cost)...")
    start = time.time()
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    ranker = Ranker(model_name=RERANK_MODEL, cache_dir="./opt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    
    init_time = time.time() - start
    print(f"  ✅ System initialized in {init_time:.2f}s")
    
    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "ranker": ranker,
        "llm": llm
    }

def new_approach_query(question, system):
    """Use pre-loaded system"""
    start = time.time()
    
    # Retrieval (using pre-loaded store)
    docs = system["vectorstore"].similarity_search(question, k=10)
    
    # Rerank (using pre-loaded ranker)
    passages = [{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    results = system["ranker"].rerank(rerank_request)
    
    # Generation would use pre-loaded LLM
    
    elapsed = time.time() - start
    return elapsed

print("\n" + "=" * 70)
print("🚀 NEW APPROACH (Load once, reuse for all queries):")
print("-" * 70)

# Initialize once
system = initialize_system()

print("\n  Running queries with pre-loaded system:")
print("-" * 70)

new_times = []
for i, q in enumerate(TEST_QUESTIONS, 1):
    print(f"  Query {i}: {q[:50]}...", end=" ")
    t = new_approach_query(q, system)
    new_times.append(t)
    print(f"⏱️  {t:.2f}s")

new_avg = sum(new_times) / len(new_times)
print(f"\n  📊 Average: {new_avg:.2f}s")

# ==================== RESULTS ====================

print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)

print(f"\n  Old Approach Average: {old_avg:.2f}s")
print(f"  New Approach Average: {new_avg:.2f}s")

improvement = ((old_avg - new_avg) / old_avg) * 100
time_saved = old_avg - new_avg

print(f"\n  ⚡ Improvement: {improvement:.1f}% faster")
print(f"  ⏱️  Time Saved: {time_saved:.2f}s per query")
print(f"  💰 Cost Savings: ~{improvement:.0f}% fewer repeated loads")

# Calculate projected savings
queries_per_day = 100
print(f"\n  📈 Projected savings for {queries_per_day} queries/day:")
print(f"     • Time saved: {(time_saved * queries_per_day):.1f}s (~{(time_saved * queries_per_day / 60):.1f} minutes)")
print(f"     • Better user experience (faster responses)")
print(f"     • Lower memory footprint (no redundant loading)")

print("\n" + "=" * 70)
print("✅ Test complete!")
print("=" * 70)