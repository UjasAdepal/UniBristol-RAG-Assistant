import os
import csv
import time
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from flashrank import Ranker, RerankRequest

# Load shared config
try:
    from config import CONFIG, PATHS
except ImportError:
    st.error(" config.py not found! Create it first.")
    st.stop()

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="BristolBot AI Tutor", layout="wide")

# CACHED RESOURCES

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=CONFIG["retrieval"]["embedding_model"])

@st.cache_resource
def get_reranker():
    return Ranker(model_name=CONFIG["retrieval"]["reranker_model"], cache_dir="./opt")

@st.cache_resource
def load_vectorstore(path):
    if not os.path.exists(path):
        return None
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG components on first load."""
    embeddings = get_embeddings()
    reranker = get_reranker()
    llm = ChatOpenAI(
        model_name=CONFIG["model"]["name"], 
        temperature=CONFIG["model"]["temperature"]
    )
    
    course_store = load_vectorstore(PATHS["course_store"])
    faq_store = load_vectorstore(PATHS["faq_store"]) if PATHS["faq_store"] else None
    
    return {
        "embeddings": embeddings,
        "reranker": reranker,
        "llm": llm,
        "course_store": course_store,
        "faq_store": faq_store
    }

# BACKEND LOGIC 

def rerank_docs(query_text, docs, reranker):
    """Apply cross-encoder reranking to retrieved documents."""
    if not docs:
        return []
    
    passages = [{"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(docs)]
    
    rerank_request = RerankRequest(query=query_text, passages=passages)
    results = reranker.rerank(rerank_request)
    
    sorted_docs = []
    threshold = CONFIG["retrieval"]["score_threshold"]
    
    for res in results:
        if res['score'] > threshold:
            doc = docs[int(res['id'])]
            doc.metadata["score"] = res['score']
            sorted_docs.append(doc)
    
    if not sorted_docs and results:
        if results[0]['score'] > 0.20:
            best_doc = docs[int(results[0]['id'])]
            best_doc.metadata["score"] = results[0]['score']
            sorted_docs.append(best_doc)
            
    return sorted_docs[:CONFIG["retrieval"]["final_k"]]

def get_answer(question, rag_system, debug_mode=False):
    """Execute RAG pipeline: retrieval, reranking, and generation."""
    timings = {}
    start_total = time.time()
    
    course_store = rag_system["course_store"]
    faq_store = rag_system["faq_store"]
    reranker = rag_system["reranker"]
    llm = rag_system["llm"]
    
    start_retrieval = time.time()
    all_retrieved = []
    
    if course_store:
        all_retrieved.extend(
            course_store.similarity_search(question, k=CONFIG["retrieval"]["initial_k"])
        )
    
    if faq_store:
        all_retrieved.extend(
            faq_store.similarity_search(question, k=CONFIG["retrieval"]["initial_k"])
        )
    
    timings["retrieval"] = time.time() - start_retrieval
    
    start_rerank = time.time()
    best_docs = rerank_docs(question, all_retrieved, reranker)
    timings["rerank"] = time.time() - start_rerank
    
    if not best_docs:
        return "I couldn't find relevant information in the database.", [], None
    
    start_generation = time.time()
    prompt = PromptTemplate.from_template(CONFIG["prompt_template"])
    
    context_text = "\n\n".join([d.page_content for d in best_docs])
    response = llm.invoke(prompt.format(context=context_text, question=question))
    timings["generation"] = time.time() - start_generation
    
    timings["total"] = time.time() - start_total
    
    sources = [{
        "title": d.metadata.get("title", "Unknown"),
        "url": d.metadata.get("url", "#"),
        "score": d.metadata.get("score", 0),
        "content": d.page_content
    } for d in best_docs]
    
    debug_info = {
        "total_retrieved": len(all_retrieved),
        "after_rerank": len(best_docs),
        "threshold": CONFIG["retrieval"]["score_threshold"],
        "timings": timings
    } if debug_mode else {"timings": timings}
    
    return response.content, sources, debug_info

def save_feedback(question, response, is_helpful):
    """Log feedback to CSV"""
    file_exists = os.path.isfile(PATHS["feedback_file"])
    with open(PATHS["feedback_file"], mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Response", "Helpful"])
        writer.writerow([datetime.datetime.now(), question, response, "Yes" if is_helpful else "No"])

# INITIALIZATION 

rag_system = initialize_rag_system()

# Initialize session state
if "query_times" not in st.session_state:
    st.session_state.query_times = []

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm BristolBot, your AI assistant for the University of Bristol. I can help you with questions about:\n\n•  Courses and admissions\n• Fees and scholarships\n• Accommodation\n• Regulations and policies\n• 🎓 Student life\n\nWhat would you like to know?"
    })

# SIDEBAR 

with st.sidebar:
    st.header("⚙️ Settings")
    
    debug_mode = st.toggle(" Debug Mode", value=False, help="Show retrieval diagnostics")
    
    st.subheader("Database Status")
    
    course_loaded = rag_system["course_store"] is not None
    faq_loaded = rag_system["faq_store"] is not None
    
    st.write("Course Store:", "success" if course_loaded else "fail")
    
    if PATHS["faq_store"]:
        st.write("FAQ Store:", "success" if faq_loaded else "fail")
        if not course_loaded or not faq_loaded:
            st.error(" Some vector stores are missing!")
    else:
        st.info("Using single unified vector store")
        if not course_loaded:
            st.error(" Vector store is missing!")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.success("Chat cleared!")
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Threshold: {CONFIG['retrieval']['score_threshold']}")
    st.caption(f"Model: {CONFIG['model']['name']}")
    
    # Performance statistics
    if len(st.session_state.query_times) > 0:
        st.markdown("---")
        st.subheader("Performance Stats")
        
        avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times)
        min_time = min(st.session_state.query_times)
        max_time = max(st.session_state.query_times)
        
        st.metric("Average Response", f"{avg_time:.2f}s")
        st.caption(f"Fastest: {min_time:.2f}s | Slowest: {max_time:.2f}s")
        st.caption(f"Total queries: {len(st.session_state.query_times)}")
        
        if avg_time < 1.0:
            st.success("🚀 Excellent performance!")
        elif avg_time < 2.0:
            st.info("Good performance")
        else:
            st.warning("Consider optimization")
    
    # Conversation stats
    if len(st.session_state.messages) > 1:
        st.markdown("---")
        st.subheader("Conversation")
        
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant") - 1
        
        st.metric("Messages", f"{user_msgs + assistant_msgs}")
        st.caption(f"👤 You: {user_msgs} | 🤖 Bot: {assistant_msgs}")

# MAIN UI

st.title("🎓 BristolBot - University of Bristol AI Tutor")
st.caption("Ask me anything about admissions, fees, scholarships, accommodation, and more!")

# Show example questions if chat is new
if len(st.session_state.messages) <= 1:
    st.markdown("### Try these example questions:")
    
    example_questions = [
        "How much is the Cratchley Scholarship worth?",
        "Can I pay tuition fees in installments?",
        "What are the accommodation fee dates?",
        "What is the pass mark for a Masters dissertation?"
    ]
    
    cols = st.columns(2)
    for i, example_q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"{example_q}", key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example_q})
                st.rerun()
    
    st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for j, src in enumerate(message["sources"], 1):
                    st.markdown(f"**{j}. {src['title']}** (Score: {src['score']:.3f})")
                    if debug_mode:
                        st.code(src['content'][:300] + "...", language="text")
                    st.markdown(f"[View source]({src['url']})")
                    if j < len(message["sources"]):
                        st.divider()
        
        # Show timing
        if message["role"] == "assistant" and "timing" in message:
            st.caption(f"Response time: {message['timing']:.2f}s")

# Chat input
if user_input := st.chat_input("Ask your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                answer, sources, debug_info = get_answer(user_input, rag_system, debug_mode=debug_mode)
                
                if debug_info and "timings" in debug_info:
                    st.session_state.query_times.append(debug_info["timings"]["total"])
                
                st.markdown(answer)
                
                # Show timing
                if debug_info and "timings" in debug_info:
                    st.success(f"**Response generated in {debug_info['timings']['total']:.2f}s**")
                    
                    if debug_mode:
                        st.info(f"Retrieved {debug_info.get('total_retrieved', 0)} docs → Filtered to {debug_info.get('after_rerank', 0)}")
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for j, src in enumerate(sources, 1):
                            st.markdown(f"**{j}. {src['title']}** (Score: {src['score']:.3f})")
                            if debug_mode:
                                st.code(src['content'][:300] + "...", language="text")
                            st.markdown(f"[View source]({src['url']})")
                            if j < len(sources):
                                st.divider()
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "timing": debug_info["timings"]["total"] if debug_info and "timings" in debug_info else 0
                })
                
                st.rerun()
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                if debug_mode:
                    st.exception(e)
                st.rerun()