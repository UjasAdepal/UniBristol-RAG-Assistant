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

st.set_page_config(
    page_title="BristolBot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# STYLING

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
    --uob-red: #C02F38;
    --uob-red-dark: #9A252C;
    --ink: #1A1A1A;
    --ink-soft: #5B5B5B;
    --rule: #E3E0DC;
    --paper: #FFFFFF;
    --paper-warm: #F7F5F3;
}

.stApp {
    background: var(--paper);
}

html, body, [class*="css"], .stMarkdown, .stChatMessage p {
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--ink);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
    max-width: 780px;
}

/* Masthead */
.uob-bar {
    background: var(--uob-red);
    margin: -1rem -100vw 0 -100vw;
    padding: 0.9rem 100vw 0.9rem 100vw;
}
.uob-bar-inner {
    max-width: 780px;
    margin: 0 auto;
}
.uob-bar-inner .name {
    color: #fff;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.uob-bar-inner .org {
    color: rgba(255,255,255,0.82);
    font-size: 0.9rem;
    font-weight: 400;
    margin-left: 0.6rem;
}

.uob-intro {
    color: var(--ink-soft);
    font-size: 1.02rem;
    line-height: 1.55;
    margin: 1.6rem 0 1.9rem 0;
    max-width: 62ch;
}

.uob-section-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.7rem;
}

/* Suggested question buttons */
div[data-testid="stButton"] > button {
    background: var(--paper);
    color: var(--ink);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 0.75rem 0.95rem;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    text-align: left;
    line-height: 1.4;
    transition: border-color 0.15s ease, background 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--uob-red);
    background: var(--paper-warm);
    color: var(--ink);
}
div[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--uob-red);
    outline-offset: 2px;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0;
}
[data-testid="stChatMessage"] .stMarkdown p {
    font-size: 1.02rem;
    line-height: 1.62;
}

/* Source cards */
.src-card {
    border: 1px solid var(--rule);
    border-left: 3px solid var(--uob-red);
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
    background: var(--paper-warm);
}
.src-card .src-title {
    font-size: 0.94rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.35;
    margin-bottom: 0.25rem;
}
.src-card a {
    font-size: 0.87rem;
    color: var(--uob-red);
    text-decoration: none;
}
.src-card a:hover { text-decoration: underline; }
.src-card .src-score {
    font-size: 0.8rem;
    color: var(--ink-soft);
    margin-left: 0.5rem;
}

.uob-meta {
    font-size: 0.82rem;
    color: var(--ink-soft);
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1rem;
}

/* Footer */
.uob-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--rule);
    font-size: 0.82rem;
    color: var(--ink-soft);
    line-height: 1.5;
}

/* Expander */
[data-testid="stExpander"] {
    border: none;
}
[data-testid="stExpander"] summary {
    font-size: 0.9rem;
    color: var(--uob-red);
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

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

# UI HELPERS

def render_sources(sources, debug_mode):
    """Render retrieved sources as cards inside an expander."""
    if not sources:
        return
    label = f"Sources ({len(sources)})"
    with st.expander(label):
        for src in sources:
            st.markdown(
                f"""<div class="src-card">
                        <div class="src-title">{src['title']}</div>
                        <a href="{src['url']}" target="_blank">View on bristol.ac.uk</a>
                        <span class="src-score">relevance {src['score']:.2f}</span>
                    </div>""",
                unsafe_allow_html=True,
            )
            if debug_mode:
                st.code(src['content'][:300] + "...", language="text")

# INITIALIZATION

rag_system = initialize_rag_system()

if "query_times" not in st.session_state:
    st.session_state.query_times = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR (technical panel)

with st.sidebar:
    st.markdown("**Technical panel**")
    debug_mode = st.toggle("Show retrieval diagnostics", value=False)

    if debug_mode:
        st.markdown("---")
        course_loaded = rag_system["course_store"] is not None
        st.write("Vector store:", "loaded" if course_loaded else "missing")
        if not course_loaded:
            st.error("Vector store is missing.")

        st.caption(f"Embedding: {CONFIG['retrieval']['embedding_model']}")
        st.caption(f"Reranker: {CONFIG['retrieval']['reranker_model']}")
        st.caption(f"Score threshold: {CONFIG['retrieval']['score_threshold']}")
        st.caption(f"Generation model: {CONFIG['model']['name']}")

        if st.session_state.query_times:
            times = st.session_state.query_times
            st.markdown("---")
            st.metric("Mean response", f"{sum(times)/len(times):.2f}s")
            st.caption(f"Fastest {min(times):.2f}s · Slowest {max(times):.2f}s · {len(times)} queries")

        st.markdown("---")
        if st.button("Clear cache"):
            st.cache_resource.clear()
            st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.rerun()

# MASTHEAD

st.markdown(
    """<div class="uob-bar"><div class="uob-bar-inner">
           <span class="name">BristolBot</span><span class="org">University of Bristol student enquiries</span>
       </div></div>""",
    unsafe_allow_html=True,
)

# LANDING (empty conversation)

if not st.session_state.messages:
    st.markdown(
        """<div class="uob-intro">
        Answers to questions about admissions, tuition fees, scholarships, accommodation
        and University regulations, drawn from published University of Bristol pages.
        Every answer links to the page it came from.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="uob-section-label">Common questions</div>', unsafe_allow_html=True)

    example_questions = [
        "How much is the Cratchley Scholarship worth?",
        "Can I pay tuition fees in instalments?",
        "What are the accommodation fee payment dates?",
        "What is the pass mark for a Masters dissertation?",
    ]

    cols = st.columns(2)
    for i, example_q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(example_q, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example_q})
                try:
                    answer, sources, debug_info = get_answer(example_q, rag_system, debug_mode=debug_mode)
                    if debug_info and "timings" in debug_info:
                        st.session_state.query_times.append(debug_info["timings"]["total"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timing": debug_info["timings"]["total"] if debug_info and "timings" in debug_info else 0
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Something went wrong reaching the answering service: {e}"
                    })
                st.rerun()

# CONVERSATION

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            render_sources(message["sources"], debug_mode)

        if message["role"] == "assistant" and "timing" in message and debug_mode:
            st.markdown(
                f'<div class="uob-meta">Answered in {message["timing"]:.2f}s</div>',
                unsafe_allow_html=True,
            )

# INPUT

if user_input := st.chat_input("Ask about fees, scholarships, accommodation or regulations"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching University pages"):
            try:
                answer, sources, debug_info = get_answer(user_input, rag_system, debug_mode=debug_mode)

                if debug_info and "timings" in debug_info:
                    st.session_state.query_times.append(debug_info["timings"]["total"])

                st.markdown(answer)
                render_sources(sources, debug_mode)

                if debug_mode and debug_info:
                    st.markdown(
                        f'<div class="uob-meta">Retrieved {debug_info.get("total_retrieved", 0)} passages, '
                        f'kept {debug_info.get("after_rerank", 0)} above threshold · '
                        f'{debug_info["timings"]["total"]:.2f}s</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "timing": debug_info["timings"]["total"] if debug_info and "timings" in debug_info else 0
                })
                st.rerun()

            except Exception as e:
                st.error(f"Something went wrong reaching the answering service: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Something went wrong reaching the answering service: {e}"
                })
                if debug_mode:
                    st.exception(e)
                st.rerun()

# FOOTER

st.markdown(
    """<div class="uob-footer">
    An independent project, not affiliated with or endorsed by the University of Bristol.
    Answers are generated from published University web pages and may be out of date —
    check the linked source before acting on anything.
    </div>""",
    unsafe_allow_html=True,
)
