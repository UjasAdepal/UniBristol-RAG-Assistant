import os
import csv
import html
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
    page_title="BristolBot — student enquiries",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# STYLING

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
    --red: #C02F38;
    --red-deep: #8E2028;
    --ink: #16171A;
    --ink-soft: #63666B;
    --rule: #DFDBD6;
    --paper: #FFFFFF;
    --wash: #F7F5F2;
}

/* ---- base ---- */
.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }

html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

.block-container {
    max-width: 720px;
    padding-top: 0 !important;
    padding-bottom: 8rem;
}

/* ---- masthead ---- */
.mast {
    background: var(--red);
    margin: 0 calc(50% - 50vw) 2.6rem calc(50% - 50vw);
    border-bottom: 4px solid var(--red-deep);
}
.mast-in {
    max-width: 720px;
    margin: 0 auto;
    padding: 1.05rem 0;
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
}
.mast-in b {
    color: #fff;
    font-size: 1.22rem;
    font-weight: 700;
    letter-spacing: -0.015em;
}
.mast-in span {
    color: rgba(255,255,255,0.85);
    font-size: 0.92rem;
}

/* ---- opening ---- */
.lede {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.1rem;
    line-height: 1.22;
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.015em;
    margin: 0 0 1rem 0;
}
.standfirst {
    font-size: 1.03rem;
    line-height: 1.6;
    color: var(--ink-soft);
    max-width: 58ch;
    margin-bottom: 2.6rem;
}
.rubric {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--ink-soft);
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 0.2rem;
}

/* ---- suggested questions as list rows ---- */
div[data-testid="stButton"] > button {
    width: 100%;
    background: transparent;
    color: var(--ink);
    border: none;
    border-bottom: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.95rem 0.2rem;
    font-size: 1.02rem;
    font-weight: 400;
    text-align: left;
    line-height: 1.4;
    transition: background 0.12s ease, padding-left 0.12s ease;
}
div[data-testid="stButton"] > button:hover {
    background: var(--wash);
    color: var(--red-deep);
    padding-left: 0.7rem;
}
div[data-testid="stButton"] > button:focus:not(:active) { color: var(--red-deep); }
div[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--red);
    outline-offset: -2px;
}

/* ---- exchange ---- */
.ask {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.42rem;
    line-height: 1.32;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
    margin: 2.4rem 0 1.1rem 0;
    padding-top: 1.6rem;
    border-top: 1px solid var(--rule);
}
.first-ask { border-top: none; padding-top: 0; margin-top: 0.5rem; }

.reply { padding-left: 1.1rem; border-left: 3px solid var(--red); }
.reply .stMarkdown p,
.reply .stMarkdown li {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 1.08rem;
    line-height: 1.68;
    color: var(--ink);
}

/* ---- references ---- */
.refs { margin: 1rem 0 0 1.1rem; }
.refs .refs-head {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.refs ol { margin: 0; padding-left: 1.1rem; }
.refs li {
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--ink);
    margin-bottom: 0.4rem;
}
.refs a {
    color: var(--red-deep);
    text-decoration: none;
    border-bottom: 1px solid rgba(142,32,40,0.3);
}
.refs a:hover { border-bottom-color: var(--red-deep); }
.refs .rel { color: var(--ink-soft); font-size: 0.83rem; }

.timing { font-size: 0.82rem; color: var(--ink-soft); margin: 0.8rem 0 0 1.1rem; }

/* ---- input pinned at bottom ---- */
[data-testid="stBottomBlockContainer"], .stBottom, [data-testid="stBottom"] {
    background: var(--paper) !important;
    border-top: 1px solid var(--rule);
}
[data-testid="stBottomBlockContainer"] { max-width: 720px; padding-bottom: 1rem; }
[data-testid="stChatInput"] {
    background: var(--paper);
    border: 1.5px solid var(--rule);
    border-radius: 4px;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--red); }
[data-testid="stChatInput"] textarea { font-size: 1rem; color: var(--ink); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--ink-soft); }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--wash);
    border-right: 1px solid var(--rule);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; font-size: 0.88rem; }
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    border: 1px solid var(--rule);
    background: var(--paper);
    padding: 0.5rem;
}

/* ---- colophon ---- */
.colophon {
    margin-top: 4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--rule);
    font-size: 0.83rem;
    line-height: 1.55;
    color: var(--ink-soft);
    max-width: 60ch;
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

# PRESENTATION HELPERS

def answer_body(text):
    """The prompt asks the model to list its sources, and this page prints them
    as a reference list underneath. Trim the model's trailing copy so the same
    links don't appear twice."""
    for marker in ("\nSources:", "\nSource:", "\n**Sources", "\n**Source"):
        if marker in text:
            return text.split(marker)[0].rstrip()
    return text

def render_question(text, first=False):
    cls = "ask first-ask" if first else "ask"
    st.markdown(f'<div class="{cls}">{html.escape(text)}</div>', unsafe_allow_html=True)

def render_answer(text):
    st.markdown('<div class="reply">', unsafe_allow_html=True)
    st.markdown(answer_body(text))
    st.markdown('</div>', unsafe_allow_html=True)

def render_refs(sources, debug_mode):
    if not sources:
        return
    items = "".join(
        f'<li><a href="{html.escape(s["url"])}" target="_blank" rel="noopener">'
        f'{html.escape(s["title"])}</a>'
        + (f'<span class="rel"> · relevance {s["score"]:.2f}</span>' if debug_mode else "")
        + "</li>"
        for s in sources
    )
    st.markdown(
        f'<div class="refs"><div class="refs-head">Sources</div><ol>{items}</ol></div>',
        unsafe_allow_html=True,
    )
    if debug_mode:
        with st.expander("Retrieved passages"):
            for s in sources:
                st.code(s["content"][:400] + "...", language="text")

# INITIALIZATION

rag_system = initialize_rag_system()

if "query_times" not in st.session_state:
    st.session_state.query_times = []

if "messages" not in st.session_state:
    st.session_state.messages = []

def answer_and_store(question, debug_mode):
    try:
        answer, sources, debug_info = get_answer(question, rag_system, debug_mode=debug_mode)
        if debug_info and "timings" in debug_info:
            st.session_state.query_times.append(debug_info["timings"]["total"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timing": debug_info["timings"]["total"] if debug_info and "timings" in debug_info else 0,
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"The answering service could not be reached: {e}",
            "sources": [],
            "timing": 0,
        })

# SIDEBAR

with st.sidebar:
    st.markdown("**Technical panel**")
    debug_mode = st.toggle("Show retrieval diagnostics", value=False)

    if debug_mode:
        st.divider()
        st.write("Vector store:", "loaded" if rag_system["course_store"] else "missing")
        st.caption(f"Embedding · {CONFIG['retrieval']['embedding_model']}")
        st.caption(f"Reranker · {CONFIG['retrieval']['reranker_model']}")
        st.caption(f"Score threshold · {CONFIG['retrieval']['score_threshold']}")
        st.caption(f"Generation · {CONFIG['model']['name']}")

        if st.session_state.query_times:
            t = st.session_state.query_times
            st.divider()
            st.metric("Mean response", f"{sum(t)/len(t):.2f}s")
            st.caption(f"Fastest {min(t):.2f}s · slowest {max(t):.2f}s · {len(t)} queries")

        st.divider()
        if st.button("Clear cache"):
            st.cache_resource.clear()
            st.rerun()

    if st.button("Start again"):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.rerun()

# MASTHEAD

st.markdown(
    '<div class="mast"><div class="mast-in">'
    '<b>BristolBot</b><span>Student enquiries</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# OPENING SCREEN

if not st.session_state.messages:
    st.markdown('<div class="lede">What do you need to know?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="standfirst">Answers about admissions, tuition fees, scholarships, '
        'accommodation and University regulations, taken from published University of '
        'Bristol pages. Each answer lists the pages it came from.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rubric">Frequently asked</div>', unsafe_allow_html=True)

    for i, q in enumerate([
        "How much is the Cratchley Scholarship worth?",
        "Can I pay tuition fees in instalments?",
        "What are the accommodation fee payment dates?",
        "What is the pass mark for a Masters dissertation?",
    ]):
        if st.button(q, key=f"eg_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            answer_and_store(q, debug_mode)
            st.rerun()

# TRANSCRIPT

pair_index = 0
for message in st.session_state.messages:
    if message["role"] == "user":
        render_question(message["content"], first=(pair_index == 0))
        pair_index += 1
    else:
        render_answer(message["content"])
        render_refs(message.get("sources", []), debug_mode)
        if debug_mode and message.get("timing"):
            st.markdown(
                f'<div class="timing">Answered in {message["timing"]:.2f}s</div>',
                unsafe_allow_html=True,
            )

# INPUT

if user_input := st.chat_input("Ask about fees, scholarships, accommodation or regulations"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Searching University pages"):
        answer_and_store(user_input, debug_mode)
    st.rerun()

# COLOPHON

st.markdown(
    '<div class="colophon">An independent project, not affiliated with or endorsed by '
    'the University of Bristol. Answers are generated from published University web pages '
    'and may be out of date. Check the linked source before acting on anything.</div>',
    unsafe_allow_html=True,
)
