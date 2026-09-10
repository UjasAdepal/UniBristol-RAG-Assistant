import os
import re
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
    layout="wide",
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
    --ink-soft: #6B6E73;
    --rule: #E2DED9;
    --paper: #FFFFFF;
    --wash: #F7F5F2;
}

.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }
html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* strip Streamlit chrome, including the sidebar entirely */
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

.block-container {
    max-width: 1000px;
    padding: 0 2rem 7rem 2rem !important;
}

/* ---------- masthead ---------- */
.mast {
    margin: 0 0 3.2rem 0;
    padding: 1.5rem 0 0.9rem 0;
    border-bottom: 3px solid var(--red);
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
}
.mast b {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--red-deep);
    letter-spacing: -0.02em;
}
.mast span { font-size: 0.9rem; color: var(--ink-soft); }

/* ---------- opening ---------- */
.lede {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.45rem;
    line-height: 1.16;
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.02em;
    max-width: 17ch;
    margin-bottom: 1.2rem;
}
.standfirst {
    font-size: 1.04rem;
    line-height: 1.62;
    color: var(--ink-soft);
    max-width: 54ch;
    margin-bottom: 3rem;
}
.rubric {
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--ink-soft);
    padding-bottom: 0.45rem;
    border-bottom: 2px solid var(--ink);
    max-width: 640px;
}

/* suggested questions as list rows */
div[data-testid="stButton"] > button {
    width: 100%;
    max-width: 640px;
    background: transparent;
    color: var(--ink);
    border: none;
    border-bottom: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.95rem 0.15rem;
    font-weight: 400;
    line-height: 1.4;
    display: block;
    text-align: left;
    transition: background 0.12s ease, padding-left 0.12s ease;
}
/* Streamlit wraps button text in its own <p>, which is what actually needs
   the alignment and size. */
div[data-testid="stButton"] > button p {
    text-align: left;
    font-size: 1.03rem;
    margin: 0;
}
div[data-testid="stButton"] > button:hover {
    background: var(--wash);
    color: var(--red-deep);
    padding-left: 0.65rem;
}
div[data-testid="stButton"] > button:active { background: var(--wash); }
/* don't let the clicked button keep its hover background */
div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:focus:not(:active) {
    background: transparent;
    color: var(--ink);
    box-shadow: none;
}
div[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--red);
    outline-offset: -2px;
}

/* ---------- the exchange: main column + sidenotes ---------- */
.xchg {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 230px;
    gap: 2.6rem;
    padding-top: 2.2rem;
    margin-top: 2.2rem;
    border-top: 1px solid var(--rule);
}
.xchg.opening { border-top: none; padding-top: 0; margin-top: 0; }

.ask {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.55rem;
    line-height: 1.28;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.012em;
    margin-bottom: 0.45rem;
}
.provenance {
    font-size: 0.82rem;
    color: var(--ink-soft);
    margin-bottom: 1.3rem;
}

.reply {
    border-left: 3px solid var(--red);
    padding-left: 1.4rem;
    max-width: 66ch;
}
.reply p, .reply li {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.09rem;
    line-height: 1.7;
    color: var(--ink);
}
.reply p { margin: 0 0 1rem 0; }
.reply p:last-child { margin-bottom: 0; }
.reply ul, .reply ol { margin: 0 0 1rem 0; padding-left: 1.3rem; }
.reply li { margin-bottom: 0.4rem; }
.reply a { color: var(--red-deep); }

/* sidenotes */
.notes { padding-top: 0.35rem; position: sticky; top: 1.5rem; align-self: start; }
.notes-head {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 0.85rem;
}
.notes ol { margin: 0; padding-left: 1.15rem; }
.notes li {
    font-size: 0.86rem;
    line-height: 1.45;
    margin-bottom: 0.9rem;
    padding-left: 0.15rem;
}
.notes li::marker { color: var(--red); font-weight: 700; font-size: 0.78rem; }
.notes a {
    color: var(--ink);
    text-decoration: none;
    border-bottom: 1px solid var(--rule);
}
.notes a:hover { color: var(--red-deep); border-bottom-color: var(--red-deep); }
.notes a:focus-visible, .reply a:focus-visible {
    outline: 2px solid var(--red);
    outline-offset: 2px;
}
.notes .rel { display: block; font-size: 0.75rem; color: var(--ink-soft); margin-top: 0.2rem; }

@media (max-width: 860px) {
    /* stack, and let the answer come before its sources */
    .xchg { grid-template-columns: 1fr; gap: 1.4rem; }
    .lede { font-size: 1.95rem; max-width: none; }
    .notes {
        position: static;
        border-top: 1px solid var(--rule);
        padding-top: 1rem;
    }
    .notes-head { border-bottom: none; padding-bottom: 0; }
    .reply { padding-left: 1rem; }
}

/* ---------- input ---------- */
[data-testid="stBottomBlockContainer"], .stBottom, [data-testid="stBottom"] {
    background: var(--paper) !important;
    border-top: 1px solid var(--rule);
}
[data-testid="stBottomBlockContainer"] { max-width: 1000px; padding-bottom: 1rem; }
[data-testid="stChatInput"] {
    background: var(--paper);
    border: 1.5px solid var(--rule);
    border-radius: 3px;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--red); }
[data-testid="stChatInput"] textarea { font-size: 1rem; color: var(--ink); }

/* ---------- diagnostics + colophon ---------- */
/* diagnostics: a quiet line near the foot of the page, not a panel */
[data-testid="stExpander"] {
    border: none !important;
    border-radius: 0;
    box-shadow: none;
    margin-top: 4.5rem;
    background: transparent;
}
[data-testid="stExpander"] details { border: none !important; background: transparent; }
[data-testid="stExpander"] summary {
    font-size: 0.8rem;
    color: var(--ink-soft);
    font-weight: 400;
    padding: 0;
}
[data-testid="stExpander"] summary:hover { color: var(--red-deep); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding-top: 0.9rem; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * { font-size: 0.8rem; }

.colophon {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--rule);
    font-size: 0.81rem;
    line-height: 1.55;
    color: var(--ink-soft);
    max-width: 62ch;
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

# PRESENTATION

def tidy_title(raw):
    """Scraped page titles carry the site's navigation trail, e.g.
    'Cratchley scholarship in history | Current students | University of Bristol'.
    Only the first segment names the page, so drop the rest. Falls back to the
    whole string if splitting would leave nothing useful."""
    first = str(raw).split("|")[0].strip()
    return first if len(first) > 3 else str(raw).strip()


def strip_model_sources(text):
    """The prompt asks the model to list its sources; this page shows them as
    sidenotes instead. Drop the model's trailing copy so links aren't duplicated."""
    for marker in ("\nSources:", "\nSource:", "\n**Sources", "\n**Source"):
        if marker in text:
            return text.split(marker)[0].rstrip()
    return text

def md_to_html(text):
    """Convert the subset of Markdown the model emits into HTML.

    st.markdown renders into its own container, so a styled wrapper <div> never
    applies to it. Emitting the answer as one HTML block is what lets the answer
    be typeset properly and sit in a grid beside its sidenotes. Input is escaped
    first, so model output cannot inject markup."""
    text = html.escape(text.strip())

    def inline(s):
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                   r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
        s = re.sub(r'(?<!["=>])\b(https?://[^\s<)]+)',
                   r'<a href="\1" target="_blank" rel="noopener">\1</a>', s)
        return s

    out, buffer, mode = [], [], None

    def flush():
        nonlocal buffer, mode
        if not buffer:
            return
        if mode == "ul":
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buffer) + "</ul>")
        elif mode == "ol":
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in buffer) + "</ol>")
        else:
            out.append(f"<p>{inline(' '.join(buffer))}</p>")
        buffer, mode = [], None

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        bullet = re.match(r"^[-*]\s+(.*)", line)
        number = re.match(r"^\d+[.)]\s+(.*)", line)
        if bullet:
            if mode != "ul":
                flush()
                mode = "ul"
            buffer.append(bullet.group(1))
        elif number:
            if mode != "ol":
                flush()
                mode = "ol"
            buffer.append(number.group(1))
        else:
            if mode in ("ul", "ol"):
                flush()
            mode = "p"
            buffer.append(line)
    flush()
    return "".join(out)

def render_exchange(question, answer, sources, opening, debug_mode):
    """One question, its answer, and its sources as a single HTML grid."""
    if sources:
        rows = []
        for s in sources:
            rel = ""
            if debug_mode:
                rel = '<span class="rel">relevance {:.2f}</span>'.format(s["score"])
            rows.append(
                '<li><a href="{}" target="_blank" rel="noopener">{}</a>{}</li>'.format(
                    html.escape(s["url"]), html.escape(tidy_title(s["title"])), rel
                )
            )
        aside = (
            '<div class="notes"><div class="notes-head">Sources</div><ol>'
            + "".join(rows)
            + "</ol></div>"
        )
        count = len(sources)
        provenance = "Drawn from {} University page{}".format(count, "" if count == 1 else "s")
    else:
        aside = '<div class="notes"></div>'
        provenance = "No matching University page found"

    st.markdown(
        f'<div class="xchg{" opening" if opening else ""}">'
        f'<div><div class="ask">{html.escape(question)}</div>'
        f'<div class="provenance">{provenance}</div>'
        f'<div class="reply">{md_to_html(strip_model_sources(answer))}</div></div>'
        f"{aside}</div>",
        unsafe_allow_html=True,
    )

# INITIALIZATION

rag_system = initialize_rag_system()

if "query_times" not in st.session_state:
    st.session_state.query_times = []
if "messages" not in st.session_state:
    st.session_state.messages = []

debug_mode = st.session_state.get("diagnostics", False)

def answer_and_store(question):
    try:
        answer, sources, debug_info = get_answer(question, rag_system, debug_mode=debug_mode)
        if debug_info and "timings" in debug_info:
            st.session_state.query_times.append(debug_info["timings"]["total"])
        st.session_state.messages.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timing": debug_info["timings"]["total"] if debug_info and "timings" in debug_info else 0,
            "retrieved": debug_info.get("total_retrieved") if debug_info else None,
        })
    except Exception as e:
        st.session_state.messages.append({
            "question": question,
            "answer": f"The answering service could not be reached: {e}",
            "sources": [],
            "timing": 0,
            "retrieved": None,
        })

# MASTHEAD

st.markdown(
    '<div class="mast"><b>BristolBot</b><span>Student enquiries</span></div>',
    unsafe_allow_html=True,
)

# OPENING

if not st.session_state.messages:
    st.markdown('<div class="lede">What do you need to know?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="standfirst">Answers about admissions, tuition fees, scholarships, '
        'accommodation and University regulations, taken from published University of '
        'Bristol pages. Every answer shows the pages it came from.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rubric">Frequently asked</div>', unsafe_allow_html=True)

    for i, q in enumerate([
        "How much is the Cratchley Scholarship worth?",
        "Can I pay tuition fees in instalments?",
        "What are the accommodation fee payment dates?",
        "What is the pass mark for a Masters dissertation?",
    ]):
        if st.button(q, key=f"eg_{i}"):
            answer_and_store(q)
            st.rerun()

# TRANSCRIPT

for i, m in enumerate(st.session_state.messages):
    render_exchange(m["question"], m["answer"], m["sources"], opening=(i == 0),
                    debug_mode=debug_mode)

# INPUT

if user_input := st.chat_input("Ask about fees, scholarships, accommodation or regulations"):
    with st.spinner("Searching University pages"):
        answer_and_store(user_input)
    st.rerun()

# DIAGNOSTICS

with st.expander("How this answer was produced"):
    st.toggle("Show retrieval scores and model details", key="diagnostics")

    if debug_mode:
        st.write("Vector store:", "loaded" if rag_system["course_store"] else "missing")
        st.caption(f"Embedding · {CONFIG['retrieval']['embedding_model']}")
        st.caption(f"Reranker · {CONFIG['retrieval']['reranker_model']}")
        st.caption(f"Score threshold · {CONFIG['retrieval']['score_threshold']}")
        st.caption(f"Generation · {CONFIG['model']['name']}")

        if st.session_state.query_times:
            t = st.session_state.query_times
            st.caption(
                f"Mean {sum(t)/len(t):.2f}s · fastest {min(t):.2f}s · "
                f"slowest {max(t):.2f}s · {len(t)} queries"
            )
        if st.session_state.messages:
            last = st.session_state.messages[-1]
            if last.get("retrieved") is not None:
                st.caption(
                    f"Last query retrieved {last['retrieved']} passages, "
                    f"kept {len(last['sources'])} above threshold, "
                    f"answered in {last['timing']:.2f}s"
                )

    if st.session_state.messages and st.button("Start again"):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.rerun()

st.markdown(
    '<div class="colophon">An independent project, not affiliated with or endorsed by '
    'the University of Bristol. Answers are generated from published University web pages '
    'and may be out of date. Check the linked source before acting on anything.</div>',
    unsafe_allow_html=True,
)
