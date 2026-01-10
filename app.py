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
from langchain_core.runnables import RunnableLambda
from flashrank import Ranker, RerankRequest

# CONFIGURATION
load_dotenv()

# Prevent tokenizer deadlocks in Docker
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Bristol AI Tutor", layout="centered")

# CONSTANTS 
COURSE_VECTOR_PATH = "./faiss_course_store"
FAQ_VECTOR_PATH = "./faiss_faq_store"
FEEDBACK_FILE = "feedback_log.csv"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# CACHED RESOURCES

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_reranker():
    return Ranker(model_name=RERANK_MODEL, cache_dir="./opt")

def load_vectorstore(path):
    if not os.path.exists(path):
        return None
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)

# BACKEND LOGIC

def rerank_docs(query, docs):
    if not docs: return []
    
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
    faq_store = load_vectorstore(FAQ_VECTOR_PATH)
    course_store = load_vectorstore(COURSE_VECTOR_PATH)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    template = """
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
    """
    prompt = PromptTemplate.from_template(template)

    def run_hybrid(inputs):
        question = inputs["question"]
        topics = inputs.get("topics", [])

        # 1. FAQ Search
        faq_docs = []
        if faq_store:
            search_kwargs = {"k": 10}
            if topics:
                search_kwargs["filter"] = {"topics": {"$in": topics}}
            faq_docs = faq_store.similarity_search(question, **search_kwargs)

        # 2. Course Search
        course_docs = []
        if course_store:
            course_docs = course_store.similarity_search(question, k=10)

        # 3. Rerank
        all_retrieved = faq_docs + course_docs
        best_docs = rerank_docs(question, all_retrieved)

        # 4. Generate
        context_text = "\n\n".join([d.page_content for d in best_docs])
        response = llm.invoke(prompt.format(context=context_text, question=question))
        
        # Format sources for UI
        source_data = []
        for doc in best_docs:
            source_data.append({
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "#"),
                "content": doc.page_content,
                "score": doc.metadata.get("score", 0)
            })

        return response.content, source_data

    return RunnableLambda(run_hybrid)

def save_feedback(query, response, is_helpful):
    """Log feedback to CSV for future analysis."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Response", "Helpful"])
        writer.writerow([datetime.datetime.now(), query, response, "Yes" if is_helpful else "No"])

# FRONTEND UI 

st.title("🎓 Bristol AI Tutor")
st.caption("Ask anything about the University of Bristol — from admissions to accommodation!")

# Topic Filtering
TOPIC_OPTIONS = [
    "Undergraduate", "Postgraduate", "Student Support", "Accommodation", "International Students",
    "Scholarships", "Fees & Funding", "Library & IT", "Vet School", "Admissions", "Research"
]
selected_topics = st.multiselect("📂 Filter by topic (optional):", TOPIC_OPTIONS)

query = st.text_input("💬 Ask your question here:")

if query:
    with st.spinner("Thinking..."):
        try:
            chain = get_qa_chain()
            response, sources = chain.invoke({"question": query, "topics": selected_topics})

            st.subheader("📘 Answer")
            st.markdown(response)

            # View Sources
            with st.expander("🔍 View Retrieved Sources & Context"):
                if not sources:
                    st.warning("No relevant sources found.")
                else:
                    for i, doc in enumerate(sources):
                        st.markdown(f"### Source {i+1}: {doc['title']}")
                        st.caption(f"Relevance Score: {doc['score']:.4f}")
                        st.code(doc['content'], language="text")
                        st.divider()

            if sources:
                with st.expander("🔗 Sources used"):
                    seen = set()
                    for s in sources:
                        if s['source'] not in seen:
                            st.markdown(f"- [{s['title']}]({s['source']})")
                            seen.add(s['source'])

            # Feedback Section
            st.subheader("👍 Was this helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes"):
                    save_feedback(query, response, True)
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No"):
                    save_feedback(query, response, False)
                    st.warning("Sorry! We'll try to improve.")

        except Exception as e:
            st.error(f"An error occurred: {e}")