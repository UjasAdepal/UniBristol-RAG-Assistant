import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
INPUT_FILE = "structured_course_data.json" # The file you are about to generate
OUTPUT_DIR = "faiss_course_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def format_requirements(reqs):
    """
    Turns the requirements dictionary into a readable list.
    """
    if isinstance(reqs, dict):
        # formatted: "A-level: AAA; BTEC: DDD..."
        return "\n".join([f"- {k}: {v}" for k, v in reqs.items()])
    return str(reqs)

def format_list(data):
    """
    Turns a list (like careers) into a comma-separated string.
    """
    if isinstance(data, list):
        return ", ".join(data)
    return str(data)

def create_document(entry):
    title = entry.get("title", "Unknown Course")
    url = entry.get("url", "")
    type_ = entry.get("type", "Course")
    
    # 1. Clean Fields
    summary = entry.get("summary", "No summary available.")
    fees = entry.get("fees", "Check website")
    structure = entry.get("structure", "Check website")
    
    # 2. Smart Formatting for Complex Fields
    requirements = format_requirements(entry.get("requirements", "Check website"))
    careers = format_list(entry.get("careers", "Not specified"))
    
    # 3. Create the "Blob" for the AI to read
    # We add explicit headers so the Reranker knows what section is what.
    page_content = f"""
    COURSE TITLE: {title}
    TYPE: {type_}
    LINK: {url}
    
    --- SUMMARY ---
    {summary}
    
    --- TUITION FEES ---
    {fees}
    
    --- ENTRY REQUIREMENTS ---
    {requirements}
    
    --- COURSE STRUCTURE ---
    {structure}
    
    --- CAREER PROSPECTS ---
    {careers}
    """
    
    # 4. Rich Metadata (Helps with filtering later)
    metadata = {
        "title": title,
        "url": url,
        "type": type_,
        "topics": ["courses", "fees", "admissions", "careers"]
    }
    
    return Document(page_content=page_content, metadata=metadata)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File {INPUT_FILE} not found. Run structure_data.py first!")
        return

    print(f"📖 Loading structured data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🧠 Processing {len(data)} courses...")
    docs = []
    for entry in data:
        doc = create_document(entry)
        docs.append(doc)

    # Splitter
    # We use a larger chunk size (1200) because your data is now dense and high-quality.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    print(f"✅ Created {len(split_docs)} chunks.")

    print(f"🧩 Generating Embeddings with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectorstore.save_local(OUTPUT_DIR)
    
    # Save pickle for debugging
    with open(os.path.join(OUTPUT_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(split_docs, f)

    print(f"🎉 Success! Database saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()