import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# --- CONFIGURATION ---
# 1. The "Academic" Data (Courses)
COURSE_FILE = "structured_course_data.json" 

# 2. The "Universe" Data (Your new 501-page scrape)
GENERAL_FILE = "general_knowledge_curated.json" 

# 3. The "Gold Standard" Manual Rules (Safety Net)
MANUAL_FILES = ["data_manual/finance_rules.json", "data_manual/campus_map.json"]

# 4. The PDF Folder (Deep Rules)
PDF_FOLDER = "data_pdfs"

# 5. Output
OUTPUT_DIR = "faiss_course_store" 
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- HELPER FUNCTIONS ---
def format_complex_field(data):
    if isinstance(data, dict):
        return "\n".join([f"- {k}: {v}" for k, v in data.items()])
    elif isinstance(data, list):
        return ", ".join(data)
    return str(data)

def create_course_document(entry):
    text = f"""
    COURSE TITLE: {entry.get('title')}
    TYPE: {entry.get('type')}
    LINK: {entry.get('url')}
    --- SUMMARY ---
    {entry.get('summary')}
    --- FEES ---
    {entry.get('fees')}
    --- ENTRY REQUIREMENTS ---
    {format_complex_field(entry.get('requirements'))}
    --- STRUCTURE ---
    {entry.get('structure')}
    --- CAREERS ---
    {format_complex_field(entry.get('careers'))}
    """
    metadata = {"title": entry.get('title'), "url": entry.get('url'), "category": "Course", "type": entry.get('type')}
    return Document(page_content=text, metadata=metadata)

def create_general_document(entry, is_manual=False):
    text = f"""
    TOPIC: {entry.get('title')}
    CATEGORY: {entry.get('category', 'General Info')}
    LINK: {entry.get('url')}
    --- CONTENT ---
    {entry.get('content')}
    """
    metadata = {
        "title": entry.get('title'), 
        "url": entry.get('url'), 
        "category": entry.get('category', "General"), 
        "type": "Manual Rule" if is_manual else "General Knowledge"
    }
    return Document(page_content=text, metadata=metadata)

def main():
    docs = []

    # 1. LOAD COURSES
    if os.path.exists(COURSE_FILE):
        print(f"📖 Loading Courses from {COURSE_FILE}...")
        with open(COURSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            docs.append(create_course_document(entry))
        print(f"   --> Loaded {len(data)} courses.")

    # 2. LOAD THE 501 SCRAPED PAGES
    if os.path.exists(GENERAL_FILE):
        print(f"🌍 Loading The Universe from {GENERAL_FILE}...")
        with open(GENERAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            docs.append(create_general_document(entry))
        print(f"   --> Loaded {len(data)} general pages.")
    else:
        print(f"⚠️ Warning: {GENERAL_FILE} not found!")

    # 3. LOAD MANUAL RULES (Priority Overrides)
    print("🛡️  Loading Manual Gold Standard Rules...")
    for manual_file in MANUAL_FILES:
        if os.path.exists(manual_file):
            with open(manual_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                docs.append(create_general_document(entry, is_manual=True))
            print(f"   --> Ingested manual facts from {manual_file}")

    # 4. LOAD PDFs (Deep Rules)
    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        print(f"📚 Loading {len(pdf_files)} PDFs...")
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, pdf))
                pages = loader.load()
                for page in pages:
                    page.metadata["source"] = pdf
                    page.metadata["category"] = "Official Document"
                docs.extend(pages)
            except Exception as e:
                print(f"   ❌ Error loading {pdf}: {e}")
        print(f"   --> PDFs Processed.")

    # 5. CHUNK & SAVE
    print(f"🧠 Fusion: Processing {len(docs)} total documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    print(f"🧩 Generating Embeddings ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(OUTPUT_DIR)
    
    print(f"🎉 Success! The Master Brain is ready in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()