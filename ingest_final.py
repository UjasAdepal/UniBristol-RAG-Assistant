import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
# 1. The "Academic" Data
COURSE_FILE = "structured_course_data.json" 

# 2. The "Student Life" Data (Your new massive file)
GENERAL_FILE = "general_knowledge_massive.json" 

# 3. The Output Database Folder
OUTPUT_DIR = "faiss_course_store" 
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- HELPER FUNCTIONS ---
def format_complex_field(data):
    """Turns dictionaries/lists into readable text."""
    if isinstance(data, dict):
        return "\n".join([f"- {k}: {v}" for k, v in data.items()])
    elif isinstance(data, list):
        return ", ".join(data)
    return str(data)

def create_course_document(entry):
    """Format structured course data into a document."""
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
    metadata = {
        "title": entry.get('title'),
        "url": entry.get('url'),
        "category": "Course",
        "type": entry.get('type')
    }
    return Document(page_content=text, metadata=metadata)

def create_general_document(entry):
    """Format unstructured general knowledge into a document."""
    # We add the 'Category' to the text so the AI knows the context
    # e.g., "CATEGORY: Accommodation" helps it answer "Where can I live?"
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
        "type": "General Knowledge"
    }
    return Document(page_content=text, metadata=metadata)

def main():
    docs = []

    # 1. Load Courses (The Academic Brain)
    if os.path.exists(COURSE_FILE):
        print(f"📖 Loading Courses from {COURSE_FILE}...")
        with open(COURSE_FILE, "r", encoding="utf-8") as f:
            course_data = json.load(f)
        for entry in course_data:
            docs.append(create_course_document(entry))
        print(f"   --> Loaded {len(course_data)} courses.")
    else:
        print(f"⚠️ Warning: {COURSE_FILE} not found. Skipping courses.")

    # 2. Load General Knowledge (The Student Life Brain)
    if os.path.exists(GENERAL_FILE):
        print(f"📖 Loading General Knowledge from {GENERAL_FILE}...")
        with open(GENERAL_FILE, "r", encoding="utf-8") as f:
            general_data = json.load(f)
        for entry in general_data:
            docs.append(create_general_document(entry))
        print(f"   --> Loaded {len(general_data)} general pages.")
    else:
        print(f"⚠️ Warning: {GENERAL_FILE} not found. Skipping general info.")

    # 3. Create the Database
    print(f"🧠 Fusion: Processing {len(docs)} total documents...")

    # We use a chunk size of 1500 to keep long accommodation rules together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ Created {len(split_docs)} chunks.")

    print(f"🧩 Generating Embeddings with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(OUTPUT_DIR)
    
    # Save pickle for debugging/inspection
    with open(os.path.join(OUTPUT_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(split_docs, f)

    print(f"🎉 Success! The Super Brain is saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()