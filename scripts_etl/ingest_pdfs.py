import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
PDF_FOLDER = "data_pdfs"
OUTPUT_DIR = "faiss_course_store" # We append to the existing brain
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def main():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder '{PDF_FOLDER}' not found. Please create it and add PDFs.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ No PDFs found in {PDF_FOLDER}.")
        return

    print(f"📚 Found {len(pdf_files)} PDFs. Processing...")
    
    all_pages = []
    
    for pdf_file in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"   - Reading: {pdf_file}...", end=" ")
        
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            
            # Add metadata so the bot knows where this came from
            for page in pages:
                page.metadata["source"] = pdf_file
                page.metadata["category"] = "Official Handbook"
            
            all_pages.extend(pages)
            print(f"✅ ({len(pages)} pages)")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Chunking (PDFs are dense, so we need smaller chunks to be precise)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_pages)
    
    print(f"📄 Created {len(split_docs)} text chunks from PDFs.")

    # Load Existing Vector Store (if it exists) to append to it
    print(f"🧠 Merging into {OUTPUT_DIR}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    if os.path.exists(OUTPUT_DIR):
        vectorstore = FAISS.load_local(OUTPUT_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(split_docs)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save Updated Brain
    vectorstore.save_local(OUTPUT_DIR)
    print("🎉 Success! PDFs have been added to the brain.")

if __name__ == "__main__":
    main()