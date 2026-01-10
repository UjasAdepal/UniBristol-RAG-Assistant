from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the Brain
print("Loading database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("./faiss_course_store", embeddings, allow_dangerous_deserialization=True)

# 2. Search for the problem
query = "MSc Data Science"
docs = vectorstore.similarity_search(query, k=1)

# 3. Show me the EXACT text the bot sees
print("\n--- WHAT THE BOT SEES ---")
print(docs[0].page_content)
print("-------------------------")