import pandas as pd
import os
import warnings
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
VECTOR_STORE_DIR = "faiss_course_store"  
TEST_FILE = "test_dataset.xlsx"          
OUTPUT_FILE = "report_card.csv"          

# 1. SETUP THE BOT (Modern LCEL Way)
print("🤖 Waking up the Bot (LCEL Mode)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if not os.path.exists(VECTOR_STORE_DIR):
    raise FileNotFoundError(f"❌ Cannot find {VECTOR_STORE_DIR}.")

vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# The Bot's Brain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the Prompt Template (Standard RAG)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Build the Chain (This replaces RetrievalQA)
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 2. LOAD EXAM
print(f"📝 Loading Exam Questions from {TEST_FILE}...")
try:
    df_test = pd.read_excel(TEST_FILE)
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")
    exit()

questions = []
ground_truths = []
answers = []
contexts = []

# 3. TAKE EXAM
print("🚀 Starting the Exam...")
for index, row in df_test.iterrows():
    q = row['Question']
    truth = row['Ground_Truth']
    
    print(f"   [{index+1}/{len(df_test)}] Q: {q[:40]}...", end=" ")
    
    try:
        # Ask the bot (New Method)
        ans = rag_chain.invoke(q)
        
        # Manually fetch context for grading (since chain returns string only)
        retrieved_docs = retriever.invoke(q)
        source_docs = [doc.page_content for doc in retrieved_docs]
        
        questions.append(q)
        ground_truths.append(truth)
        answers.append(ans)
        contexts.append(source_docs)
        print("✅")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        questions.append(q)
        ground_truths.append(truth)
        answers.append("Error generating answer")
        contexts.append(["No context"])

# 4. GRADE
print("\n👨‍🏫 Grading (This calls OpenAI)...")

data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

metrics = [
    faithfulness,
    answer_relevancy, 
    context_recall,
    answer_correctness
]

try:
    results = evaluate(dataset=dataset, metrics=metrics)
    
    # 5. REPORT
    print("\n📊 FINAL REPORT CARD:")
    print(results)

    df_results = results.to_pandas()
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"❌ Grading Failed: {e}")
    print("Ensure OPENAI_API_KEY is set in your environment.")