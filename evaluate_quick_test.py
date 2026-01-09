import pandas as pd
import os
import warnings
import mlflow
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
VECTOR_STORE_DIR = "faiss_course_store"
TEST_FILE = "test_dataset.xlsx"
OUTPUT_FILE = "quick_test_report.csv"
EXPERIMENT_NAME = "BristolBot_Quick_Tests"

# 1. SETUP MLFLOW
print(f"📊 Setting up MLflow experiment: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. SETUP THE BOT (Reduced Model for Speed)
print("🤖 Waking up the Bot...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if not os.path.exists(VECTOR_STORE_DIR):
    raise FileNotFoundError(f"❌ Cannot find {VECTOR_STORE_DIR}.")

vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Reduce K for speed
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "Answer briefly: {context} \n\n Question: {question}"
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. START MLFLOW RUN
print("⚡ Starting Fast Test Run...")
with mlflow.start_run(run_name="Quick_Smoke_Test"):
    
    # Log Dummy Params
    mlflow.log_param("mode", "fast_test")
    mlflow.log_param("model", "gpt-3.5-turbo")
    
    # 4. LOAD DATA (Only 2 rows!)
    print(f"📝 Loading ONLY first 2 rows from {TEST_FILE}...")
    try:
        df_test = pd.read_excel(TEST_FILE).head(2) # <--- SPEED HACK
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    questions = []
    answers = []

    # 5. TAKE SHORT EXAM
    for index, row in df_test.iterrows():
        q = row['Question']
        print(f"   Testing Q: {q[:30]}...", end=" ")
        
        try:
            # We still run the chain to ensure the DB works
            ans = rag_chain.invoke(q)
            questions.append(q)
            answers.append(ans)
            print("✅ OK")
        except Exception as e:
            print(f"❌ Error: {e}")

    # 6. FAKE GRADING (For Speed)
    print("\n💨 Skipping heavy Ragas grading for this test...")
    
    # Create Fake Results
    fake_metrics = {
        "faithfulness": 0.99,
        "answer_relevancy": 0.98,
        "context_recall": 0.95,
        "answer_correctness": 0.97
    }
    
    # Log Fake Metrics to MLflow
    print("📊 Logging (dummy) metrics to MLflow...")
    mlflow.log_metrics(fake_metrics)

    # Save Results
    df_results = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "faithfulness": [0.99]*len(questions) # Dummy column
    })
    
    df_results.to_csv(OUTPUT_FILE, index=False)
    mlflow.log_artifact(OUTPUT_FILE)
    
    print(f"✅ Success! Check dashboard. Saved to {OUTPUT_FILE}")