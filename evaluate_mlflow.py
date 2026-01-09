import pandas as pd
import os
import warnings
import mlflow
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
OUTPUT_FILE = "report_card_mlflow.csv"
EXPERIMENT_NAME = "BristolBot_Experiments"

# 1. SETUP MLFLOW
print(f"📊 Setting up MLflow experiment: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. SETUP THE BOT
print("🤖 Waking up the Bot (LCEL Mode)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if not os.path.exists(VECTOR_STORE_DIR):
    raise FileNotFoundError(f"❌ Cannot find {VECTOR_STORE_DIR}.")

vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

# CONFIG VARS
RETRIEVER_K = 3
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
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
with mlflow.start_run():
    
    # --- LOG PARAMETERS ---
    print("📝 Logging parameters...")
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("retriever_k", RETRIEVER_K)
    
    # 4. LOAD EXAM
    print(f"📝 Loading Exam Questions from {TEST_FILE}...")
    try:
        df_test = pd.read_excel(TEST_FILE) 
        mlflow.log_artifact(TEST_FILE) 
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        exit()

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    # 5. TAKE EXAM
    print("🚀 Starting the Exam...")
    for index, row in df_test.iterrows():
        q = row['Question']
        truth = row['Ground_Truth']
        
        print(f"   [{index+1}/{len(df_test)}] Q: {q[:40]}...", end=" ")
        
        try:
            ans = rag_chain.invoke(q)
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
            answers.append("Error")
            contexts.append(["No context"])

    # 6. GRADE (ROBUST VERSION)
    print("\n👨‍🏫 Grading (This calls OpenAI)...")

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    metrics = [faithfulness, answer_relevancy, context_recall, answer_correctness]

    try:
        results = evaluate(
            dataset=dataset, 
            metrics=metrics, 
            llm=llm, 
            embeddings=embeddings
        )
        
        # --- SAFE SAVING BLOCK ---
        print("\n📊 Raw Results Object:")
        print(results) # Print to screen immediately so we see it

        # 1. Convert to Pandas first (Most reliable method)
        df_results = results.to_pandas()
        
        # 2. Save CSV to disk IMMEDIATELY
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 SAFEGUARD: Saved detailed results to {OUTPUT_FILE}")

        # 3. Upload CSV to MLflow
        mlflow.log_artifact(OUTPUT_FILE)

        # 4. Calculate Averages Manually (Avoids object errors)
        print("📊 Calculating averages for MLflow...")
        safe_metrics = {}
        for m in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
            if m in df_results.columns:
                safe_metrics[m] = df_results[m].mean()
            else:
                print(f"⚠️ Metric {m} not found in results.")

        # 5. Log Metrics
        mlflow.log_metrics(safe_metrics)
        print("✅ Success! Metrics logged.")

    except Exception as e:
        print(f"❌ Grading/Logging Failed: {e}")
        # Even if it fails, try to save whatever lists we have
        if 'answers' in locals() and len(answers) > 0:
             pd.DataFrame(data_dict).to_csv("emergency_dump.csv", index=False)
             print("⚠️ Saved emergency_dump.csv with your answers.")