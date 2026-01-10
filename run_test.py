import pandas as pd
import json
import os
import warnings
from datetime import datetime
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness

# backend imports
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from flashrank import Ranker, RerankRequest 
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# disable parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# main config
CONFIG = {
    "experiment_name": "BristolBot_Redesign_Hybrid",
    "model": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.1, 
        "provider": "OpenAI"
    },
    "retrieval": {
        "course_store": "./faiss_course_store",
        "faq_store": "./faiss_faq_store", 
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "reranker_model": "ms-marco-MiniLM-L-12-v2", 
        "initial_k": 10, 
        "final_k": 5     
    },
    "data": {
        "test_file": "test_dataset.xlsx",
        "output_json": "latest_experiment_result.json"
    },
    "prompt_template": """You are an expert academic advisor for the University of Bristol.
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
}

# helper for reranking
def get_reranker():
    return Ranker(model_name=CONFIG["retrieval"]["reranker_model"], cache_dir="./opt")

def rerank_docs(query, docs, ranker):
    if not docs:
        return []
    
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(docs)
    ]
    
    # run reranking
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    sorted_docs = []
    for res in results:
        if res['score'] > 0.001: 
            doc = docs[int(res['id'])]
            doc.metadata["score"] = res['score']
            sorted_docs.append(doc)
            
    return sorted_docs[:CONFIG["retrieval"]["final_k"]]

def run_experiment():
    print("Initializing pipeline...")
    
    # load embedding model and reranker
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["retrieval"]["embedding_model"])
    ranker = get_reranker()
    
    # try loading vector stores
    course_store = None
    if os.path.exists(CONFIG["retrieval"]["course_store"]):
        course_store = FAISS.load_local(CONFIG["retrieval"]["course_store"], embeddings, allow_dangerous_deserialization=True)
    
    faq_store = None
    if os.path.exists(CONFIG["retrieval"]["faq_store"]):
        faq_store = FAISS.load_local(CONFIG["retrieval"]["faq_store"], embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        model_name=CONFIG["model"]["name"], 
        temperature=CONFIG["model"]["temperature"]
    )
    
    prompt = PromptTemplate.from_template(CONFIG["prompt_template"])

    print(f"Loading test data: {CONFIG['data']['test_file']}...")
    try:
        df_test = pd.read_excel(CONFIG['data']['test_file'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    questions, ground_truths, answers, contexts = [], [], [], []

    print("Starting evaluation...")
    for index, row in df_test.iterrows():
        q = row['Question']
        truth = row['Ground_Truth']
        
        print(f"[{index+1}/{len(df_test)}] Q: {q[:30]}...", end=" ")
        try:
            # retrieval logic
            faq_docs = []
            if faq_store:
                faq_docs = faq_store.similarity_search(q, k=CONFIG["retrieval"]["initial_k"])

            course_docs = []
            if course_store:
                course_docs = course_store.similarity_search(q, k=CONFIG["retrieval"]["initial_k"])

            all_retrieved = faq_docs + course_docs
            best_docs = rerank_docs(q, all_retrieved, ranker)

            # generate answer
            context_text = "\n\n".join([d.page_content for d in best_docs])
            chain = prompt | llm
            ans_message = chain.invoke({"context": context_text, "question": q})
            ans_text = ans_message.content

            # store results
            questions.append(q)
            ground_truths.append(truth)
            answers.append(ans_text)
            contexts.append([d.page_content for d in best_docs]) 
            print("Done")
            
        except Exception as e:
            print(f"Error: {e}")
            questions.append(q)
            ground_truths.append(truth)
            answers.append("Error")
            contexts.append(["No context"])

    # ragas evaluation
    print("\nGrading results...")
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    ragas_metrics = [faithfulness, answer_relevancy, context_recall, answer_correctness]
    
    try:
        # passing llm/embeddings explicitly to avoid connection issues
        results = evaluate(dataset=dataset, metrics=ragas_metrics, llm=llm, embeddings=embeddings)
        print("\nScores:", results)
        
        # package results
        
        # manual extraction because ragas dict conversion can be buggy
        metrics_dict = {}
        for m in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
            try:
                metrics_dict[m] = results[m] 
            except Exception:
                print(f"Warning: missing score for {m}")
                metrics_dict[m] = 0

        # try converting details to list of dicts
        detailed_records = []
        try:
            detailed_records = results.to_pandas().to_dict(orient="records")
        except Exception as e:
            print(f"Could not save detailed breakdown ({e}). Saving aggregates only.")
            # fallback to manual data
            detailed_records = [
                {"question": q, "answer": a, "ground_truth": t} 
                for q, a, t in zip(questions, answers, ground_truths)
            ]

        full_package = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Success",
                "logic": "Hybrid (Course+FAQ) + Rerank"
            },
            "configuration": CONFIG,
            "metrics": metrics_dict,
            "detailed_results": detailed_records
        }
        
        # save to disk
        with open(CONFIG["data"]["output_json"], "w") as f:
            json.dump(full_package, f, indent=4)
            
        print(f"\nDone. Results saved to {CONFIG['data']['output_json']}")
        print("Run 'python upload_to_mlflow.py' to visualize.")
        
    except Exception as e:
        print(f"Grading failed: {e}")
        # emergency save
        if len(answers) > 0:
            emergency_df = pd.DataFrame(data_dict)
            emergency_df.to_csv("emergency_results.csv", index=False)
            print("Saved raw answers to 'emergency_results.csv'")

if __name__ == "__main__":
    run_experiment()