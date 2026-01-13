import pandas as pd
import json
import os
import warnings
import numpy as np 
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
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

# FIX JSON SERIALIZATION ERROR 
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

#  MAIN CONFIGURATION 
CONFIG = {
    "experiment_name": "BristolBot_Experiments",
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
        "final_k": 5,
        "score_threshold": 0.40  
    },
    "data": {
        "test_file": "test_dataset.xlsx",
        "output_json": "latest_experiment_result.json"
    },
    "prompt_template": """You are an expert academic advisor for the University of Bristol.
Use the provided context to answer the student's question accurately.

Context:
{context}

CRITICAL RULES:
1. **Numbers over Words:** If the text contains specific thresholds (e.g., "85%", "70%"), prioritize them over general statements.
2. **Conditional Logic:** If rules change by year (e.g., "pre-2024" vs "post-2024"), YOU MUST STATE BOTH. Do not guess.
3. **Closed World Assumption:** If a payment method or course is NOT listed in the text, explicitly state that it is "Not accepted" or "Not available".
4. **Citations:** Answer first, then list the source names used.

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
    
    # EXPERIMENT CHANGE: STRICT FILTERING
    # Retrieve the threshold from CONFIG
    threshold = CONFIG["retrieval"]["score_threshold"]
    
    for res in results:
        # Filter based on the config threshold
        if res['score'] > threshold:
            doc = docs[int(res['id'])]
            doc.metadata["score"] = res['score']
            sorted_docs.append(doc)
            
    # If we filtered EVERYTHING out, keep the best one (only if it has at least some relevance, > 0.20)
    if not sorted_docs and results:
        if results[0]['score'] > 0.20:
            best_doc = docs[int(results[0]['id'])]
            best_doc.metadata["score"] = results[0]['score']
            sorted_docs.append(best_doc)
            
    return sorted_docs[:CONFIG["retrieval"]["final_k"]]

def run_experiment():
    print("Initializing pipeline")
    
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

    # Initialize lists
    questions, ground_truths, answers, contexts, retrieval_scores = [], [], [], [], []
    
    print("Starting evaluation")
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
            
            # extract scores for debugging
            scores = [d.metadata.get("score", 0) for d in best_docs]
            
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
            retrieval_scores.append(scores) 
            print("Done")
            
        except Exception as e:
            print(f"Error: {e}")
            questions.append(q)
            ground_truths.append(truth)
            answers.append("Error")
            contexts.append(["No context"])
            retrieval_scores.append([])

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
            # Inject scores
            for i, record in enumerate(detailed_records):
                if i < len(retrieval_scores):
                    record["retrieved_scores"] = retrieval_scores[i]
        except Exception as e:
            print(f"Could not save detailed breakdown ({e}). Saving aggregates only.")
            
            # fallback to manual data
            detailed_records = [
                {
                    "question": q, 
                    "answer": a, 
                    "ground_truth": t,
                    "retrieved_scores": s
                }
                for q, a, t, s in zip(questions, answers, ground_truths, retrieval_scores)
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
        
        # save to disk using NumpyEncoder to fix the crash
        with open(CONFIG["data"]["output_json"], "w") as f:
            json.dump(full_package, f, indent=4, cls=NumpyEncoder)
            
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