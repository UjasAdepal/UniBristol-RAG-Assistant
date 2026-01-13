"""
Shared Configuration for BristolBot
All files (app.py, backend.py, run_test.py) import from here
"""

CONFIG = {
    "experiment_name": "BristolBot_Production",
    
    "model": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.1,
        "provider": "OpenAI"
    },
    
    "retrieval": {
        "course_store": "./faiss_course_store",
        "faq_store": None,  # ✅ FIXED: No separate FAQ store
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "reranker_model": "ms-marco-MiniLM-L-12-v2",
        "initial_k": 10,
        "final_k": 5,
        "score_threshold": 0.40  # Strict filtering to prevent hallucinations
    },
    
    "data": {
        "test_file": "test_dataset.xlsx",
        "output_json": "latest_experiment_result.json",
        "feedback_file": "feedback_log.csv"
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

# Paths for easy access
PATHS = {
    "course_store": CONFIG["retrieval"]["course_store"],
    "faq_store": CONFIG["retrieval"]["faq_store"],  # Will be None
    "feedback_file": CONFIG["data"]["feedback_file"]
}