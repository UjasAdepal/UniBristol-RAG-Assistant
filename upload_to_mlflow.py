import mlflow
import json
import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "latest_experiment_result.json"
EXPERIMENT_NAME = "BristolBot_Experiments"

def upload_results():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find {INPUT_FILE}. Run 'run_test.py' first.")
        return

    print(f"📂 Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Extract sections
    config = data["configuration"]
    raw_metrics = data["metrics"]
    details = data["detailed_results"]
    timestamp = data["metadata"]["timestamp"]

    print(f"📊 Connecting to MLflow Experiment: {EXPERIMENT_NAME}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"Run_{timestamp}"):
        
        # 1. LOG PARAMETERS
        print("📝 Logging Parameters...")
        
        # Model
        mlflow.log_param("model.name", config["model"]["name"])
        mlflow.log_param("model.temperature", config["model"]["temperature"])
        
        # Retrieval
        mlflow.log_param("retrieval.course_store", config["retrieval"].get("course_store", "N/A"))
        mlflow.log_param("retrieval.faq_store", config["retrieval"].get("faq_store", "N/A"))
        mlflow.log_param("retrieval.reranker", config["retrieval"].get("reranker_model", "None"))
        mlflow.log_param("retrieval.initial_k", config["retrieval"].get("initial_k", 0))
        
        # Prompt
        mlflow.log_param("prompt.template", config["prompt_template"])
        
        # 2. LOG METRICS (Sanitized)
        print("📈 Logging Metrics...")
        clean_metrics = {}
        
        # CRITICAL FIX: Loop through and fix any lists/weird formats
        for key, value in raw_metrics.items():
            try:
                if isinstance(value, list):
                    # If it's a list like [0.8], take the first number
                    clean_val = float(value[0]) if len(value) > 0 else 0.0
                else:
                    clean_val = float(value)
                clean_metrics[key] = clean_val
            except Exception as e:
                print(f"⚠️ Warning: Could not process metric {key}: {e}")

        mlflow.log_metrics(clean_metrics)

        # 3. LOG ARTIFACTS
        print("fw Logging Artifacts...")
        
        df_details = pd.DataFrame(details)
        csv_filename = "detailed_report_card.csv"
        df_details.to_csv(csv_filename, index=False)
        
        mlflow.log_artifact(csv_filename)
        mlflow.log_dict(config, "config_snapshot.json")
        
        print("✅ Success! Experiment uploaded to MLflow.")

if __name__ == "__main__":
    upload_results()