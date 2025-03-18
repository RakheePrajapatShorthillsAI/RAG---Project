import pandas as pd
import re
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load  # For BLEU score
import numpy as np
from tqdm import tqdm
from mainRAG import qa_pipeline

# Configuration
LOG_FILE = "qa_interactions.log"
RESULTS_FILE = "evaluation_results_100.csv"
TEST_CASES_FILE = "generated_test_cases_new.txt"
BATCH_SIZE = 100  # Process 100 test cases at a time

# Initialize models and metrics
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_metric = load("bleu")

def setup_logger():
    """Initialize logging file with header"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,context,question,generated_answer,reference_answer,BLEU,ROUGE-1,ROUGE-2,ROUGE-L,Cosine,final_score\n")

def log_interaction(context, question, generated, reference, metrics):
    """Log complete interaction with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "context": context,
        "question": question,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_metrics(generated, reference):
    """Calculate all evaluation metrics"""
    # BLEU Score
    bleu_score = bleu_metric.compute(predictions=[generated], references=[[reference]])["bleu"]

    # Cosine similarity
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

    # ROUGE scores
    rouge_scores = rouge.score(reference, generated)
    rouge_1 = rouge_scores['rouge1'].fmeasure
    rouge_2 = rouge_scores['rouge2'].fmeasure
    rouge_L = rouge_scores['rougeL'].fmeasure

    # Final score with weights
    final_score = (
        bleu_score * 0.15 +
        rouge_1 * 0.15 +
        rouge_2 * 0.15 +
        rouge_L * 0.15 +
        cosine_sim * 0.40
    )

    return {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_1,
        "ROUGE-2": rouge_2,
        "ROUGE-L": rouge_L,
        "Cosine": float(cosine_sim),
        "final_score": final_score
    }

def load_test_cases(file_path, start, end):
    """Load test cases for a specific batch"""
    test_cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    test_cases_raw = re.split(r"\d+\.\s*Context:", data)[start:end+1]

    for case in test_cases_raw:
        parts = re.split(r"\nQ:\s*", case)
        if len(parts) < 2:
            continue

        context = parts[0].strip()
        qa_pair = re.split(r"\nA:\s*", parts[1])

        if len(qa_pair) < 2:
            continue

        question = qa_pair[0].strip()
        reference = qa_pair[1].strip()
        
        test_cases.append({
            "context": context,
            "question": question,
            "reference_answer": reference
        })
    
    return pd.DataFrame(test_cases)



def process_test_cases(start, end):
    """Process test cases in batches"""
    setup_logger()
    df = load_test_cases(TEST_CASES_FILE, start, end)

    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=[
            "question", "context", "generated_answer", "reference_answer",
            "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Cosine", "final_score"
        ]).to_csv(RESULTS_FILE, index=False)

    pbar = tqdm(total=len(df), desc=f"Processing test cases {start}-{end}")

    for idx, row in df.iterrows():
        try:
            # Generate answer
            generated = qa_pipeline(row["question"])
            
            # Calculate all metrics
            metrics = calculate_metrics(
                generated=generated,
                reference=row["reference_answer"]
            )
            
            # Log interaction
            log_interaction(
                context=row["context"],
                question=row["question"],
                generated=generated,
                reference=row["reference_answer"],
                metrics=metrics
            )
            
            # Save to CSV (appending results)
            result_row = pd.DataFrame([{
                "question": row["question"],
                "context": row["context"],
                "generated_answer": generated,
                "reference_answer": row["reference_answer"],
                **metrics
            }])
            
            result_row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
            
            pbar.update(1)
            pbar.set_postfix({
                "Processed": f"{start+idx}/{end}",
                "Score": f"{metrics['final_score']:.2f}"
            })
            
        except Exception as e:
            error_msg = f"Error processing case {start+idx}: {str(e)}"
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "context": row.get("context", ""),
                    "question": row.get("question", "")
                }) + "\n")
            continue

    pbar.close()
    print(f"\nBatch {start}-{end} complete! Results in {RESULTS_FILE}, logs in {LOG_FILE}")

if __name__ == "__main__":
    total_queries = 1100  # Total test cases
    batch_size = 100

    for start in range(928, total_queries + 1, batch_size):
        end = min(start + batch_size - 1, total_queries)
        process_test_cases(start, end)
    
    # Display summary
    final_df = pd.read_csv(RESULTS_FILE)
    print("\nFinal Metrics Summary:")
    print(final_df[['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Cosine', 'final_score']].mean())
