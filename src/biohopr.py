import os
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ── Authentication ───────────────────────────────────────────────────────────
login(token="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")

# ── Device Setup ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load Knowledge Graph ─────────────────────────────────────────────────────
# Convert CSV → Parquet on first run for 5-10x faster loading in future runs
KG_CSV_PATH = os.path.join("data", "kg.csv")
KG_PARQUET_PATH = os.path.join("data", "kg.parquet")

if os.path.exists(KG_PARQUET_PATH):
    print("Loading kg from parquet (fast path)...")
    kg_df = pd.read_parquet(KG_PARQUET_PATH)
else:
    print("Loading kg from CSV (first run — converting to parquet)...")
    kg_df = pd.read_csv(KG_CSV_PATH, low_memory=False)
    kg_df.to_parquet(KG_PARQUET_PATH, index=False)
    print(f"Saved parquet to {KG_PARQUET_PATH}")

print(f"Knowledge graph loaded: {kg_df.shape[0]} rows, {kg_df.shape[1]} cols")

# ── Load Benchmark Dataset ───────────────────────────────────────────────────
print("Loading BioHopR benchmark dataset...")
biohopr = load_dataset("knowlab-research/BioHopR")
benchmark_df = biohopr["train"].to_pandas()
print(f"Benchmark loaded: {benchmark_df.shape[0]} samples")

# ── Load LLM ─────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Loading tokenizer and model: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
if device == "cpu":
    model.to(device)

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("LLM loaded.")

# ── Load Embedding Model ─────────────────────────────────────────────────────
print("Loading sentence embedder (BioLORD-2023-C)...")
embedder = SentenceTransformer("FremyCompany/BioLORD-2023-C").to(device)
print("Embedder loaded.")


# ── Helper Functions ─────────────────────────────────────────────────────────
def generate_answer(question: str) -> str:
    """Generate an answer from the LLM given a question."""
    messages = [
        {"role": "system", "content": "Provide exactly one specific answer."},
        {"role": "user", "content": question},
    ]
    outputs = llm_pipeline(messages, max_new_tokens=50, do_sample=False)
    return outputs[0]["generated_text"][-1]["content"].strip()


def evaluate_instance(prediction: str, targets: list, threshold: float = 0.9):
    """Evaluate a prediction against target entities using cosine similarity."""
    if targets is None or len(targets) == 0:
        print("Warning: No target entities found for this sample.")
        return False, 0.0
    with torch.no_grad():
        pred_emb = embedder.encode(prediction, convert_to_tensor=True)
        target_embs = embedder.encode(targets, convert_to_tensor=True)
        if target_embs.dim() == 1:
            target_embs = target_embs.unsqueeze(0)
        cos_sims = F.cosine_similarity(pred_emb.unsqueeze(0), target_embs)
        max_sim = torch.max(cos_sims).item()
    return max_sim > threshold, max_sim


# ── Run on First Sample ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Print available columns for reference
    print(f"\nDataset columns: {list(benchmark_df.columns)}")

    sample_row = benchmark_df.iloc[0]
    question = sample_row.get("prompt") or sample_row.get("question_single")
    targets = sample_row.get("answer", [])
    if isinstance(targets, str):
        targets = [targets]

    print(f"\nQuestion: {question}")
    print(f"Targets: {targets}")

    prediction = generate_answer(question)
    is_correct, score = evaluate_instance(prediction, targets)

    print(f"Prediction: {prediction}")
    print(f"Correct: {is_correct} | Max Cosine Similarity: {score:.3f}")
