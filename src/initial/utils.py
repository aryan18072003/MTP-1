"""
Shared utilities for the Knowledge Footprint pipeline.
Contains model loading, evaluation functions, config constants, and helpers.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login

# ── Path Setup (relative to src/, one level up from initial/) ────────────────
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config Constants ─────────────────────────────────────────────────────────
HF_TOKEN = "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"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDER_NAME = "FremyCompany/BioLORD-2023-C"
KG_CSV_PATH = os.path.join(SRC_DIR, "data", "kg.csv")
KG_PARQUET_PATH = os.path.join(SRC_DIR, "data", "kg.parquet")
RESULTS_DIR = os.path.join(SRC_DIR, "results")
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results.csv")
COSINE_THRESHOLD = 0.9

# BioHopR focuses on these 4 node types
BIOHOPR_NODE_TYPES = ["drug", "disease", "gene/protein", "effect/phenotype"]


def setup_device():
    """Detect and return the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def authenticate_hf():
    """Login to Hugging Face Hub."""
    login(token=HF_TOKEN)


def load_kg():
    """Load PrimeKG, converting CSV to Parquet on first run for speed."""
    if os.path.exists(KG_PARQUET_PATH):
        print("Loading KG from parquet (fast path)...")
        kg_df = pd.read_parquet(KG_PARQUET_PATH)
    else:
        print("Loading KG from CSV (first run — converting to parquet)...")
        kg_df = pd.read_csv(KG_CSV_PATH, low_memory=False)
        kg_df.to_parquet(KG_PARQUET_PATH, index=False)
        print(f"Saved parquet to {KG_PARQUET_PATH}")
    print(f"Knowledge graph loaded: {kg_df.shape[0]:,} rows, {kg_df.shape[1]} cols")
    return kg_df


def load_biohopr():
    """Load the BioHopR benchmark dataset."""
    print("Loading BioHopR benchmark dataset...")
    biohopr = load_dataset("knowlab-research/BioHopR")
    benchmark_df = biohopr["train"].to_pandas()
    print(f"Benchmark loaded: {benchmark_df.shape[0]:,} samples")
    return benchmark_df


def load_llm(device):
    """Load the LLM and return (pipeline, tokenizer, model)."""
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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

    # Set pad token to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("LLM loaded.")
    return llm_pipe


def load_embedder(device):
    """Load the sentence embedding model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence embedder ({EMBEDDER_NAME})...")
    embedder = SentenceTransformer(EMBEDDER_NAME).to(device)
    print("Embedder loaded.")
    return embedder


def generate_answer(llm_pipe, question: str) -> str:
    """Generate an answer from the LLM given a question."""
    messages = [
        {"role": "system", "content": "Provide exactly one specific answer."},
        {"role": "user", "content": question},
    ]
    outputs = llm_pipe(messages, max_new_tokens=50, do_sample=False)
    return outputs[0]["generated_text"][-1]["content"].strip()


def evaluate_instance(embedder, prediction: str, targets, threshold: float = COSINE_THRESHOLD):
    """
    Evaluate a prediction against target entities using cosine similarity.
    Returns (is_correct: bool, max_similarity: float).
    """
    # Handle various target formats
    if targets is None:
        return False, 0.0
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()
    if len(targets) == 0:
        return False, 0.0

    with torch.no_grad():
        pred_emb = embedder.encode(prediction, convert_to_tensor=True)
        target_embs = embedder.encode(targets, convert_to_tensor=True)
        if target_embs.dim() == 1:
            target_embs = target_embs.unsqueeze(0)
        cos_sims = F.cosine_similarity(pred_emb.unsqueeze(0), target_embs)
        max_sim = torch.max(cos_sims).item()
    return max_sim > threshold, max_sim


def ensure_results_dir():
    """Create the results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
