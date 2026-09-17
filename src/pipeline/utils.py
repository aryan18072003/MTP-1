import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
EMBEDDER_NAME = "FremyCompany/BioLORD-2023-C"
KG_CSV_PATH = os.path.join(SRC_DIR, "data", "kg.csv")
KG_PARQUET_PATH = os.path.join(SRC_DIR, "data", "kg.parquet")
RESULTS_DIR = os.path.join(SRC_DIR, "results")
COSINE_THRESHOLD = 0.9

def get_dataset_dir(dataset_name):
    # "knowlab-research/BioHopR" -> "BioHopR"
    short_name = dataset_name.split("/")[-1]
    path = os.path.join(RESULTS_DIR, short_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_model_dir(dataset_name, model_name):
    # "meta-llama/Llama-3.2-3B-Instruct" -> "Llama-3.2-3B-Instruct"
    d_path = get_dataset_dir(dataset_name)
    m_name = model_name.split("/")[-1]
    path = os.path.join(d_path, m_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_eval_results_path(dataset_name, model_name):
    return os.path.join(get_model_dir(dataset_name, model_name), "evaluation_results.csv")



def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def authenticate_hf():
    login(token=HF_TOKEN)


def load_kg():
    if os.path.exists(KG_PARQUET_PATH):
        print("Loading KG from parquet (fast path)...")
        kg_df = pd.read_parquet(KG_PARQUET_PATH)
    else:
        print("Loading KG from CSV (first run - converting to parquet)...")
        kg_df = pd.read_csv(KG_CSV_PATH, low_memory=False)
        kg_df.to_parquet(KG_PARQUET_PATH, index=False)
        print(f"Saved parquet to {KG_PARQUET_PATH}")
    print(f"Knowledge graph loaded: {kg_df.shape[0]:,} rows, {kg_df.shape[1]} cols")
    return kg_df


def load_benchmark(dataset_name="knowlab-research/BioHopR"):
    print(f"Loading {dataset_name} benchmark dataset...")
    ds = load_dataset(dataset_name)
    split = list(ds.keys())[0]
    benchmark_df = ds[split].to_pandas()
    print(f"Benchmark loaded: {benchmark_df.shape[0]:,} samples (split: {split})")
    return benchmark_df


def load_llm(device, model_name="meta-llama/Llama-3.2-3B-Instruct"):
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("LLM loaded.")
    return llm_pipe


def load_embedder(device):
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence embedder ({EMBEDDER_NAME})...")
    embedder = SentenceTransformer(EMBEDDER_NAME).to(device)
    print("Embedder loaded.")
    return embedder


def generate_answer(llm_pipe, question: str) -> str:
    tokenizer = llm_pipe.tokenizer
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": "Provide exactly one specific answer."},
            {"role": "user", "content": question},
        ]
        outputs = llm_pipe(messages, max_new_tokens=50, do_sample=False)
        return outputs[0]["generated_text"][-1]["content"].strip()
    else:
        prompt = f"Question: {question}\nAnswer:"
        outputs = llm_pipe(prompt, max_new_tokens=50, do_sample=False)
        generated = outputs[0]["generated_text"]
        answer = generated[len(prompt):].strip()
        return answer.split("\n")[0].strip()


def evaluate_instance(embedder, prediction: str, targets, threshold: float = COSINE_THRESHOLD):
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "PrimeKG"), exist_ok=True)
