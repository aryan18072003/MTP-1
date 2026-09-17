from utils import (
    setup_device, authenticate_hf, load_benchmark,
    load_llm, load_embedder,
    generate_answer, evaluate_instance,
)
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="knowlab-research/BioHopR")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    authenticate_hf()
    device = setup_device()

    benchmark_df = load_benchmark(args.dataset)

    llm_pipe = load_llm(device, args.model)
    embedder = load_embedder(device)


    print(f"\nDataset columns: {list(benchmark_df.columns)}")

    sample_row = benchmark_df.iloc[0]
    question = sample_row.get("prompt") or sample_row.get("question_single")
    targets = sample_row.get("answer", [])
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    print(f"\nQuestion: {question}")
    print(f"Targets: {targets}")

    prediction = generate_answer(llm_pipe, question)
    is_correct, score = evaluate_instance(embedder, prediction, targets)

    print(f"Prediction: {prediction}")
    print(f"Correct: {is_correct} | Max Cosine Similarity: {score:.3f}")
