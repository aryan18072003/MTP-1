import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import time
import os
from utils import (
    setup_device, authenticate_hf, load_biohopr,
    load_llm, load_embedder,
    generate_answer, evaluate_instance,
    ensure_results_dir, RESULTS_DIR, EVAL_RESULTS_PATH,
)

sns.set_theme(style="whitegrid", font_scale=1.2)

CHECKPOINT_INTERVAL = 50


def load_checkpoint():
    if os.path.exists(EVAL_RESULTS_PATH):
        df = pd.read_csv(EVAL_RESULTS_PATH)
        print(f"  Loaded checkpoint: {len(df)} samples already evaluated")
        return df
    return pd.DataFrame()


def save_checkpoint(results_df):
    results_df.to_csv(EVAL_RESULTS_PATH, index=False)


def evaluate_all(benchmark_df, llm_pipe, embedder, max_samples=None):
    existing_df = load_checkpoint()
    completed_indices = set(existing_df["sample_idx"].tolist()) if len(existing_df) > 0 else set()

    total_samples = len(benchmark_df)
    if max_samples:
        total_samples = min(max_samples, total_samples)

    results = existing_df.to_dict("records") if len(existing_df) > 0 else []
    new_count = 0
    start_time = time.time()

    print(f"\nEvaluating {total_samples} samples (skipping {len(completed_indices)} already done)...")

    for idx in range(total_samples):
        if idx in completed_indices:
            continue

        row = benchmark_df.iloc[idx]
        targets = row.get("answer", [])
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()

        hop1_type = row.get("hop1_type", "unknown")
        hop2_type = row.get("hop2_type", "unknown")
        target_type = row.get("target_type", "unknown")
        relation_hop1 = row.get("relation_hop1", "")
        relation_hop2 = row.get("relation_hop2", "")

        hop1_q = row.get("hop1_question")
        hop1_pred, hop1_correct, hop1_sim = "", False, 0.0
        if pd.notna(hop1_q) and hop1_q:
            try:
                hop1_pred = generate_answer(llm_pipe, hop1_q)
                hop1_correct, hop1_sim = evaluate_instance(embedder, hop1_pred, targets)
            except Exception as e:
                print(f"  Error on sample {idx} (hop1): {e}")

        hop2_q = row.get("hop2_question")
        hop2_pred, hop2_correct, hop2_sim = "", False, 0.0
        if pd.notna(hop2_q) and hop2_q:
            try:
                hop2_pred = generate_answer(llm_pipe, hop2_q)
                hop2_correct, hop2_sim = evaluate_instance(embedder, hop2_pred, targets)
            except Exception as e:
                print(f"  Error on sample {idx} (hop2): {e}")

        results.append({
            "sample_idx": idx,
            "hop1_type": hop1_type,
            "hop2_type": hop2_type,
            "target_type": target_type,
            "relation_hop1": relation_hop1,
            "relation_hop2": relation_hop2,
            "hop1_question": hop1_q if pd.notna(hop1_q) else "",
            "hop2_question": hop2_q if pd.notna(hop2_q) else "",
            "hop1_prediction": hop1_pred,
            "hop2_prediction": hop2_pred,
            "hop1_correct": hop1_correct,
            "hop2_correct": hop2_correct,
            "hop1_similarity": hop1_sim,
            "hop2_similarity": hop2_sim,
            "n_targets": len(targets),
            "targets": str(targets),
        })

        new_count += 1
        elapsed = time.time() - start_time
        rate = new_count / elapsed if elapsed > 0 else 0
        remaining = (total_samples - idx - 1) / rate if rate > 0 else 0

        if new_count % 10 == 0:
            print(f"  [{idx + 1}/{total_samples}] "
                  f"hop1_sim={hop1_sim:.3f} hop2_sim={hop2_sim:.3f} "
                  f"({rate:.1f} samples/s, ~{remaining / 60:.0f}m remaining)")

        if new_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(pd.DataFrame(results))
            print(f"  Checkpoint saved ({len(results)} total results)")

    results_df = pd.DataFrame(results)
    save_checkpoint(results_df)
    print(f"\n  Evaluation complete: {len(results_df)} total results")
    return results_df


def plot_evaluation_summary(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    hop1_mask = results_df["hop1_question"].astype(str).str.len() > 0
    hop2_mask = results_df["hop2_question"].astype(str).str.len() > 0

    hop1_acc = results_df.loc[hop1_mask, "hop1_correct"].mean() * 100
    hop2_acc = results_df.loc[hop2_mask, "hop2_correct"].mean() * 100

    bars = ax.bar(["1-hop", "2-hop"], [hop1_acc, hop2_acc],
                  color=["#4CAF50", "#FF7043"], edgecolor="gray", width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy", fontweight="bold")
    ax.set_ylim(0, max(hop1_acc, hop2_acc) * 1.3 + 5)
    for bar, acc in zip(bars, [hop1_acc, hop2_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax = axes[1]
    if hop1_mask.any():
        ax.hist(results_df.loc[hop1_mask, "hop1_similarity"], bins=30, alpha=0.6,
                label="1-hop", color="#4CAF50", edgecolor="white")
    if hop2_mask.any():
        ax.hist(results_df.loc[hop2_mask, "hop2_similarity"], bins=30, alpha=0.6,
                label="2-hop", color="#FF7043", edgecolor="white")
    ax.axvline(x=0.9, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.9)")
    ax.set_xlabel("Max Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Similarity Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[2]
    sim_by_type = results_df.groupby("target_type").agg(
        hop1_mean=("hop1_similarity", "mean"),
        hop2_mean=("hop2_similarity", "mean"),
    ).sort_values("hop1_mean", ascending=True)

    x = np.arange(len(sim_by_type))
    width = 0.35
    ax.barh(x - width / 2, sim_by_type["hop1_mean"], width, label="1-hop", color="#4CAF50")
    ax.barh(x + width / 2, sim_by_type["hop2_mean"], width, label="2-hop", color="#FF7043")
    ax.set_yticks(x)
    ax.set_yticklabels(sim_by_type.index, fontsize=9)
    ax.set_xlabel("Mean Cosine Similarity")
    ax.set_title("Similarity by Target Type", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle(f"LLM Evaluation Summary (n={len(results_df):,})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved evaluation_summary.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, correct_col, title in zip(
        axes,
        ["hop1_correct", "hop2_correct"],
        ["1-Hop Accuracy by (Hop1 Type -> Target Type)", "2-Hop Accuracy by (Hop1 Type -> Target Type)"],
    ):
        acc_pivot = results_df.groupby(["hop1_type", "target_type"])[correct_col].mean().unstack(fill_value=0) * 100
        count_pivot = results_df.groupby(["hop1_type", "target_type"]).size().unstack(fill_value=0)

        annot = acc_pivot.copy().astype(str)
        for i in acc_pivot.index:
            for j in acc_pivot.columns:
                if i in count_pivot.index and j in count_pivot.columns:
                    annot.loc[i, j] = f"{acc_pivot.loc[i, j]:.1f}%\n(n={count_pivot.loc[i, j]})"

        sns.heatmap(acc_pivot, annot=annot, fmt="", cmap="RdYlGn",
                    linewidths=1, linecolor="white", ax=ax,
                    vmin=0, vmax=100, cbar_kws={"label": "Accuracy (%)"})
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Target Type")
        ax.set_ylabel("Hop 1 Type")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_by_node_type.png"), dpi=150)
    plt.close()
    print("  Saved accuracy_by_node_type.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM on BioHopR")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    args = parser.parse_args()

    ensure_results_dir()
    device = setup_device()
    authenticate_hf()

    benchmark_df = load_biohopr()
    llm_pipe = load_llm(device)
    embedder = load_embedder(device)

    results_df = evaluate_all(benchmark_df, llm_pipe, embedder, max_samples=args.max_samples)

    print("\nGenerating plots...")
    plot_evaluation_summary(results_df)

    hop1_mask = results_df["hop1_question"].astype(str).str.len() > 0
    hop2_mask = results_df["hop2_question"].astype(str).str.len() > 0
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total samples evaluated: {len(results_df):,}")
    if hop1_mask.any():
        print(f"1-hop accuracy: {results_df.loc[hop1_mask, 'hop1_correct'].mean() * 100:.2f}%")
        print(f"1-hop mean similarity: {results_df.loc[hop1_mask, 'hop1_similarity'].mean():.3f}")
    if hop2_mask.any():
        print(f"2-hop accuracy: {results_df.loc[hop2_mask, 'hop2_correct'].mean() * 100:.2f}%")
        print(f"2-hop mean similarity: {results_df.loc[hop2_mask, 'hop2_similarity'].mean():.3f}")

    print("\n Step 3 complete")
