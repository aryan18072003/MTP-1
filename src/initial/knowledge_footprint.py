"""
Knowledge Footprint Generation
Creates heatmaps showing LLM accuracy across different KG node types.

Outputs (saved to src/results/):
  - knowledge_footprint_overall.png
  - knowledge_footprint_1hop.png
  - knowledge_footprint_2hop.png
  - knowledge_footprint_by_relation.png
  - knowledge_footprint_dashboard.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from utils import ensure_results_dir, RESULTS_DIR, EVAL_RESULTS_PATH


sns.set_theme(style="white", font_scale=1.1)


def load_evaluation_results():
    """Load the evaluation results CSV from the evaluation step."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        raise FileNotFoundError(
            f"Evaluation results not found at {EVAL_RESULTS_PATH}.\n"
            "Run evaluate_llm.py first."
        )
    df = pd.read_csv(EVAL_RESULTS_PATH)
    print(f"Loaded {len(df):,} evaluation results")
    return df


def create_heatmap(pivot_acc, pivot_count, title, filename, figsize=(10, 7)):
    """Create and save a single heatmap with accuracy + count annotations."""
    # Build annotation text
    annot = pivot_acc.copy().astype(str)
    for i in pivot_acc.index:
        for j in pivot_acc.columns:
            acc = pivot_acc.loc[i, j]
            cnt = pivot_count.loc[i, j] if i in pivot_count.index and j in pivot_count.columns else 0
            if cnt > 0:
                annot.loc[i, j] = f"{acc:.1f}%\n(n={int(cnt)})"
            else:
                annot.loc[i, j] = "N/A"

    fig, ax = plt.subplots(figsize=figsize)

    # Mask cells with no samples
    mask = pivot_count == 0

    sns.heatmap(
        pivot_acc,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        linewidths=2,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=100,
        mask=mask,
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Target Node Type", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Source Node Type", fontsize=12, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=10)

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def generate_overall_footprint(results_df):
    """
    Knowledge footprint: accuracy of LLM for each (source_type → target_type) pair.
    Combines both hop1 and hop2 results.
    """
    print("\n" + "=" * 60)
    print("OVERALL KNOWLEDGE FOOTPRINT")
    print("=" * 60)

    # Combine hop1 and hop2 into a unified view
    rows = []

    for _, row in results_df.iterrows():
        hop1_q = str(row.get("hop1_question", ""))
        hop2_q = str(row.get("hop2_question", ""))

        if len(hop1_q) > 0 and hop1_q != "nan":
            rows.append({
                "source_type": row["hop1_type"],
                "target_type": row["target_type"],
                "correct": row["hop1_correct"],
                "similarity": row["hop1_similarity"],
                "hop": "1-hop",
            })
        if len(hop2_q) > 0 and hop2_q != "nan":
            rows.append({
                "source_type": row["hop1_type"],
                "target_type": row["target_type"],
                "correct": row["hop2_correct"],
                "similarity": row["hop2_similarity"],
                "hop": "2-hop",
            })

    combined_df = pd.DataFrame(rows)

    # Overall heatmap
    acc_pivot = combined_df.groupby(["source_type", "target_type"])["correct"].mean().unstack(fill_value=0) * 100
    count_pivot = combined_df.groupby(["source_type", "target_type"]).size().unstack(fill_value=0)

    create_heatmap(
        acc_pivot, count_pivot,
        "Knowledge Footprint — LLM Accuracy\n(Llama-3.2-3B on BioHopR / PrimeKG)",
        "knowledge_footprint_overall.png",
    )

    # Print summary table
    print("\nAccuracy by (Source Type → Target Type):")
    for src in acc_pivot.index:
        for tgt in acc_pivot.columns:
            cnt = count_pivot.loc[src, tgt] if src in count_pivot.index and tgt in count_pivot.columns else 0
            if cnt > 0:
                print(f"  {src} → {tgt}: {acc_pivot.loc[src, tgt]:.1f}% (n={int(cnt)})")

    return combined_df


def generate_hop_footprints(results_df):
    """Generate separate heatmaps for 1-hop and 2-hop."""
    print("\n" + "=" * 60)
    print("1-HOP & 2-HOP KNOWLEDGE FOOTPRINTS")
    print("=" * 60)

    for hop_label, correct_col, sim_col, q_col, filename in [
        ("1-Hop", "hop1_correct", "hop1_similarity", "hop1_question", "knowledge_footprint_1hop.png"),
        ("2-Hop", "hop2_correct", "hop2_similarity", "hop2_question", "knowledge_footprint_2hop.png"),
    ]:
        mask = results_df[q_col].astype(str).str.len() > 0
        mask = mask & (results_df[q_col].astype(str) != "nan")
        subset = results_df[mask]

        if len(subset) == 0:
            print(f"  ⚠ No {hop_label} samples found, skipping")
            continue

        acc_pivot = subset.groupby(["hop1_type", "target_type"])[correct_col].mean().unstack(fill_value=0) * 100
        count_pivot = subset.groupby(["hop1_type", "target_type"]).size().unstack(fill_value=0)

        create_heatmap(
            acc_pivot, count_pivot,
            f"Knowledge Footprint — {hop_label} Accuracy\n(Llama-3.2-3B on BioHopR / PrimeKG)",
            filename,
        )

        total_acc = subset[correct_col].mean() * 100
        mean_sim = subset[sim_col].mean()
        print(f"\n{hop_label} Summary:")
        print(f"  Samples: {len(subset):,}")
        print(f"  Overall accuracy: {total_acc:.2f}%")
        print(f"  Mean similarity: {mean_sim:.3f}")


def generate_relation_footprint(results_df):
    """Generate heatmap by relation type."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE FOOTPRINT BY RELATION TYPE")
    print("=" * 60)

    # Focus on hop1 relations (more samples)
    mask = results_df["hop1_question"].astype(str).str.len() > 0
    mask = mask & (results_df["hop1_question"].astype(str) != "nan")
    subset = results_df[mask].copy()

    if len(subset) == 0:
        print("  ⚠ No data available for relation footprint")
        return

    # Get top relations by sample count
    relation_counts = subset["relation_hop1"].value_counts()
    top_relations = relation_counts.head(10).index.tolist()
    subset_top = subset[subset["relation_hop1"].isin(top_relations)]

    acc_pivot = subset_top.groupby(["relation_hop1", "target_type"])["hop1_correct"].mean().unstack(fill_value=0) * 100
    count_pivot = subset_top.groupby(["relation_hop1", "target_type"]).size().unstack(fill_value=0)

    create_heatmap(
        acc_pivot, count_pivot,
        "Knowledge Footprint by Relation Type\n(Top 10 Relations — 1-Hop Accuracy)",
        "knowledge_footprint_by_relation.png",
        figsize=(12, 8),
    )


def generate_summary_dashboard(results_df):
    """Generate a comprehensive summary dashboard."""
    print("\n" + "=" * 60)
    print("SUMMARY DASHBOARD")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 1. Accuracy comparison bar chart ─────────────────────────────────
    ax = axes[0, 0]
    hop1_mask = results_df["hop1_question"].astype(str).str.len() > 0
    hop1_mask = hop1_mask & (results_df["hop1_question"].astype(str) != "nan")
    hop2_mask = results_df["hop2_question"].astype(str).str.len() > 0
    hop2_mask = hop2_mask & (results_df["hop2_question"].astype(str) != "nan")

    node_types = sorted(results_df["target_type"].unique())
    hop1_accs = []
    hop2_accs = []
    for nt in node_types:
        nt_mask = results_df["target_type"] == nt
        h1 = results_df[hop1_mask & nt_mask]["hop1_correct"].mean() * 100 if (hop1_mask & nt_mask).any() else 0
        h2 = results_df[hop2_mask & nt_mask]["hop2_correct"].mean() * 100 if (hop2_mask & nt_mask).any() else 0
        hop1_accs.append(h1)
        hop2_accs.append(h2)

    x = np.arange(len(node_types))
    width = 0.35
    ax.bar(x - width / 2, hop1_accs, width, label="1-hop", color="#4CAF50", edgecolor="white")
    ax.bar(x + width / 2, hop2_accs, width, label="2-hop", color="#FF7043", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(node_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Target Node Type", fontweight="bold")
    ax.legend()

    # ── 2. Sample count distribution ─────────────────────────────────────
    ax = axes[0, 1]
    combo_counts = results_df.groupby(["hop1_type", "target_type"]).size().unstack(fill_value=0)
    combo_counts.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_xlabel("Source Node Type")
    ax.set_ylabel("Count")
    ax.set_title("Samples per Node Type Pair", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Target Type", fontsize=8, title_fontsize=9)

    # ── 3. Similarity box plot ───────────────────────────────────────────
    ax = axes[1, 0]
    plot_data = []
    if hop1_mask.any():
        for _, row in results_df[hop1_mask].iterrows():
            plot_data.append({"Type": row["target_type"], "Hop": "1-hop", "Similarity": row["hop1_similarity"]})
    if hop2_mask.any():
        for _, row in results_df[hop2_mask].iterrows():
            plot_data.append({"Type": row["target_type"], "Hop": "2-hop", "Similarity": row["hop2_similarity"]})

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x="Type", y="Similarity", hue="Hop", ax=ax,
                    palette={"1-hop": "#4CAF50", "2-hop": "#FF7043"})
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="Threshold")
    ax.set_title("Similarity Distribution by Type", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    # ── 4. Performance drop: 1-hop vs 2-hop ──────────────────────────────
    ax = axes[1, 1]
    drops = []
    for nt in node_types:
        nt_mask = results_df["target_type"] == nt
        h1 = results_df[hop1_mask & nt_mask]["hop1_correct"].mean() * 100 if (hop1_mask & nt_mask).any() else 0
        h2 = results_df[hop2_mask & nt_mask]["hop2_correct"].mean() * 100 if (hop2_mask & nt_mask).any() else 0
        drops.append(h1 - h2)

    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in drops]
    ax.barh(node_types, drops, color=colors, edgecolor="white")
    ax.set_xlabel("Accuracy Drop (1-hop − 2-hop) in pp")
    ax.set_title("Performance Drop: 1-Hop → 2-Hop", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.suptitle("Knowledge Footprint Dashboard — Llama-3.2-3B on BioHopR",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "knowledge_footprint_dashboard.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved knowledge_footprint_dashboard.png")


if __name__ == "__main__":
    ensure_results_dir()

    results_df = load_evaluation_results()

    combined_df = generate_overall_footprint(results_df)
    generate_hop_footprints(results_df)
    generate_relation_footprint(results_df)
    generate_summary_dashboard(results_df)

    print("\n✅ Step 4 complete — all knowledge footprint plots saved to results/")
