import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
from utils import load_kg, load_benchmark, ensure_results_dir, RESULTS_DIR, get_dataset_dir
import os
import argparse

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("viridis", 12)


def analyze_kg(kg_df):

    print("\n" + "=" * 60)
    print("PRIMEKG ANALYSIS")
    print("=" * 60)

    print(f"\nColumns: {list(kg_df.columns)}")
    print(f"Total edges: {kg_df.shape[0]:,}")

    all_node_types = pd.concat([kg_df["x_type"], kg_df["y_type"]]).value_counts()
    print(f"\nNode types ({len(all_node_types)}):")
    for ntype, count in all_node_types.items():
        print(f"  {ntype}: {count:,}")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        all_node_types.index[::-1],
        all_node_types.values[::-1],
        color=PALETTE[: len(all_node_types)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Count (edges involving this node type)")
    ax.set_title("PrimeKG - Node Type Distribution", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    for bar, val in zip(bars, all_node_types.values[::-1]):
        ax.text(bar.get_width() + max(all_node_types.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    plt.tight_layout()
    kg_dir = os.path.join(RESULTS_DIR, "PrimeKG")
    plt.savefig(os.path.join(kg_dir, "kg_node_type_distribution.png"), dpi=150)
    plt.close()
    print("  Saved kg_node_type_distribution.png")

    relation_counts = kg_df["display_relation"].value_counts()
    print(f"\nRelation types ({len(relation_counts)}):")
    for rtype, count in relation_counts.head(15).items():
        print(f"  {rtype}: {count:,}")

    top_n = 20
    top_relations = relation_counts.head(top_n)
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        top_relations.index[::-1],
        top_relations.values[::-1],
        color=sns.color_palette("magma", top_n),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Count")
    ax.set_title(f"PrimeKG - Top {top_n} Relation Types", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    for bar, val in zip(bars, top_relations.values[::-1]):
        ax.text(bar.get_width() + max(top_relations.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(kg_dir, "kg_relation_type_distribution.png"), dpi=150)
    plt.close()
    print("  Saved kg_relation_type_distribution.png")

    unique_x = kg_df[["x_index", "x_type", "x_name"]].drop_duplicates(subset="x_index")
    unique_y = kg_df[["y_index", "y_type", "y_name"]].drop_duplicates(subset="y_index")
    unique_x.columns = ["index", "type", "name"]
    unique_y.columns = ["index", "type", "name"]
    all_nodes = pd.concat([unique_x, unique_y]).drop_duplicates(subset="index")
    print(f"\nUnique nodes: {len(all_nodes):,}")
    node_type_counts = all_nodes["type"].value_counts()
    print("Unique nodes by type:")
    for ntype, count in node_type_counts.items():
        print(f"  {ntype}: {count:,}")

    return all_node_types, relation_counts


def analyze_biohopr(benchmark_df, dataset_name):
    print("\n" + "=" * 60)
    print(f"{dataset_name.split('/')[-1].upper()} ANALYSIS")
    print("=" * 60)

    print(f"\nColumns: {list(benchmark_df.columns)}")
    print(f"Total samples: {benchmark_df.shape[0]:,}")

    out_dir = get_dataset_dir(dataset_name)
    sample_path = os.path.join(out_dir, "raw_sample.csv")
    benchmark_df.head(10).to_csv(sample_path, index=False)
    print(f"  Saved raw_sample.csv (first 10 rows) to {out_dir}")

    has_hop2 = benchmark_df["hop2"].notna() & (benchmark_df["hop2"] != "")
    n_1hop = (~has_hop2).sum()
    n_2hop = has_hop2.sum()
    print(f"\n1-hop samples: {n_1hop:,}")
    print(f"2-hop samples: {n_2hop:,}")

    print("\nhop1_type distribution:")
    print(benchmark_df["hop1_type"].value_counts().to_string())
    print("\nhop2_type distribution:")
    print(benchmark_df["hop2_type"].value_counts().to_string())
    print("\ntarget_type distribution:")
    print(benchmark_df["target_type"].value_counts().to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col, title, color in zip(
        axes,
        ["hop1_type", "hop2_type", "target_type"],
        ["Hop 1 Node Type", "Hop 2 Node Type", "Target Node Type"],
        [sns.color_palette("Set2"), sns.color_palette("Set3"), sns.color_palette("Pastel1")],
    ):
        counts = benchmark_df[col].value_counts()
        bars = ax.bar(range(len(counts)), counts.values, color=color[: len(counts)],
                      edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(val), ha="center", va="bottom", fontsize=9)

    dataset_short = dataset_name.split("/")[-1]
    plt.suptitle(f"{dataset_short} - Node Type Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_dir = get_dataset_dir(dataset_name)
    plt.savefig(os.path.join(out_dir, f"{dataset_short.lower()}_hop_type_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {dataset_short.lower()}_hop_type_distribution.png")

    combo_counts = benchmark_df.groupby(["hop1_type", "target_type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        combo_counts,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Sample Count"},
    )
    ax.set_xlabel("Target Node Type", fontweight="bold")
    ax.set_ylabel("Hop 1 Node Type", fontweight="bold")
    ax.set_title(f"{dataset_short} - Sample Counts by (Hop1 Type -> Target Type)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_short.lower()}_sample_counts.png"), dpi=150)
    plt.close()
    print(f"  Saved {dataset_short.lower()}_sample_counts.png")

    print("\nrelation_hop1 distribution:")
    print(benchmark_df["relation_hop1"].value_counts().to_string())
    print("\nrelation_hop2 distribution:")
    print(benchmark_df["relation_hop2"].value_counts().to_string())

    answer_lengths = benchmark_df["answer"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
    print(f"\nAnswer count statistics:")
    print(f"  Mean: {answer_lengths.mean():.1f}")
    print(f"  Median: {answer_lengths.median():.1f}")
    print(f"  Min: {answer_lengths.min()}")
    print(f"  Max: {answer_lengths.max()}")

    print("\n--- First Sample ---")
    row = benchmark_df.iloc[0]
    for col in benchmark_df.columns:
        val = row[col]
        if isinstance(val, str) and len(val) > 100:
            val = val[:100] + "..."
        print(f"  {col}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="knowlab-research/BioHopR")
    args = parser.parse_args()

    ensure_results_dir()
    kg_df = load_kg()
    benchmark_df = load_benchmark(args.dataset)
    analyze_kg(kg_df)
    analyze_biohopr(benchmark_df, args.dataset)
    print("\n Step 1 complete - all plots saved to results/")
