import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import load_kg, load_benchmark, ensure_results_dir, get_dataset_dir
import os
import argparse

sns.set_theme(style="whitegrid", font_scale=1.2)


def build_kg_name_index(kg_df):
    x_names = set(kg_df["x_name"].dropna().str.lower().unique())
    y_names = set(kg_df["y_name"].dropna().str.lower().unique())
    return x_names | y_names


def check_entity_coverage(benchmark_df, kg_names):
    results = {}

    for col in ["hop1", "hop2"]:
        entities = benchmark_df[col].dropna()
        total = len(entities)
        found = sum(1 for e in entities if str(e).lower() in kg_names)
        results[col] = {"total": total, "found": found, "pct": found / total * 100 if total > 0 else 0}
        print(f"  {col}: {found}/{total} ({results[col]['pct']:.1f}%) found in PrimeKG")


    all_answers = []
    for answers in benchmark_df["answer"]:
        if isinstance(answers, (list, np.ndarray)):
            all_answers.extend([str(a).lower() for a in answers])
        elif isinstance(answers, str):
            all_answers.append(answers.lower())

    total_ans = len(all_answers)
    found_ans = sum(1 for a in all_answers if a in kg_names)
    results["answer"] = {"total": total_ans, "found": found_ans,
                         "pct": found_ans / total_ans * 100 if total_ans > 0 else 0}
    print(f"  answer entities: {found_ans}/{total_ans} ({results['answer']['pct']:.1f}%) found in PrimeKG")

    return results


def build_mapping_table(benchmark_df, kg_df):
    x_lookup = kg_df[["x_name", "x_index", "x_type"]].drop_duplicates(subset="x_name")
    x_lookup.columns = ["name", "kg_index", "kg_type"]
    y_lookup = kg_df[["y_name", "y_index", "y_type"]].drop_duplicates(subset="y_name")
    y_lookup.columns = ["name", "kg_index", "kg_type"]
    name_to_info = pd.concat([x_lookup, y_lookup]).drop_duplicates(subset="name")
    name_to_info["name_lower"] = name_to_info["name"].str.lower()
    name_to_info = name_to_info.set_index("name_lower")

    mapping_rows = []
    for idx, row in benchmark_df.iterrows():
        hop1 = str(row.get("hop1", "")).lower()
        hop2 = str(row.get("hop2", "")).lower() if pd.notna(row.get("hop2")) else None

        hop1_info = name_to_info.loc[hop1] if hop1 in name_to_info.index else None
        hop2_info = name_to_info.loc[hop2] if hop2 and hop2 in name_to_info.index else None

        mapping_rows.append({
            "biohopr_idx": idx,
            "hop1": row.get("hop1"),
            "hop1_type": row.get("hop1_type"),
            "hop1_kg_index": hop1_info["kg_index"] if hop1_info is not None and not isinstance(hop1_info, pd.DataFrame) else None,
            "hop2": row.get("hop2"),
            "hop2_type": row.get("hop2_type"),
            "hop2_kg_index": hop2_info["kg_index"] if hop2_info is not None and not isinstance(hop2_info, pd.DataFrame) else None,
            "target_type": row.get("target_type"),
            "relation_hop1": row.get("relation_hop1"),
            "relation_hop2": row.get("relation_hop2"),
            "n_answers": len(row["answer"]) if isinstance(row["answer"], (list, np.ndarray)) else 1,
        })

    mapping_df = pd.DataFrame(mapping_rows)
    return mapping_df


def plot_coverage(coverage_results, mapping_df, dataset_name):
    dataset_short = dataset_name.split("/")[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    ax = axes[0]
    entities = list(coverage_results.keys())
    pcts = [coverage_results[e]["pct"] for e in entities]
    totals = [coverage_results[e]["total"] for e in entities]
    colors = sns.color_palette("Greens_d", len(entities))

    bars = ax.bar(entities, pcts, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"{dataset_short} Entity Coverage in PrimeKG", fontweight="bold")
    ax.set_ylim(0, 110)
    for bar, pct, total in zip(bars, pcts, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n({total:,})", ha="center", va="bottom", fontsize=9)


    ax = axes[1]
    hop1_mapped = mapping_df["hop1_kg_index"].notna().sum()
    hop2_mapped = mapping_df["hop2_kg_index"].notna().sum()
    hop2_total = mapping_df["hop2"].notna().sum()
    total = len(mapping_df)

    labels = ["Hop 1 -> KG", "Hop 2 -> KG"]
    mapped_pcts = [hop1_mapped / total * 100, hop2_mapped / hop2_total * 100 if hop2_total > 0 else 0]
    colors2 = sns.color_palette("Blues_d", 2)
    bars = ax.bar(labels, mapped_pcts, color=colors2, edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("Mapping Rate (%)")
    ax.set_title(f"{dataset_short} -> PrimeKG Node Mapping", fontweight="bold")
    ax.set_ylim(0, 110)
    for bar, pct in zip(bars, mapped_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_dir = get_dataset_dir(dataset_name)
    plt.savefig(os.path.join(out_dir, "node_mapping_coverage.png"), dpi=150)
    plt.close()
    print("  Saved node_mapping_coverage.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="knowlab-research/BioHopR")
    args = parser.parse_args()

    ensure_results_dir()

    print("Loading data...")
    kg_df = load_kg()
    benchmark_df = load_benchmark(args.dataset)

    print("\n" + "=" * 60)
    print("ENTITY COVERAGE CHECK")
    print("=" * 60)
    kg_names = build_kg_name_index(kg_df)
    print(f"Unique entity names in PrimeKG: {len(kg_names):,}")
    coverage = check_entity_coverage(benchmark_df, kg_names)

    print("\n" + "=" * 60)
    print("BUILDING NODE MAPPING TABLE")
    print("=" * 60)
    mapping_df = build_mapping_table(benchmark_df, kg_df)
    out_dir = get_dataset_dir(args.dataset)
    mapping_path = os.path.join(out_dir, "node_mapping_summary.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"  Saved node_mapping_summary.csv ({len(mapping_df):,} rows)")

    print(f"\nMapping statistics:")
    print(f"  Hop 1 mapped to KG: {mapping_df['hop1_kg_index'].notna().sum():,} / {len(mapping_df):,}")
    hop2_total = mapping_df["hop2"].notna().sum()
    print(f"  Hop 2 mapped to KG: {mapping_df['hop2_kg_index'].notna().sum():,} / {hop2_total:,}")

    print("\nGenerating plots...")
    plot_coverage(coverage, mapping_df, args.dataset)

    print("\n Step 2 complete - mapping saved to results/")
