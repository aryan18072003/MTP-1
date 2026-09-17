"""One optional dashboard, with two-hop paths labelled by their actual source."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_dashboard(results, path, kg_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for hop, ax in zip((1, 2), axes[0]):
        subset = results[(results["hop"] == hop) & results["applicable"]].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No applicable questions", ha="center")
            ax.set_axis_off()
            continue
        subset["path_start"] = subset["source_type"]
        if hop == 2:
            subset["path_start"] += " via " + subset["intermediate_type"]
        grouped = subset.groupby(["path_start", "target_type"])["hit"].agg(["mean", "size"])
        acc = grouped["mean"].unstack() * 100
        counts = grouped["size"].unstack().reindex_like(acc)
        annotations = acc.copy().astype(object)
        for i in acc.index:
            for j in acc.columns:
                annotations.loc[i, j] = "" if pd.isna(acc.loc[i, j]) else f"{acc.loc[i, j]:.1f}%\n(n={int(counts.loc[i, j])})"
        sns.heatmap(acc, mask=acc.isna(), annot=annotations, fmt="", vmin=0, vmax=100,
                    cmap="RdYlGn", ax=ax, cbar_kws={"label": "Hit rate (%)"})
        ax.set_title(f"{hop}-hop reference-answer hit rate")
        ax.set_xlabel("Target type")
        ax.set_ylabel("Source type" if hop == 1 else "Source type via intermediate type")
    valid = results[results["applicable"]]
    if not valid.empty:
        metrics = valid.groupby("hop")[["hit", "recall", "precision", "f1"]].mean() * 100
        metrics.rename(columns={"hit": "Hit rate", "recall": "Recall", "precision": "Precision", "f1": "F1"}).plot.bar(ax=axes[1, 0])
        axes[1, 0].set_ylabel("Macro average (%)")
        axes[1, 0].set_title("Agreement with released answers")
        coverage_data = valid.assign(path_supported=valid["supported_types"] & valid["supported_relations"])
        coverage = coverage_data.groupby("hop")[["start_node_exists", "path_supported"]].mean() * 100
        coverage.rename(columns={"start_node_exists": "Start entity found", "path_supported": "Relation path supported"}).plot.bar(ax=axes[1, 1])
        axes[1, 1].set_ylabel("Questions (%)")
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].set_title("Coverage")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)
    policy = results["relation_policy"].iloc[0]
    fig.suptitle(f"{kg_name} on BioHopR — {policy} retrieval", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

