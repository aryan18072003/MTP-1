"""Evaluate BioHopR with relation-aware KG retrieval and exact-name scoring.

Run from the repository root:
    python -m src.kg_eval.evaluate_kg --kg primekg hetionet drkg biokg
No model inference or embedding-based scoring is performed.
"""
import argparse
import gc
import gzip
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

try:
    from .kg_loader import KG_REGISTRY, get_kg
    from .utils import (create_run_dir, load_benchmark, normalize, parse_targets,
                        sha256_file, utc_now, write_json)
except ImportError:
    from kg_loader import KG_REGISTRY, get_kg
    from utils import (create_run_dir, load_benchmark, normalize, parse_targets,
                       sha256_file, utc_now, write_json)

PROTOCOL = {
    "version": "3.0",
    "task": "relation-aware biomedical retrieval",
    "one_hop": "hop1 -> target_type",
    "two_hop": "hop2 -> any node of hop1_type -> target_type",
    "edge_predicates": "semantic mode enforces positive treatment, phenotype and side-effect edges; other associations pooled by type",
    "direction": "traverse relations or their inverses according to requested endpoint types",
    "matching": "casefolded preferred names with whitespace normalization; no fuzzy matching",
    "ambiguous_start_names": "union of matching IDs, count reported; intermediate IDs never merged",
    "answer_unit": "unique normalized preferred name; unlabelled IDs count as unmatched candidates",
    "missing_question_or_fields_or_targets": "excluded, with explicit status",
    "missing_nodes_or_unsupported_paths": "scored as zero; coverage and supported-only metrics also reported",
    "aggregation": "macro average over applicable benchmark rows separately for each hop",
    "interpretation": "agreement with released reference answers, not exhaustive biomedical truth or LLM accuracy",
}


def compute_metrics(predicted, ground_truth):
    predicted = {normalize(x) for x in predicted if normalize(x)}
    truth = set(parse_targets(ground_truth))
    overlap = predicted & truth
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(truth) if truth else 0.0
    return {"hit": bool(overlap), "recall": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "n_predicted": len(predicted), "n_ground_truth": len(truth), "n_overlap": len(overlap)}


def evaluate_all(benchmark_df, kg, dataset_name=None, max_samples=None, predictions_path=None, relation_policy="semantic"):
    """One result row per sample and hop; never saves or overwrites legacy results."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if relation_policy not in {"semantic", "type-only"}:
        raise ValueError("Unknown relation policy")
    benchmark = benchmark_df.iloc[:max_samples] if max_samples is not None else benchmark_df
    if benchmark.empty:
        raise ValueError("Cannot evaluate an empty benchmark")
    results = []
    supported_types = kg.supported_types
    evidence = gzip.open(predictions_path, "wt", encoding="utf-8") if predictions_path else None
    try:
        for idx, (_, row) in enumerate(benchmark.iterrows()):
            targets = parse_targets(row["answer"])
            target_type = normalize(row["target_type"])
            target_signature = hashlib.sha256(json.dumps(targets).encode()).hexdigest()
            for hop in (1, 2):
                start_name = normalize(row.get(f"hop{hop}"))
                start_type = normalize(row.get(f"hop{hop}_type"))
                intermediate = normalize(row.get("hop1_type")) if hop == 2 else ""
                question = normalize(row.get(f"hop{hop}_question"))
                type_path = [start_type] + ([intermediate] if hop == 2 else []) + [target_type]
                relation = normalize(row.get(f"relation_hop{hop}"))
                if relation and all(type_path) and relation != ":".join(type_path):
                    raise ValueError(f"Sample {idx}, hop {hop}: relation path {relation!r} "
                                     f"does not match node types {type_path!r}")
                applicable = bool(question and start_name and all(type_path) and targets)
                supported = all(t in supported_types for t in type_path)
                relation_supported = kg.supports_path(type_path, relation_policy)
                starts = kg.resolve(start_name, start_type) if applicable else set()
                predicted_ids = set()
                if not question:
                    status = "missing_question"
                elif not start_name or not all(type_path):
                    status = "missing_fields"
                elif not targets:
                    status = "empty_targets"
                elif not supported:
                    status = "unsupported_type"
                elif not relation_supported:
                    status = "unsupported_relation"
                elif not starts:
                    status = "start_not_found"
                else:
                    predicted_ids = kg.query_ids(start_name, start_type, target_type, intermediate or None, relation_policy)
                    status = "ok" if predicted_ids else "no_candidates"
                names = kg.answer_names(predicted_ids)
                metrics = compute_metrics(names, targets)
                if not applicable:
                    metrics = {k: None for k in metrics}
                query_signature = hashlib.sha256(json.dumps(
                    [hop, question, type_path, target_signature]).encode()).hexdigest()
                result = {
                    "sample_idx": idx, "hop": hop, "question_id": query_signature,
                    "source_name": start_name, "source_type": start_type,
                    "intermediate_type": intermediate, "target_type": target_type,
                    "status": status, "applicable": applicable, "supported_types": supported,
                    "supported_relations": relation_supported, "relation_policy": relation_policy,
                    "start_node_exists": bool(starts), "n_start_matches": len(starts),
                    "n_predicted_nodes": len(predicted_ids),
                    "n_unlabelled_predictions": sum(not kg.names[n] for n in predicted_ids),
                    "n_targets_present": sum(kg.node_exists(t, target_type) for t in targets),
                    **metrics,
                }
                results.append(result)
                if evidence:
                    evidence.write(json.dumps({
                        "sample_idx": idx, "hop": hop, "status": status,
                        "targets": targets,
                        "predictions": [
                            {"id": kg.external_ids[n], "name": kg.names[n]}
                            for n in sorted(predicted_ids, key=lambda n: kg.external_ids[n])
                        ],
                    }, ensure_ascii=False) + "\n")
            if (idx + 1) % 500 == 0 or idx + 1 == len(benchmark):
                print(f"  {kg.name}: {idx + 1:,}/{len(benchmark):,} samples", flush=True)
    finally:
        if evidence:
            evidence.close()
    return pd.DataFrame(results)


def summarize(frame):
    valid = frame[frame["applicable"]]
    supported = valid[valid["supported_types"] & valid["supported_relations"]]
    matched = valid[valid["start_node_exists"]]
    unique = valid.drop_duplicates("question_id")

    def mean(df, column):
        return float(df[column].mean()) if len(df) else None

    return {
        "n_rows": len(frame), "n_evaluated": len(valid), "n_excluded": len(frame) - len(valid),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "n_supported": len(supported), "n_start_found": len(matched),
        "n_ambiguous_starts": int((valid["n_start_matches"] > 1).sum()),
        "start_coverage": mean(valid, "start_node_exists"),
        "hit_rate": mean(valid, "hit"), "recall": mean(valid, "recall"),
        "precision": mean(valid, "precision"), "f1": mean(valid, "f1"),
        "mean_predictions": mean(valid, "n_predicted"),
        "supported_only_hit_rate": mean(supported, "hit"),
        "start_found_only_hit_rate": mean(matched, "hit"),
        "n_unique_question_answer_pairs": len(unique), "unique_pair_hit_rate": mean(unique, "hit"),
    }


def save_results(results, run_dir, benchmark_df, kg, max_samples, plots=False):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "per_question_metrics.csv"
    results.to_csv(metrics_path, index=False)
    groups = []
    for key, subset in results.groupby(["hop", "source_type", "intermediate_type", "target_type"], dropna=False):
        summary = summarize(subset)
        groups.append(dict(zip(["hop", "source_type", "intermediate_type", "target_type"], key),
                           **{k: v for k, v in summary.items() if k != "status_counts"}))
    pd.DataFrame(groups).to_csv(run_dir / "metrics_by_type_path.csv", index=False)
    references = benchmark_df["answer"].apply(lambda values: tuple(parse_targets(values)))
    benchmark_checks = {}
    for hop in (1, 2):
        questions = benchmark_df[f"hop{hop}_question"].apply(normalize)
        variants = pd.DataFrame({"question": questions, "references": references})
        variants = variants[variants["question"] != ""].groupby("question")["references"].nunique()
        benchmark_checks[str(hop)] = {"unique_question_texts": len(variants),
                                      "questions_with_differing_answer_sets": int((variants > 1).sum())}
    summary = {
        "completed_at": utc_now(), "protocol": PROTOCOL, "graph": kg.metadata(),
        "relation_policy": str(results["relation_policy"].iloc[0]),
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ("pandas", "numpy", "requests", "pyarrow")}},
        "benchmark_checks": benchmark_checks,
        "benchmark": benchmark_df.attrs.get("provenance", {}),
        "n_available_samples": len(benchmark_df), "n_evaluated_samples": int(results["sample_idx"].nunique()),
        "requested_max_samples": max_samples,
        "scope": "full" if results["sample_idx"].nunique() == len(benchmark_df) else "first-n",
        "hops": {str(hop): summarize(group) for hop, group in results.groupby("hop")},
        "code_sha256": {name: sha256_file(Path(__file__).parent / name)
                        for name in ("evaluate_kg.py", "kg_loader.py", "utils.py", "sources.py", "visualize_kg.py")},
        "outputs": {},
    }
    if plots:
        try:
            from .visualize_kg import generate_dashboard
        except ImportError:
            from visualize_kg import generate_dashboard
        generate_dashboard(results, run_dir / "plots" / "retrieval_dashboard.png", kg.name)
    summary["outputs"] = {
        str(p.relative_to(run_dir)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "summary.json"
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", nargs="+", choices=list(KG_REGISTRY) + ["all"], default=["primekg"])
    parser.add_argument("--dataset", default="knowlab-research/BioHopR")
    parser.add_argument("--split")
    parser.add_argument("--revision", help="Dataset revision; BioHopR defaults to a pinned commit")
    parser.add_argument("--max-samples", type=int, help="First N samples; written to a separate run folder")
    parser.add_argument("--results-dir", type=Path, help="Override the KG evaluation results root")
    parser.add_argument("--plots", action="store_true", help="Save one dashboard per graph")
    parser.add_argument("--save-predictions", action="store_true", help="Also save compressed candidate IDs/names")
    parser.add_argument("--relation-policy", choices=["semantic", "type-only"], default="semantic",
                        help="semantic respects treatment/positive phenotype relations; type-only reproduces the broad legacy baseline")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    graphs = list(KG_REGISTRY) if "all" in args.kg else list(dict.fromkeys(args.kg))
    benchmark = load_benchmark(args.dataset, args.split, args.revision)
    comparison = []
    for graph_name in graphs:
        kg = get_kg(graph_name)
        run_dir = create_run_dir(args.dataset, kg.name, args.max_samples, args.results_dir, args.relation_policy)
        try:
            results = evaluate_all(
                benchmark, kg, max_samples=args.max_samples,
                predictions_path=run_dir / "predictions.jsonl.gz" if args.save_predictions else None,
                relation_policy=args.relation_policy)
            summary = save_results(results, run_dir, benchmark, kg, args.max_samples, plots=args.plots)
        except Exception as exc:
            write_json(run_dir / "failed.json", {"failed_at": utc_now(), "error": str(exc)})
            raise
        for hop, values in summary["hops"].items():
            rate = values["hit_rate"]
            display = f"{rate:.2%}" if rate is not None else "N/A"
            print(f"{kg.name} {hop}-hop: hit={display}, n={values['n_evaluated']:,}")
            comparison.append({"graph": kg.name, "hop": int(hop), "relation_policy": args.relation_policy,
                               "n_evaluated": values["n_evaluated"], "hit_rate": values["hit_rate"],
                               "recall": values["recall"], "precision": values["precision"], "f1": values["f1"],
                               "start_coverage": values["start_coverage"], "n_supported": values["n_supported"],
                               "summary": str(Path("..") / ".." / kg.name / run_dir.name / "summary.json")})
        print(f"Results: {run_dir}", flush=True)
        del kg, results
        gc.collect()
    if len(graphs) > 1:
        comparison_dir = create_run_dir(args.dataset, "comparison", args.max_samples, args.results_dir, args.relation_policy)
        pd.DataFrame(comparison).to_csv(comparison_dir / "graph_comparison.csv", index=False)
        print(f"Comparison: {comparison_dir}", flush=True)


if __name__ == "__main__":
    main()

