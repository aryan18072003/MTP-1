"""Compare standalone and KG-assisted MedGemma answers on identical questions.

This compares LLM single-answer accuracy, not the graph-only evaluator's
multi-answer retrieval hit/precision/recall. By default every applicable question
must have successful inference in every declared variant. --allow-incomplete
explicitly limits comparison to the shared successfully generated intersection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .scoring import atomic_csv, atomic_json, index_questions, load_experiment, sha256_file

COMPARISON_FIELDS = [
    "variant", "hop", "comparison_scope", "n_questions", "n_applicable", "n_compared",
    "n_success", "n_errors", "n_missing", "n_excluded", "n_omitted_incomplete",
    "exact_correct", "exact_accuracy", "exact_accuracy_pct", "exact_delta_baseline_pp",
    "cosine_correct", "cosine_accuracy", "cosine_accuracy_pct", "cosine_delta_baseline_pp",
]
TYPE_FIELDS = ["source_type", "intermediate_type", "target_type"]


def _read_metrics(path: Path, questions: dict, scoring: dict) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question_id", "hop", "sample_idx", "applicable", "status",
                    "exact_correct", "cosine_correct", "max_similarity", *TYPE_FIELDS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing metric columns in {path}")
        metrics = {}
        for row in reader:
            question_id = row["question_id"]
            if question_id not in questions or question_id in metrics:
                raise ValueError(f"Unknown or duplicate metric question_id {question_id} in {path}")
            question = questions[question_id]
            if row["hop"] != str(question["hop"]) or row["sample_idx"] != str(question["sample_idx"]):
                raise ValueError(f"Metric identity mismatch for {question_id} in {path}")
            if row["applicable"] != str(int(question["applicable"])):
                raise ValueError(f"Metric applicability mismatch for {question_id} in {path}")
            for field in TYPE_FIELDS:
                if row[field] != str(question.get(field) or ""):
                    raise ValueError(f"Metric {field} mismatch for {question_id} in {path}")
            if row["status"] not in {"ok", "error", "missing", "excluded"}:
                raise ValueError(f"Invalid metric status for {question_id} in {path}")
            if question["applicable"] == (row["status"] == "excluded"):
                raise ValueError(f"Metric status/applicability mismatch for {question_id} in {path}")
            for field, enabled in (("exact_correct", scoring["scorer"] in {"exact", "both"}),
                                   ("cosine_correct", scoring["scorer"] in {"biolord", "both"})):
                value = row[field]
                if question["applicable"] and enabled:
                    if value not in {"0", "1"}:
                        raise ValueError(f"Missing/invalid {field} for {question_id} in {path}")
                    row[field] = int(value)
                    if row["status"] != "ok" and row[field] != 0:
                        raise ValueError(f"Failed inference cannot have positive {field}: {question_id}")
                elif value:
                    raise ValueError(f"Unexpected {field} for excluded/unscored {question_id} in {path}")
                else:
                    row[field] = None
            if row["max_similarity"] and not math.isfinite(float(row["max_similarity"])):
                raise ValueError(f"Nonfinite similarity for {question_id} in {path}")
            metrics[question_id] = row
    if set(metrics) != set(questions):
        raise ValueError(f"Incomplete metrics in {path}; rerun scoring")
    return metrics


def _aggregate(variant: str, hop: str, scope: str, question_ids: list[str],
               comparable: set[str], questions: dict, metrics: dict, baseline: dict) -> dict:
    applicable = [qid for qid in question_ids if questions[qid]["applicable"]]
    selected = [qid for qid in applicable if qid in comparable]
    statuses = Counter(metrics[qid]["status"] for qid in question_ids)
    row = {"variant": variant, "hop": hop, "comparison_scope": scope,
           "n_questions": len(question_ids), "n_applicable": len(applicable),
           "n_compared": len(selected), "n_success": statuses["ok"], "n_errors": statuses["error"],
           "n_missing": statuses["missing"], "n_excluded": statuses["excluded"],
           "n_omitted_incomplete": len(applicable) - len(selected)}
    for field, prefix in (("exact_correct", "exact"), ("cosine_correct", "cosine")):
        values = [metrics[qid][field] for qid in selected]
        enabled = bool(values) and all(value is not None for value in values)
        correct = sum(values) if enabled else None
        accuracy = correct / len(values) if enabled else None
        baseline_accuracy = sum(baseline[qid][field] for qid in selected) / len(selected) if enabled else None
        row.update({f"{prefix}_correct": correct, f"{prefix}_accuracy": accuracy,
                    f"{prefix}_accuracy_pct": accuracy * 100 if accuracy is not None else None,
                    f"{prefix}_delta_baseline_pp": (accuracy - baseline_accuracy) * 100 if enabled else None})
    return row


def _plot(output_dir: Path, rows: list[dict], variants: list[str], scoring: dict, scope: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scorers = []
    if scoring["scorer"] in {"exact", "both"}:
        scorers.append(("exact", "Exact name accuracy"))
    if scoring["scorer"] in {"biolord", "both"}:
        scorers.append(("cosine", f"BioLORD cosine > {scoring['threshold']:g}"))
    figure, axes = plt.subplots(1, len(scorers), figsize=(max(8, len(variants) * 1.5) * len(scorers), 5),
                                squeeze=False, constrained_layout=True)
    display = lambda name: "MedGemma" if name == "baseline" else "MedGemma + " + {
        "primekg": "PrimeKG", "hetionet": "Hetionet", "drkg": "DRKG", "biokg": "BioKG"}.get(name, name)
    indexed = {(row["variant"], row["hop"]): row for row in rows}
    for axis, (prefix, title) in zip(axes[0], scorers):
        for hop, offset, color in (("1", -0.19, "#2b6cb0"), ("2", 0.19, "#dd6b20")):
            values = [indexed[(variant, hop)][f"{prefix}_accuracy_pct"] for variant in variants]
            axis.bar([i + offset for i in range(len(variants))],
                     [value if value is not None else float("nan") for value in values],
                     width=0.36, color=color, label=f"{hop}-hop")
        axis.set_xticks(range(len(variants)), [display(v) for v in variants], rotation=20, ha="right")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Single-answer accuracy (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend()
    figure.suptitle("BioHopR: " + ("shared successful subset" if scope != "complete" else "all applicable questions"))
    temporary = output_dir / "comparison.png.part"
    figure.savefig(temporary, format="png", dpi=180)
    plt.close(figure)
    temporary.replace(output_dir / "comparison.png")


def compare_experiment(experiment_dir: Path, run_name: str, allow_incomplete: bool = False,
                       plots: bool = False) -> Path:
    run_dir, run, question_rows, provenance = load_experiment(experiment_dir, run_name)
    questions = index_questions(question_rows)
    variants = run["config"]["variants"]
    if "baseline" not in variants:
        raise ValueError("Comparison needs the baseline variant in run.json")
    all_metrics = {}
    summaries = {}
    common_scoring = None
    for variant in variants:
        directory = run_dir / variant
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"No scores for {variant}; run scoring for every declared variant first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("variant") != variant or summary.get("schema_version") != 1:
            raise ValueError(f"Wrong variant/schema in {summary_path}")
        saved_provenance = summary.get("provenance", {})
        for field in ("prepared_manifest_sha256", "question_file_sha256", "run_config_sha256"):
            if saved_provenance.get(field) != provenance[field]:
                raise ValueError(f"Stale {variant} scores: {field} differs; rerun scoring")
        for filename, hash_field in (("predictions.jsonl", "predictions_sha256"), ("metrics.csv", "metrics_sha256")):
            if sha256_file(directory / filename) != saved_provenance.get(hash_field):
                raise ValueError(f"Stale or modified {variant} {filename}; rerun scoring")
        scoring = summary.get("scoring")
        if not isinstance(scoring, dict) or scoring.get("scorer") not in {"exact", "biolord", "both"}:
            raise ValueError(f"Invalid scoring configuration in {summary_path}")
        if common_scoring is None:
            common_scoring = scoring
        elif scoring != common_scoring:
            raise ValueError("Scoring configurations differ across variants; rerun all with identical scorer settings")
        all_metrics[variant] = _read_metrics(directory / "metrics.csv", questions, scoring)
        summaries[variant] = summary
    applicable_ids = {qid for qid, question in questions.items() if question["applicable"]}
    if not applicable_ids:
        raise ValueError("No applicable questions to compare")
    comparable = applicable_ids.copy()
    failures = []
    for variant, metrics in all_metrics.items():
        successful = {qid for qid in applicable_ids if metrics[qid]["status"] == "ok"}
        comparable.intersection_update(successful)
        if successful != applicable_ids:
            failures.append(f"{variant}: {len(applicable_ids - successful)} missing/error answers")
    if failures and not allow_incomplete:
        raise ValueError("Incomplete inference (" + "; ".join(failures) +
                         "). Resume generation, rescore, or explicitly use --allow-incomplete for the shared successful subset")
    if applicable_ids and not comparable:
        raise ValueError("No shared successfully generated applicable questions to compare")
    scope = "shared_successful_intersection" if failures else "complete"
    rows = []
    type_rows = []
    type_groups = {}
    for question_id, question in questions.items():
        key = (str(question["hop"]), *(str(question.get(field) or "") for field in TYPE_FIELDS))
        type_groups.setdefault(key, []).append(question_id)
    for variant in variants:
        for hop in ("1", "2", "overall"):
            ids = [qid for qid, question in questions.items() if hop == "overall" or str(question["hop"]) == hop]
            rows.append(_aggregate(variant, hop, scope, ids, comparable, questions, all_metrics[variant], all_metrics["baseline"]))
        for key, ids in sorted(type_groups.items()):
            row = _aggregate(variant, key[0], scope, ids, comparable, questions, all_metrics[variant], all_metrics["baseline"])
            row.update(zip(TYPE_FIELDS, key[1:]))
            type_rows.append(row)
    output_dir = run_dir / "comparison"
    atomic_csv(output_dir / "comparison.csv", rows, COMPARISON_FIELDS)
    atomic_csv(output_dir / "by_type.csv", type_rows, [*COMPARISON_FIELDS[:3], *TYPE_FIELDS, *COMPARISON_FIELDS[3:]])
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "comparison_scope": scope, "allow_incomplete": allow_incomplete, "scoring": common_scoring,
                "n_applicable": len(applicable_ids), "n_compared": len(comparable),
                "n_omitted_incomplete": len(applicable_ids - comparable),
                "compared_question_ids_sha256": hashlib.sha256(json.dumps(sorted(comparable)).encode()).hexdigest(),
                "provenance": provenance,
                "variant_prediction_sha256": {v: summaries[v]["provenance"]["predictions_sha256"] for v in variants},
                "comparison_csv_sha256": sha256_file(output_dir / "comparison.csv"),
                "by_type_csv_sha256": sha256_file(output_dir / "by_type.csv"),
                "notes": ["Accuracy is single-answer LLM accuracy; graph-only retrieval metrics are a different task.",
                          "No shared disease ontology alignment is performed by these name-based scorers.",
                          "Incomplete comparisons use the same successful question subset for every variant and may be selection-biased."]}
    if plots:
        _plot(output_dir, rows, variants, common_scoring, scope)
        manifest["comparison_png_sha256"] = sha256_file(output_dir / "comparison.png")
    else:
        # A plot from earlier scores would otherwise look current beside new CSVs.
        (output_dir / "comparison.png").unlink(missing_ok=True)
    atomic_json(output_dir / "comparison.json", manifest)
    print(f"Saved {output_dir / 'comparison.csv'} ({len(comparable)}/{len(applicable_ids)} applicable questions; {scope})")
    return output_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Compare only questions successfully generated by every variant; report omissions")
    parser.add_argument("--plots", action="store_true", help="Also write comparison.png (requires matplotlib)")
    args = parser.parse_args(argv)
    try:
        compare_experiment(args.experiment_dir, args.run_name, args.allow_incomplete, args.plots)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
