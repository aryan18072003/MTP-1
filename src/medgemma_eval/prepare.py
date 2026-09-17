"""Prepare identical BioHopR questions and separate graph evidence before inference.

Run from the repository root with ``python -m src.medgemma_eval.prepare``.
This command uses CPU/RAM and downloads graph data when it is not cached.
It does not import model libraries, load MedGemma, or require a GPU.
"""
import argparse
import gc
import json
import tempfile
from collections import Counter
from pathlib import Path

from src.kg_eval.kg_loader import KG_REGISTRY, get_kg
from src.kg_eval.utils import load_benchmark, normalize, parse_targets, sha256_file, utc_now, write_json
from .retrieval import retrieval_key, retrieve_context


SCHEMA_VERSION = "1.0"
DEFAULT_KGS = ["primekg", "hetionet", "drkg", "biokg"]


def build_questions(benchmark, max_samples=None):
    """Flatten each benchmark row into two questions without exposing its bridge."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    selected = benchmark.iloc[:max_samples] if max_samples is not None else benchmark
    if selected.empty:
        raise ValueError("Cannot prepare an empty benchmark")
    questions = []
    for sample_idx, (_, row) in enumerate(selected.iterrows()):
        targets = parse_targets(row["answer"])
        for hop in (1, 2):
            original_question = row.get(f"hop{hop}_question")
            question_text = str(original_question) if normalize(original_question) else ""
            source_name = normalize(row.get(f"hop{hop}"))
            source_type = normalize(row.get(f"hop{hop}_type"))
            intermediate_type = normalize(row.get("hop1_type")) if hop == 2 else ""
            target_type = normalize(row.get("target_type"))
            types = [source_type] + ([intermediate_type] if hop == 2 else []) + [target_type]
            relation = normalize(row.get(f"relation_hop{hop}"))
            if relation and all(types) and relation != ":".join(types):
                raise ValueError(f"Sample {sample_idx}, hop {hop}: relation {relation!r} disagrees with {types!r}")
            reason = ("missing_question" if not question_text else
                      "missing_fields" if not source_name or not all(types) else
                      "empty_targets" if not targets else "")
            questions.append({
                "question_id": f"{sample_idx}:{hop}", "sample_idx": sample_idx, "hop": hop,
                "question": question_text, "source_name": source_name, "source_type": source_type,
                "intermediate_type": intermediate_type, "target_type": target_type,
                "targets": targets, "applicable": not bool(reason), "exclusion_reason": reason,
            })
    return questions


def _json_line(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def _write_questions(path, questions):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in questions:
            stream.write(_json_line(row))


def code_hashes():
    src = Path(__file__).resolve().parents[1]
    return {name: sha256_file(src / name) for name in (
        "medgemma_eval/prepare.py", "medgemma_eval/retrieval.py",
        "kg_eval/kg_loader.py", "kg_eval/sources.py", "kg_eval/utils.py")}


def _verify_existing(output_dir, request):
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"{output_dir} is nonempty but has no preparation manifest; choose a new experiment directory")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preparation_request") != request:
        raise ValueError("Existing experiment was prepared with different data, code, or retrieval settings; "
                         "choose a new --output-dir (existing generation results are preserved)")
    records = [manifest["question_file"], *manifest["contexts"].values()]
    for record in records:
        path = output_dir / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared file is missing or modified: {path}; choose a new experiment directory")
    print(f"Prepared experiment already matches: {output_dir}", flush=True)
    return manifest


def prepare_experiment(output_dir, kgs=None, max_samples=None, max_evidence_paths=24,
                       relation_policy="semantic", dataset="knowlab-research/BioHopR", split=None,
                       revision=None):
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if max_evidence_paths <= 0:
        raise ValueError("max_evidence_paths must be positive")
    if relation_policy not in {"semantic", "type-only"}:
        raise ValueError("Unknown relation policy")
    kgs = list(dict.fromkeys(DEFAULT_KGS if kgs is None else kgs))
    if not kgs or any(name not in KG_REGISTRY for name in kgs):
        raise ValueError(f"Choose one or more graphs from {list(KG_REGISTRY)}")
    output_dir = Path(output_dir).resolve()
    benchmark = load_benchmark(dataset, split, revision)
    questions = build_questions(benchmark, max_samples)
    n_samples = len(questions) // 2
    retrieval = {
        "max_evidence_paths": max_evidence_paths, "relation_policy": relation_policy,
        "selection": "one representative per endpoint ID; paths and endpoints in native-ID order",
        "matching": "normalized preferred names; no shared disease ontology or synonym alignment",
        "ranking_uses_reference_answers": False,
        "evidence_predicate": "associated with; semantic filters applied before traversal",
        "direction": "undirected traversal of source adapter connections",
        "n_candidates_unit": "distinct reachable endpoint node IDs before evidence cap",
        "evidence_truncated_unit": "endpoint representatives omitted by evidence cap",
    }
    selection = {"method": "first-n" if max_samples is not None else "all",
                 "max_samples": max_samples, "n_available_samples": len(benchmark)}
    request = {"schema_version": SCHEMA_VERSION, "benchmark": benchmark.attrs.get("provenance", {}),
               "kgs": kgs, "selection": selection, "retrieval": retrieval, "code_sha256": code_hashes()}
    existing = _verify_existing(output_dir, request)
    if existing is not None:
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Assemble in a new sibling directory. Failed preparations cannot corrupt an
    # existing experiment or leave apparently complete files in its directory.
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.prepare-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        question_path = staging / "questions.jsonl"
        _write_questions(question_path, questions)
        question_record = {"file": "questions.jsonl", "sha256": sha256_file(question_path)}
        (staging / "contexts").mkdir()
        contexts = {}
        for graph_name in kgs:
            print(f"Preparing evidence: {graph_name}", flush=True)
            graph = get_kg(graph_name)
            supported_types = graph.supported_types
            context_path = staging / "contexts" / f"{graph_name}.jsonl"
            cache = {}
            statuses = Counter()
            with context_path.open("w", encoding="utf-8", newline="\n") as stream:
                for idx, question in enumerate(questions):
                    key = retrieval_key(question)
                    if key not in cache:
                        cache[key] = retrieve_context(question, graph, max_evidence_paths, relation_policy,
                                                      supported_types=supported_types)
                    context = dict(cache[key], question_id=question["question_id"])
                    statuses[context["status"]] += 1
                    stream.write(_json_line(context))
                    if (idx + 1) % 1000 == 0 or idx + 1 == len(questions):
                        print(f"  {graph_name}: {idx + 1:,}/{len(questions):,} questions", flush=True)
            contexts[graph_name] = {"file": f"contexts/{graph_name}.jsonl", "sha256": sha256_file(context_path),
                                    "graph": graph.metadata(), "status_counts": dict(statuses)}
            del graph, cache
            gc.collect()
        manifest = {
            "schema_version": SCHEMA_VERSION, "created_at": utc_now(), "benchmark": request["benchmark"],
            "n_samples": n_samples, "n_questions": len(questions),
            "n_applicable_questions": sum(q["applicable"] for q in questions),
            "selection": selection, "retrieval": retrieval, "kgs": kgs,
            "questions_file": question_record["file"], "questions_sha256": question_record["sha256"],
            "question_file": question_record, "contexts": contexts,
            "code_sha256": request["code_sha256"], "preparation_request": request,
        }
        write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            # rmdir only succeeds for an empty directory; never recursively
            # remove or overwrite an existing experiment.
            output_dir.rmdir()
        staging.replace(output_dir)
    print(f"Prepared {n_samples:,} samples / {len(questions):,} questions: {output_dir}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--experiment-dir", required=True, type=Path)
    parser.add_argument("--kgs", nargs="+", choices=list(KG_REGISTRY), default=DEFAULT_KGS)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-evidence-paths", type=int, default=24)
    parser.add_argument("--relation-policy", choices=["semantic", "type-only"], default="semantic")
    parser.add_argument("--dataset", default="knowlab-research/BioHopR")
    parser.add_argument("--split")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_evidence_paths <= 0:
        parser.error("--max-evidence-paths must be positive")
    prepare_experiment(**vars(args))


if __name__ == "__main__":
    main()
