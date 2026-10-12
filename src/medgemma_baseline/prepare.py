"""Prepare BioHopR questions without downloading or loading knowledge graphs."""
import argparse
import json
import tempfile
from pathlib import Path

from src.kg_eval.utils import load_benchmark, sha256_file, utc_now, write_json
from src.medgemma_eval.prepare import build_questions


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "BioHopR"


def experiment_path(max_samples=None):
    return RESULTS_DIR / ("full" if max_samples is None else f"first_{max_samples}_rows")


def prepare_baseline(experiment_dir, max_samples=None, revision=None):
    directory = Path(experiment_dir).resolve()
    benchmark = load_benchmark(revision=revision)
    questions = build_questions(benchmark, max_samples)
    src = Path(__file__).resolve().parents[1]
    request = {
        "task": "biohopr_medgemma_model_only_v1",
        "benchmark": benchmark.attrs["provenance"],
        "selection": {"method": "all" if max_samples is None else "first-n",
                      "max_samples": max_samples, "n_available_samples": len(benchmark)},
        "code_sha256": {name: sha256_file(src / name) for name in (
            "medgemma_baseline/prepare.py", "medgemma_eval/prepare.py", "kg_eval/utils.py")},
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preparation_request") != request:
            raise ValueError("Existing preparation has different data, selection, or code; "
                             "choose a new --experiment-dir to preserve its results")
        question_path = directory / "questions.jsonl"
        if not question_path.is_file() or sha256_file(question_path) != manifest["questions_sha256"]:
            raise ValueError("Prepared questions are missing or modified; choose a new --experiment-dir")
        if manifest.get("kgs") != [] or manifest.get("contexts") != {}:
            raise ValueError("This experiment must contain only the model-only baseline")
        print(f"Prepared baseline already matches: {directory}", flush=True)
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Experiment directory is nonempty without a manifest; choose a new --experiment-dir")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.prepare-", dir=directory.parent) as temp:
        staging = Path(temp)
        question_path = staging / "questions.jsonl"
        with question_path.open("w", encoding="utf-8", newline="\n") as stream:
            for question in questions:
                stream.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")
        record = {"file": "questions.jsonl", "sha256": sha256_file(question_path)}
        manifest = {
            "schema_version": "1.0", "created_at": utc_now(),
            "benchmark": request["benchmark"], "selection": request["selection"],
            "n_samples": len(questions) // 2, "n_questions": len(questions),
            "n_applicable_questions": sum(q["applicable"] for q in questions),
            "kgs": [], "contexts": {}, "retrieval": {"enabled": False},
            "question_file": record, "questions_file": record["file"],
            "questions_sha256": record["sha256"], "preparation_request": request,
            "code_sha256": request["code_sha256"],
        }
        write_json(staging / "manifest.json", manifest)
        if directory.exists():
            directory.rmdir()  # Only an empty directory can be replaced.
        staging.replace(directory)
    print(f"Prepared {manifest['n_samples']:,} rows / {len(questions):,} questions: {directory}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--max-samples", type=int, help="Dataset rows; each row yields two questions")
    parser.add_argument("--dataset-revision", help="Defaults to the project's pinned BioHopR revision")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    prepare_baseline(args.experiment_dir or experiment_path(args.max_samples),
                     args.max_samples, args.dataset_revision)


if __name__ == "__main__":
    main()
