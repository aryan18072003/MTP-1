"""Generate resumable single-answer predictions for a prepared BioHopR experiment."""
import argparse
import json
import os
import re
import time
from pathlib import Path

from src.kg_eval.utils import sha256_file, utc_now, write_json
from .model import DEFAULT_MODEL, PROMPT_VERSION, HuggingFaceBackend, build_messages, resolve_revision, runtime_versions


def read_jsonl(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["question_id"]
            if key in seen:
                raise ValueError(f"Duplicate question_id {key!r} in {path}")
            seen.add(key)
            rows.append(row)
    return rows


def load_prepared(experiment_dir):
    directory = Path(experiment_dir).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in [manifest["question_file"], *manifest["contexts"].values()]:
        path = (directory / record["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Prepared files must be inside the experiment directory")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared data was modified: {path}")
    questions = read_jsonl(directory / manifest["question_file"]["file"])
    if len(questions) != manifest["n_questions"]:
        raise ValueError("Question count differs from preparation manifest")
    return manifest, questions


def load_contexts(directory, manifest, variant, questions):
    if variant == "baseline":
        return {q["question_id"]: {"status": "baseline", "evidence": [],
                                  "evidence_truncated": False} for q in questions}
    rows = read_jsonl(Path(directory) / manifest["contexts"][variant]["file"])
    if {r["question_id"] for r in rows} != {q["question_id"] for q in questions}:
        raise ValueError(f"{variant} contexts and questions have different IDs")
    return {r["question_id"]: r for r in rows}


def read_checkpoint(path):
    """Recover an interrupted last write, preserving its bytes for inspection."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    offset = 0
    rows = []
    seen = set()
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if offset + len(line) != len(raw) or line.endswith(b"\n"):
                raise ValueError(f"Corrupt completed checkpoint line in {path}") from None
            backup = path.with_suffix(f".interrupted-{time.time_ns()}.bin")
            backup.write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            print(f"Recovered incomplete final record; original tail saved in {backup}", flush=True)
            break
        key = row["question_id"]
        if key in seen:
            raise ValueError(f"Duplicate checkpoint question_id {key!r} in {path}")
        seen.add(key)
        if row.get("status") not in {"ok", "excluded"}:
            raise ValueError(f"Unsupported checkpoint status for {key}; inspect {path}")
        rows.append(row)
        offset += len(line)
        if offset == len(raw) and not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def _append(stream, rows):
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def validate_checkpoint(previous, questions):
    """Validate partial and complete checkpoints against the same question schema."""
    known = {q["question_id"]: q for q in questions}
    for row in previous:
        key = row["question_id"]
        if key not in known or row["sample_idx"] != known[key]["sample_idx"] or row["hop"] != known[key]["hop"]:
            raise ValueError(f"Checkpoint row does not match prepared question: {key}")
        expected = "ok" if known[key]["applicable"] else "excluded"
        if row["status"] != expected:
            raise ValueError(f"Checkpoint applicability differs for question: {key}")


def generate_variant(questions, contexts, output_path, backend, batch_size):
    """Checkpoint each completed batch. Failed batches are retried on restart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_checkpoint(output_path)
    validate_checkpoint(previous, questions)
    done = {row["question_id"] for row in previous}
    pending = [q for q in questions if q["question_id"] not in done]
    started = time.monotonic()
    completed = len(previous)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            applicable = [q for q in batch if q["applicable"]]
            prompts = [backend.fit({"question": q["question"], "target_type": q["target_type"]},
                                   contexts[q["question_id"]]["evidence"]) for q in applicable]
            predictions = backend.generate(prompts) if prompts else []
            if len(predictions) != len(prompts):
                raise ValueError("Backend returned an incorrect number of predictions")
            generated = {q["question_id"]: (p, answer) for q, p, answer in
                         zip(applicable, prompts, predictions)}
            records = []
            for q in batch:
                context = contexts[q["question_id"]]
                record = {"question_id": q["question_id"], "sample_idx": q["sample_idx"], "hop": q["hop"],
                          "status": "ok" if q["applicable"] else "excluded", "prediction": "",
                          "context_status": context["status"], "evidence_paths_used": 0,
                          "input_tokens": 0, "output_tokens": 0,
                          "retrieval_evidence_truncated": context.get("evidence_truncated", False)}
                if q["applicable"]:
                    fitted, answer = generated[q["question_id"]]
                    record.update(answer)
                    record.update(input_tokens=fitted.input_tokens, prompt_sha256=fitted.prompt_sha256,
                                  evidence_paths_used=fitted.evidence_paths_used,
                                  evidence_paths_available=fitted.evidence_paths_available,
                                  prompt_evidence_truncated=fitted.evidence_paths_used < fitted.evidence_paths_available)
                else:
                    record["exclusion_reason"] = q["exclusion_reason"]
                records.append(record)
            _append(stream, records)
            completed += len(records)
            if start == 0 or completed % 100 < batch_size or completed == len(questions):
                print(f"  {output_path.parent.name}: {completed:,}/{len(questions):,} "
                      f"({time.monotonic() - started:.0f}s this session)", flush=True)
    return completed


def config_for(args, manifest, revision):
    return {
        "model": args.model, "requested_revision": args.revision, "revision": revision,
        "quantization": args.quantization, "compute_dtype": "bfloat16", "attention": "sdpa",
        "device": args.device, "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
        "seed": args.seed, "do_sample": False, "num_beams": 1,
        "min_free_gib": args.min_free_gib if args.min_free_gib is not None else
                        (22 if args.quantization == "nf4" else 65),
        "variants": ["baseline", *manifest["kgs"]], "prompt_version": PROMPT_VERSION,
        "implementation_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                   for name in ("model.py", "run.py")},
        "runtime_versions": runtime_versions(),
    }


def run_experiment(args, backend_factory=HuggingFaceBackend, revision_resolver=resolve_revision):
    from filelock import FileLock

    experiment = args.experiment_dir.resolve()
    manifest, questions = load_prepared(experiment)
    all_variants = ["baseline", *manifest["kgs"]]
    variants = list(dict.fromkeys(args.variants or all_variants))
    if not set(variants).issubset(all_variants):
        raise ValueError(f"Prepared variants available: {all_variants}")
    # Baseline is always run first when selected, independent of argument order.
    variants = [v for v in all_variants if v in variants]
    if args.preview:
        for variant in variants:
            contexts = load_contexts(experiment, manifest, variant, questions)
            for q in questions[:args.preview]:
                print(json.dumps({"variant": variant, "question_id": q["question_id"],
                                  "messages": build_messages(q["question"], q["target_type"],
                                                             contexts[q["question_id"]]["evidence"])},
                                 ensure_ascii=False))
        return
    run_dir = experiment / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_dir / ".generation.lock"), timeout=0):
        run_path = run_dir / "run.json"
        existing = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
        if existing is None and any(run_dir.glob("*/predictions.jsonl")):
            raise ValueError("Predictions exist without run.json; choose a new --run-name")
        revision = existing["config"]["revision"] if existing else revision_resolver(args.model, args.revision)
        if not revision:
            raise ValueError("Could not resolve an immutable model revision")
        config = config_for(args, manifest, revision)
        identity = {"config": config, "prepared_manifest_sha256": sha256_file(experiment / "manifest.json"),
                    "question_file_sha256": manifest["question_file"]["sha256"]}
        if existing and any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Model, data, code, or generation settings differ from the existing run. "
                             "Use the original settings or a new --run-name.")
        metadata = existing or {"schema_version": "1.0", **identity, "created_at": utc_now(),
                                "completed_variants": []}
        # Verify every checkpoint before spending GPU time, including completed runs.
        completed_variants = []
        for variant in all_variants:
            path = run_dir / variant / "predictions.jsonl"
            previous = read_checkpoint(path)
            validate_checkpoint(previous, questions)
            if len(previous) == len(questions):
                completed_variants.append(variant)
        # The final prediction flush may have succeeded just before interruption.
        # Reconstruct completion from disk instead of trusting stale metadata.
        metadata["completed_variants"] = completed_variants
        pending_variants = [v for v in variants if v not in completed_variants]
        if not pending_variants:
            metadata.update(status="complete" if len(completed_variants) == len(all_variants) else "partial",
                            updated_at=utc_now())
            write_json(run_path, metadata)
            print(f"Requested variants already complete: {run_dir}", flush=True)
            return
        metadata.update(status="running", updated_at=utc_now())
        write_json(run_path, metadata)
        try:
            backend = backend_factory(config)
            metadata["hardware"] = getattr(backend, "metadata", {})
            write_json(run_path, metadata)
            for variant in pending_variants:
                contexts = load_contexts(experiment, manifest, variant, questions)
                generate_variant(questions, contexts, run_dir / variant / "predictions.jsonl",
                                 backend, args.batch_size)
                metadata["completed_variants"] = list(dict.fromkeys(metadata["completed_variants"] + [variant]))
                metadata["updated_at"] = utc_now()
                write_json(run_path, metadata)
            metadata.update(status="complete" if set(all_variants) <= set(metadata["completed_variants"]) else "partial",
                            updated_at=utc_now())
            write_json(run_path, metadata)
        except BaseException as error:
            metadata.update(status="interrupted", last_error_type=type(error).__name__, updated_at=utc_now())
            write_json(run_path, metadata)
            raise
    print(f"Predictions saved: {run_dir}", flush=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="nf4")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "primekg", "hetionet", "drkg", "biokg"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gib", type=float)
    parser.add_argument("--preview", type=int, default=0,
                        help="Print N un-tokenized prompts per variant, without loading a model or writing results")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores and hyphens")
    if min(args.batch_size, args.max_input_tokens, args.max_new_tokens) <= 0 or args.preview < 0:
        parser.error("Batch/token counts must be positive; preview must be nonnegative")
    if args.min_free_gib is not None and args.min_free_gib < 0:
        parser.error("--min-free-gib must be nonnegative")
    run_experiment(args)


if __name__ == "__main__":
    main()
