"""Score one generated entity per BioHopR question, independently of retrieval.

Exact scoring only case-folds and collapses whitespace. It does not extract lists,
strip punctuation, match substrings, or map disease codes. Semantic scoring embeds
the complete response and accepts max cosine similarity strictly above threshold,
matching the original BioLORD baseline's rule. It is an approximate name score,
not a verified ontology/concept match.

Run ``python -m src.medgemma_eval.scoring --help`` from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EMBEDDER = "FremyCompany/BioLORD-2023-C"
EXACT_RULE = "casefold_collapse_whitespace_full_response_v1"
METRIC_FIELDS = [
    "question_id", "sample_idx", "hop", "source_type", "intermediate_type",
    "target_type", "applicable", "status", "prediction", "normalized_prediction",
    "n_targets", "exact_correct", "cosine_correct", "max_similarity",
    "context_status", "evidence_paths_used", "input_tokens", "output_tokens", "error",
]


def normalize_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def exact_correct(prediction: str, targets: list[str]) -> bool:
    answer = normalize_name(prediction)
    return bool(answer) and answer in {normalize_name(target) for target in targets}


def cosine_correct(similarity: float, threshold: float) -> bool:
    return math.isfinite(similarity) and similarity > threshold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}; finish generation first") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(row)
    return rows, hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def index_questions(questions: list[dict]) -> dict[str, dict]:
    indexed = {}
    for question in questions:
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Each question needs a nonempty question_id")
        if question_id in indexed:
            raise ValueError(f"Duplicate question_id: {question_id}")
        if question.get("hop") not in (1, 2) or not isinstance(question.get("applicable"), bool):
            raise ValueError(f"Invalid hop/applicable for {question_id}")
        targets = question.get("targets")
        if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
            raise ValueError(f"Invalid targets for {question_id}")
        if question["applicable"] and not any(normalize_name(t) for t in targets):
            raise ValueError(f"Applicable question without reference answers: {question_id}")
        indexed[question_id] = question
    if not indexed:
        raise ValueError("questions.jsonl is empty")
    return indexed


def index_predictions(predictions: list[dict], questions: dict[str, dict]) -> dict[str, dict]:
    indexed = {}
    for row in predictions:
        question_id = row.get("question_id")
        if question_id not in questions:
            raise ValueError(f"Prediction has unknown question_id: {question_id}")
        if question_id in indexed:
            raise ValueError(f"Duplicate prediction: {question_id}")
        question = questions[question_id]
        if row.get("hop") != question["hop"] or row.get("sample_idx") != question["sample_idx"]:
            raise ValueError(f"Prediction/question identity mismatch: {question_id}")
        if row.get("status") not in {"ok", "excluded", "error"}:
            raise ValueError(f"Unknown prediction status for {question_id}: {row.get('status')}")
        if question["applicable"] == (row["status"] == "excluded"):
            raise ValueError(f"Prediction applicability mismatch: {question_id}")
        if not isinstance(row.get("prediction", ""), str):
            raise ValueError(f"Prediction must be text: {question_id}")
        indexed[question_id] = row
    return indexed


def load_experiment(experiment_dir: Path, run_name: str) -> tuple[Path, dict, list[dict], dict]:
    run_dir = experiment_dir / "runs" / run_name
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    questions, question_hash = load_jsonl(experiment_dir / "questions.jsonl")
    index_questions(questions)
    manifest_hash = sha256_file(experiment_dir / "manifest.json")
    if run.get("question_file_sha256") != question_hash:
        raise ValueError("Run belongs to different questions.jsonl; prepare/use the matching experiment")
    if run.get("prepared_manifest_sha256") != manifest_hash:
        raise ValueError("Prepared manifest changed since generation; use the matching experiment")
    config = run.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("variants"), list):
        raise ValueError("run.json needs config.variants")
    variants = config["variants"]
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("run.json variants must be nonempty and unique")
    provenance = {"prepared_manifest_sha256": manifest_hash, "question_file_sha256": question_hash,
                  "run_config_sha256": config_sha256(config), "run_file_sha256": sha256_file(run_path)}
    return run_dir, run, questions, provenance


class EmbeddingScorer:
    """Cache normalized float32 vectors; dependencies load only for semantic scores."""

    def __init__(self, model: str = DEFAULT_EMBEDDER, device: str = "cpu",
                 revision: str | None = None, batch_size: int = 64, encoder=None):
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        self.batch_size = batch_size
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {"device": device, "trust_remote_code": False}
            if revision:
                kwargs["revision"] = revision
            encoder = SentenceTransformer(model, **kwargs)
        self.encoder = encoder
        self.cache: dict[str, Any] = {}
        self.resolved_revision = None
        try:
            self.resolved_revision = encoder._first_module().auto_model.config._commit_hash
        except (AttributeError, KeyError, TypeError):
            pass

    def preload(self, texts: list[str]) -> None:
        import numpy as np
        missing = list(dict.fromkeys(text for text in texts if text and text not in self.cache))
        # Limit peak vector allocation while retaining reusable target/prediction vectors.
        for offset in range(0, len(missing), 2048):
            batch = missing[offset:offset + 2048]
            vectors = np.asarray(self.encoder.encode(batch, batch_size=self.batch_size,
                                 convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError("Embedder returned unexpected vector dimensions")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or np.any(norms <= 0):
                raise ValueError("Embedder returned invalid or zero vectors")
            vectors = vectors / norms
            self.cache.update(zip(batch, vectors))

    def max_similarity(self, prediction: str, targets: list[str]) -> float:
        import numpy as np
        targets = list(dict.fromkeys(t for t in targets if t))
        if not prediction.strip() or not targets:
            return 0.0
        self.preload([prediction, *targets])
        similarities = np.stack([self.cache[t] for t in targets]) @ self.cache[prediction]
        return float(np.clip(np.max(similarities), -1.0, 1.0))


def score_rows(questions: list[dict], predictions: list[dict], scorer: str = "both",
               threshold: float = 0.9, embedder: EmbeddingScorer | None = None) -> list[dict]:
    if scorer not in {"exact", "biolord", "both"}:
        raise ValueError(f"Unknown scorer: {scorer}")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be between -1 and 1")
    indexed_questions = index_questions(questions)
    indexed_predictions = index_predictions(predictions, indexed_questions)
    semantic = scorer in {"biolord", "both"}
    exact = scorer in {"exact", "both"}
    if semantic and embedder is None:
        raise ValueError("An embedder is required for semantic scoring")
    if semantic:
        texts = []
        for question in questions:
            prediction = indexed_predictions.get(question["question_id"], {})
            if question["applicable"] and prediction.get("status") == "ok":
                texts.extend(question["targets"])
                if prediction.get("prediction", "").strip():
                    texts.append(prediction["prediction"])
        embedder.preload(texts)
    rows = []
    for question in questions:
        prediction = indexed_predictions.get(question["question_id"], {})
        applicable = question["applicable"]
        status = prediction.get("status", "missing") if applicable else "excluded"
        answer = prediction.get("prediction", "")
        similarity = None
        exact_hit = semantic_hit = None
        if applicable:
            if exact:
                exact_hit = int(status == "ok" and exact_correct(answer, question["targets"]))
            if semantic:
                similarity = embedder.max_similarity(answer, question["targets"]) if status == "ok" else None
                semantic_hit = int(status == "ok" and bool(answer.strip()) and cosine_correct(similarity, threshold))
        row = {key: question.get(key, "") for key in METRIC_FIELDS[:7]}
        row.update({"applicable": int(applicable), "status": status, "prediction": answer,
                    "normalized_prediction": normalize_name(answer), "n_targets": len(set(question["targets"])),
                    "exact_correct": exact_hit, "cosine_correct": semantic_hit, "max_similarity": similarity,
                    "context_status": prediction.get("context_status", ""),
                    "evidence_paths_used": prediction.get("evidence_paths_used", ""),
                    "input_tokens": prediction.get("input_tokens", ""), "output_tokens": prediction.get("output_tokens", ""),
                    "error": prediction.get("error", question.get("exclusion_reason", "") if not applicable else "")})
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    def summary(subset):
        applicable = [row for row in subset if row["applicable"]]
        result = {"n_questions": len(subset), "n_applicable": len(applicable),
                  "status_counts": dict(Counter(row["status"] for row in subset)),
                  "context_status_counts": dict(Counter(row["context_status"] for row in applicable)),
                  "n_with_evidence": sum(bool(row["evidence_paths_used"]) and int(row["evidence_paths_used"]) > 0
                                         for row in applicable)}
        for field, label in (("exact_correct", "exact_accuracy"), ("cosine_correct", "cosine_accuracy")):
            available = [row[field] for row in applicable if row[field] is not None]
            result[label] = sum(available) / len(applicable) if applicable and available else None
        return result
    return {"overall": summary(rows), "by_hop": {str(hop): summary([r for r in rows if r["hop"] == hop])
                                               for hop in (1, 2)}}


def score_experiment(experiment_dir: Path, run_name: str, scorer: str = "both", threshold: float = 0.9,
                     embedder_model: str = DEFAULT_EMBEDDER, embedder_device: str = "cpu",
                     embedder_revision: str | None = None, embedder_batch_size: int = 64,
                     variants: list[str] | None = None) -> list[Path]:
    if scorer not in {"exact", "biolord", "both"}:
        raise ValueError(f"Unknown scorer: {scorer}")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be between -1 and 1")
    run_dir, run, questions, provenance = load_experiment(experiment_dir, run_name)
    variants = variants or run["config"]["variants"]
    if len(set(variants)) != len(variants) or not set(variants) <= set(run["config"]["variants"]):
        raise ValueError("Selected variants must be unique and declared in run.json")
    # Check every selected input before an expensive model download/load.
    loaded = {}
    for variant in variants:
        path = run_dir / variant / "predictions.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing {path}; finish generation or select --variants baseline")
        loaded[variant] = load_jsonl(path)
        index_predictions(loaded[variant][0], index_questions(questions))
    embedder = None
    if scorer in {"biolord", "both"}:
        print(f"Loading {embedder_model} on {embedder_device} for semantic scoring", flush=True)
        embedder = EmbeddingScorer(embedder_model, embedder_device, embedder_revision, embedder_batch_size)
    versions = {}
    if embedder:
        for package in ("sentence-transformers", "transformers", "torch", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
    scoring_config = {"scorer": scorer, "exact_rule": EXACT_RULE,
                      "implementation_sha256": sha256_file(Path(__file__)),
                      "runtime_versions": versions,
                      "semantic_rule": "full_response_max_cosine_strictly_greater_v1",
                      "threshold": threshold if embedder else None,
                      "embedder": embedder_model if embedder else None,
                      "embedder_revision_requested": embedder_revision if embedder else None,
                      "embedder_revision_resolved": embedder.resolved_revision if embedder else None,
                      "embedder_device": embedder_device if embedder else None}
    outputs = []
    for variant in variants:
        predictions, prediction_hash = loaded[variant]
        print(f"Scoring {variant}: {len(predictions)} generated records", flush=True)
        rows = score_rows(questions, predictions, scorer, threshold, embedder)
        path = run_dir / variant
        if sha256_file(path / "predictions.jsonl") != prediction_hash:
            raise ValueError(f"{variant} predictions changed during scoring; wait for generation to finish")
        metrics_path = path / "metrics.csv"
        atomic_csv(metrics_path, rows, METRIC_FIELDS)
        summary = {"schema_version": 1, "variant": variant,
                   "created_at": datetime.now(timezone.utc).isoformat(), "scoring": scoring_config,
                   "provenance": {**provenance, "predictions_sha256": prediction_hash,
                                  "metrics_sha256": sha256_file(metrics_path)},
                   **summarize_rows(rows)}
        atomic_json(path / "summary.json", summary)
        outputs.append(metrics_path)
        print(f"Saved {metrics_path}", flush=True)
    return outputs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variants", nargs="+", help="Score selected variants (for example, baseline first)")
    parser.add_argument("--scorer", choices=("exact", "biolord", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.9, help="Strict cosine > threshold")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--embedder-device", default="cpu")
    parser.add_argument("--embedder-revision", help="Pin the BioLORD model revision/commit")
    parser.add_argument("--embedder-batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        score_experiment(args.experiment_dir, args.run_name, args.scorer, args.threshold, args.embedder,
                         args.embedder_device, args.embedder_revision, args.embedder_batch_size, args.variants)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
