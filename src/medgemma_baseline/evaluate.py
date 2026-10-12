"""Prepare, generate, and score a standalone MedGemma BioHopR baseline.

Run from the repository root with ``python -m src.medgemma_baseline.evaluate``.
Generation uses the shared resumable runner; no KG is loaded or queried.
"""
import argparse
import json
import math
import re
from pathlib import Path

from src.medgemma_eval.model import DEFAULT_MODEL
from src.medgemma_eval.run import run_experiment
from src.medgemma_eval.scoring import DEFAULT_EMBEDDER, score_experiment
from .prepare import experiment_path, prepare_baseline


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--max-samples", type=int, help="Dataset rows; omit for the full dataset")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--run-name", default="bf16")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main", help="Model revision; resolved and saved as an immutable commit")
    parser.add_argument("--quantization", choices=["bf16", "nf4"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gib", type=float)
    parser.add_argument("--preview", type=int, default=0, help="Prepare and print N prompts; no model loading or scoring")
    parser.add_argument("--scorer", choices=["exact", "biolord", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--embedder-device", default="cpu")
    parser.add_argument("--embedder-revision")
    parser.add_argument("--embedder-batch-size", type=int, default=64)
    parser.set_defaults(variants=["baseline"])
    return parser


def parse_args(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores and hyphens")
    if min(args.batch_size, args.max_input_tokens, args.max_new_tokens, args.embedder_batch_size) <= 0:
        parser.error("Batch sizes and token counts must be positive")
    if args.preview < 0:
        parser.error("--preview must be nonnegative")
    if not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1:
        parser.error("--threshold must be a finite cosine value between -1 and 1")
    if args.min_free_gib is not None and (not math.isfinite(args.min_free_gib) or args.min_free_gib < 0):
        parser.error("--min-free-gib must be finite and nonnegative")
    args.experiment_dir = (args.experiment_dir or experiment_path(args.max_samples)).resolve()
    return args


def evaluate(args):
    prepare_baseline(args.experiment_dir, args.max_samples, args.dataset_revision)
    run_experiment(args)
    if args.preview:
        return
    score_experiment(args.experiment_dir, args.run_name, scorer=args.scorer, threshold=args.threshold,
                     embedder_model=args.embedder, embedder_device=args.embedder_device,
                     embedder_revision=args.embedder_revision, embedder_batch_size=args.embedder_batch_size,
                     variants=["baseline"])
    output = args.experiment_dir / "runs" / args.run_name / "baseline"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for label, scores in [("Overall", summary["overall"]),
                          *[(f"{hop}-hop", value) for hop, value in summary["by_hop"].items()]]:
        metrics = [f"{name}={scores[name]:.2%}" for name in ("exact_accuracy", "cosine_accuracy")
                   if scores[name] is not None]
        print(f"{label}: {scores['n_applicable']:,} applicable questions; " + ", ".join(metrics), flush=True)
    print(f"Results: {output}", flush=True)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
