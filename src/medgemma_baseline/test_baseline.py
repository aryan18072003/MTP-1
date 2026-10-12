"""Offline integration checks for the separate model-only entry point."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.medgemma_baseline.evaluate import evaluate, parse_args
from src.medgemma_baseline.prepare import experiment_path, prepare_baseline
from src.medgemma_eval.model import FittedPrompt, build_messages
from src.medgemma_eval.run import run_experiment


def benchmark():
    frame = pd.DataFrame([{
        "hop1": "hidden bridge", "hop1_type": "gene/protein",
        "hop2": "visible drug", "hop2_type": "drug", "target_type": "disease",
        "hop1_question": "Which disease is associated with this gene?",
        "hop2_question": "Which disease is linked through a gene to this drug?",
        "relation_hop1": "gene/protein:disease", "relation_hop2": "drug:gene/protein:disease",
        "answer": ["candidate"],
    }])
    frame.attrs["provenance"] = {"dataset": "offline-fixture", "revision": "pinned"}
    return frame


class FakeBackend:
    metadata = {"test_double": True}

    def __init__(self, config):
        if config["variants"] != ["baseline"]:
            raise AssertionError("KG variant entered a model-only experiment")

    def fit(self, question, evidence):
        if set(question) != {"question", "target_type"} or evidence:
            raise AssertionError("Reference answers, hidden metadata, or KG evidence reached the model")
        text = json.dumps(build_messages(question["question"], question["target_type"], evidence))
        if "hidden bridge" in text or "candidate" in text:
            raise AssertionError("Reference/bridge leakage")
        return FittedPrompt(text, 40, 0, 0, hashlib.sha256(text.encode()).hexdigest())

    def generate(self, prompts):
        return [{"prediction": "candidate", "output_tokens": 1, "output_truncated": False} for _ in prompts]


class BaselineTests(unittest.TestCase):
    def test_full_and_subset_have_separate_defaults(self):
        full = parse_args([])
        subset = parse_args(["--max-samples", "5"])
        self.assertEqual(full.experiment_dir, experiment_path().resolve())
        self.assertNotEqual(full.experiment_dir, subset.experiment_dir)
        self.assertEqual(full.quantization, "bf16")
        self.assertEqual(full.variants, ["baseline"])

    def test_prepare_two_questions_without_loading_graphs(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.medgemma_baseline.prepare.load_benchmark", return_value=benchmark()
        ), patch("src.medgemma_eval.prepare.get_kg", side_effect=AssertionError("KG loading forbidden")) as kg:
            manifest = prepare_baseline(Path(temp) / "experiment")
            kg.assert_not_called()
            self.assertEqual(manifest["n_questions"], 2)
            self.assertEqual(manifest["n_applicable_questions"], 2)
            self.assertEqual(manifest["kgs"], [])
            self.assertEqual(manifest["contexts"], {})
            self.assertFalse((Path(temp) / "experiment" / "contexts").exists())

    def test_preparation_reuse_preserves_files_and_rejects_changed_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.medgemma_baseline.prepare.load_benchmark", return_value=benchmark()
        ):
            directory = Path(temp) / "experiment"
            first = prepare_baseline(directory)
            original = (directory / "manifest.json").read_bytes()
            self.assertEqual(first, prepare_baseline(directory))
            with self.assertRaisesRegex(ValueError, "different data"):
                prepare_baseline(directory, max_samples=1)
            self.assertEqual(original, (directory / "manifest.json").read_bytes())

    def test_modified_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.medgemma_baseline.prepare.load_benchmark", return_value=benchmark()
        ):
            directory = Path(temp) / "experiment"
            prepare_baseline(directory)
            (directory / "questions.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified"):
                prepare_baseline(directory)

    def test_evaluate_generates_scores_and_resumes_without_reloading_model(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.medgemma_baseline.prepare.load_benchmark", return_value=benchmark()
        ):
            directory = Path(temp) / "experiment"
            args = parse_args(["--experiment-dir", str(directory), "--scorer", "exact"])
            def offline_run(options):
                run_experiment(options, backend_factory=FakeBackend,
                               revision_resolver=lambda *_: "immutable-test-revision")
            with patch("src.medgemma_baseline.evaluate.run_experiment", side_effect=offline_run):
                evaluate(args)
            result = directory / "runs" / "bf16" / "baseline"
            predictions = (result / "predictions.jsonl").read_bytes()
            summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["exact_accuracy"], 1.0)
            self.assertEqual(set(summary["by_hop"]), {"1", "2"})
            for row in predictions.decode().splitlines():
                record = json.loads(row)
                self.assertEqual(record["context_status"], "baseline")
                self.assertEqual(record["evidence_paths_used"], 0)
            with patch("src.medgemma_baseline.evaluate.run_experiment", side_effect=lambda options:
                       run_experiment(options, backend_factory=lambda *_: self.fail("Completed model reloaded"))):
                evaluate(args)
            self.assertEqual((result / "predictions.jsonl").read_bytes(), predictions)
            self.assertEqual({p.name for p in (directory / "runs" / "bf16").iterdir() if p.is_dir()}, {"baseline"})

    def test_preview_has_no_references_and_does_not_score_or_load_model(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.medgemma_baseline.prepare.load_benchmark", return_value=benchmark()
        ), patch("src.medgemma_baseline.evaluate.score_experiment") as score:
            directory = Path(temp) / "experiment"
            args = parse_args(["--experiment-dir", str(directory), "--preview", "2"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                evaluate(args)
            self.assertNotIn("candidate", output.getvalue())
            self.assertNotIn("hidden bridge", output.getvalue())
            score.assert_not_called()
            self.assertFalse((directory / "runs").exists())

    def test_invalid_counts_rejected_before_preparation(self):
        for flags in (["--max-samples", "0"], ["--batch-size", "0"], ["--run-name", "../bad"],
                      ["--threshold", "nan"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(flags)


if __name__ == "__main__":
    unittest.main()
