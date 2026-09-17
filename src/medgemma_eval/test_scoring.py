"""Offline answer-scoring and fair-comparison regression tests (no model downloads)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from .compare import compare_experiment
from .scoring import (EmbeddingScorer, atomic_json, cosine_correct, exact_correct,
                      score_experiment, score_rows, sha256_file)


def question(index=0, hop=1, applicable=True, **kwargs):
    row = {"question_id": f"{index}:{hop}", "sample_idx": index, "hop": hop,
           "question": "Name one disease.", "source_name": "source", "source_type": "drug",
           "intermediate_type": "gene/protein" if hop == 2 else "", "target_type": "disease",
           "targets": ["disease a", "disease, chronic"] if applicable else [],
           "applicable": applicable, "exclusion_reason": "" if applicable else "missing_answers"}
    row.update(kwargs)
    return row


def prediction(q, answer="disease a", status="ok", **kwargs):
    row = {"question_id": q["question_id"], "sample_idx": q["sample_idx"], "hop": q["hop"],
           "status": status, "prediction": answer, "context_status": "ok", "evidence_paths_used": 2,
           "input_tokens": 100, "output_tokens": 3}
    row.update(kwargs)
    return row


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(directory, variants=("baseline", "primekg")):
    questions = [question(), question(hop=2), question(index=1, applicable=False)]
    write_jsonl(directory / "questions.jsonl", questions)
    atomic_json(directory / "manifest.json", {"fixture": True})
    run = {"schema_version": 1, "config": {"model": "mock", "revision": "pinned",
            "quantization": "none", "max_input_tokens": 512, "max_new_tokens": 32,
            "seed": 1, "variants": list(variants)},
           "prepared_manifest_sha256": sha256_file(directory / "manifest.json"),
           "question_file_sha256": sha256_file(directory / "questions.jsonl")}
    run_dir = directory / "runs" / "test"
    atomic_json(run_dir / "run.json", run)
    for variant in variants:
        rows = [prediction(q, status="ok" if q["applicable"] else "excluded") for q in questions]
        write_jsonl(run_dir / variant / "predictions.jsonl", rows)
    return questions, run_dir, run


class ScoringTests(unittest.TestCase):
    def test_exact_is_whole_answer_and_preserves_disease_commas(self):
        targets = ["Disease, chronic", "Disease A"]
        self.assertTrue(exact_correct("  DISEASE,  chronic \n", targets))
        self.assertTrue(exact_correct("disease a", targets))
        self.assertTrue(exact_correct("STRAẞE", ["strasse"]))
        for answer in ("disease", "Disease A, Disease B", "The answer is Disease A", "Disease A.", '"Disease A"', ""):
            self.assertFalse(exact_correct(answer, targets), answer)

    def test_cosine_threshold_is_strict(self):
        self.assertFalse(cosine_correct(0.9, 0.9))
        self.assertTrue(cosine_correct(0.900001, 0.9))
        self.assertFalse(cosine_correct(float("nan"), 0.9))

    def test_semantic_uses_complete_prediction_and_maximum_target(self):
        class FakeEncoder:
            def __init__(self):
                self.encoded = []

            def encode(self, texts, **kwargs):
                vectors = {"disease a": [1, 0], "disease, chronic": [0, 1],
                           "An answer with explanation": [0, 1]}
                self.encoded.extend(texts)
                return [vectors[text] for text in texts]
        encoder = FakeEncoder()
        embedder = EmbeddingScorer(encoder=encoder)
        q = question()
        rows = score_rows([q], [prediction(q, "An answer with explanation")], "both", embedder=embedder)
        self.assertEqual(rows[0]["exact_correct"], 0)
        self.assertEqual(rows[0]["cosine_correct"], 1)
        self.assertAlmostEqual(rows[0]["max_similarity"], 1)
        self.assertIn("An answer with explanation", encoder.encoded)
        self.assertEqual(len(encoder.encoded), 3)

    def test_errors_missing_and_exclusions_are_distinct(self):
        questions = [question(), question(hop=2), question(index=1, applicable=False)]
        rows = score_rows(questions, [prediction(questions[0], status="error", error="oom")], "exact")
        self.assertEqual([row["status"] for row in rows], ["error", "missing", "excluded"])
        self.assertEqual([row["exact_correct"] for row in rows], [0, 0, None])

    def test_empty_prediction_never_semantically_passes_even_with_negative_threshold(self):
        class FakeScorer:
            def preload(self, texts):
                pass

            def max_similarity(self, text, targets):
                return 0
        q = question()
        rows = score_rows([q], [prediction(q, "")], "both", threshold=-0.5, embedder=FakeScorer())
        self.assertEqual(rows[0]["cosine_correct"], 0)

    def test_duplicate_unknown_and_incorrect_identity_rejected(self):
        q = question()
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            score_rows([q], [prediction(q), prediction(q)], "exact")
        with self.assertRaisesRegex(ValueError, "unknown question_id"):
            score_rows([q], [prediction(question(index=10))], "exact")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            score_rows([q], [prediction(q, hop=2)], "exact")
        with self.assertRaisesRegex(ValueError, "applicability mismatch"):
            score_rows([q], [prediction(q, status="excluded")], "exact")

    def test_missing_targets_cannot_be_an_applicable_question(self):
        with self.assertRaisesRegex(ValueError, "without reference answers"):
            score_rows([question(targets=[])], [], "exact")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.questions, self.run_dir, self.run = fixture(self.directory)

    def score(self, **kwargs):
        score_experiment(self.directory, "test", scorer="exact", **kwargs)

    def read_comparison(self, output):
        with (output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_complete_comparison_and_delta(self):
        rows = [prediction(q, "wrong" if q["hop"] == 2 else "disease a",
                           status="ok" if q["applicable"] else "excluded") for q in self.questions]
        write_jsonl(self.run_dir / "baseline" / "predictions.jsonl", rows)
        self.score()
        output = compare_experiment(self.directory, "test")
        table = self.read_comparison(output)
        row = next(row for row in table if row["variant"] == "primekg" and row["hop"] == "2")
        self.assertEqual(float(row["exact_accuracy"]), 1)
        self.assertEqual(float(row["exact_delta_baseline_pp"]), 100)
        self.assertEqual(row["cosine_accuracy"], "")
        overall = next(row for row in table if row["variant"] == "baseline" and row["hop"] == "overall")
        self.assertEqual(overall["n_excluded"], "1")
        self.assertEqual(overall["n_compared"], "2")
        self.assertTrue((output / "by_type.csv").is_file())
        self.assertFalse(list(self.run_dir.rglob("*.part")))

    def test_stale_predictions_rejected(self):
        self.score()
        with (self.run_dir / "primekg" / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Stale or modified"):
            compare_experiment(self.directory, "test")

    def test_different_question_file_or_manifest_rejected(self):
        self.score()
        write_jsonl(self.directory / "questions.jsonl", [question(index=5)])
        with self.assertRaisesRegex(ValueError, "different questions"):
            compare_experiment(self.directory, "test")

    def test_changed_generation_config_rejected_but_progress_updates_allowed(self):
        self.score()
        self.run["completed_at"] = "later"
        atomic_json(self.run_dir / "run.json", self.run)
        compare_experiment(self.directory, "test")
        self.run["config"]["max_new_tokens"] = 100
        atomic_json(self.run_dir / "run.json", self.run)
        with self.assertRaisesRegex(ValueError, "run_config_sha256"):
            compare_experiment(self.directory, "test")

    def test_mixed_scoring_configs_rejected(self):
        self.score()
        path = self.run_dir / "primekg" / "summary.json"
        summary = json.loads(path.read_text())
        summary["scoring"]["exact_rule"] = "different_rule"
        atomic_json(path, summary)
        with self.assertRaisesRegex(ValueError, "Scoring configurations differ"):
            compare_experiment(self.directory, "test")

    def test_modified_metrics_rejected(self):
        self.score()
        path = self.run_dir / "primekg" / "metrics.csv"
        path.write_text(path.read_text().replace("disease a", "different disease"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale or modified"):
            compare_experiment(self.directory, "test")

    def test_incomplete_requires_explicit_common_successful_intersection(self):
        rows = [prediction(q, status="error" if q["hop"] == 2 else "ok") for q in self.questions if q["applicable"]]
        write_jsonl(self.run_dir / "primekg" / "predictions.jsonl", rows)
        self.score()
        with self.assertRaisesRegex(ValueError, "Incomplete inference"):
            compare_experiment(self.directory, "test")
        output = compare_experiment(self.directory, "test", allow_incomplete=True)
        overall = [row for row in self.read_comparison(output) if row["hop"] == "overall"]
        self.assertEqual([row["n_compared"] for row in overall], ["1", "1"])
        self.assertEqual([row["n_omitted_incomplete"] for row in overall], ["1", "1"])
        self.assertEqual(overall[1]["n_errors"], "1")
        manifest = json.loads((output / "comparison.json").read_text())
        self.assertEqual(manifest["comparison_scope"], "shared_successful_intersection")

    def test_missing_question_scored_and_rejected_by_default(self):
        write_jsonl(self.run_dir / "primekg" / "predictions.jsonl", [prediction(self.questions[0])])
        self.score()
        with self.assertRaisesRegex(ValueError, "Incomplete inference"):
            compare_experiment(self.directory, "test")

    def test_can_score_baseline_before_other_variants_exist(self):
        (self.run_dir / "primekg" / "predictions.jsonl").unlink()
        self.score(variants=["baseline"])
        self.assertTrue((self.run_dir / "baseline" / "summary.json").is_file())
        with self.assertRaisesRegex(ValueError, "No scores for primekg"):
            compare_experiment(self.directory, "test")


if __name__ == "__main__":
    unittest.main()
