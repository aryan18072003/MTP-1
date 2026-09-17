"""Offline tests: prompt boundaries, batch decoding, and resumable experiment identity."""
import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.kg_eval.utils import sha256_file, write_json
from src.medgemma_eval.model import FittedPrompt, HuggingFaceBackend, build_messages, fit_prompt
from src.medgemma_eval.run import generate_variant, make_parser, read_checkpoint, read_jsonl, run_experiment


class WordTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "BEGIN " + " ".join(message["content"] for message in messages) + " ASSISTANT"

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class FakeBackend:
    metadata = {"gpu": "test-double"}

    def __init__(self, config=None):
        self.calls = []

    def fit(self, question, evidence):
        # Only the approved public fields must cross the model boundary.
        if set(question) != {"question", "target_type"}:
            raise AssertionError("Reference or structured hidden fields reached the model")
        text = json.dumps(build_messages(question["question"], question["target_type"], evidence))
        return FittedPrompt(text, 50, len(evidence), len(evidence), hashlib.sha256(text.encode()).hexdigest())

    def generate(self, prompts):
        self.calls.extend(prompt.text for prompt in prompts)
        return [{"prediction": "candidate", "output_tokens": 1, "output_truncated": False} for _ in prompts]


def questions():
    return [{"question_id": f"0:{hop}", "sample_idx": 0, "hop": hop,
             "question": f"Name a disease using {hop} hops", "target_type": "disease",
             "source_name": "visible", "source_type": "drug", "intermediate_type": "gene/protein",
             "targets": ["secret reference"], "hidden_bridge": "secret bridge",
             "applicable": True, "exclusion_reason": ""} for hop in (1, 2)]


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    qs = questions()
    question_path = directory / "questions.jsonl"
    question_path.write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8")
    (directory / "contexts").mkdir()
    context_path = directory / "contexts/primekg.jsonl"
    context_path.write_text("".join(json.dumps({"question_id": q["question_id"], "status": "ok",
                                               "evidence": ["real connected path"],
                                               "evidence_truncated": False}) + "\n" for q in qs), encoding="utf-8")
    manifest = {"schema_version": "1.0", "n_samples": 1, "n_questions": 2, "kgs": ["primekg"],
                "question_file": {"file": "questions.jsonl", "sha256": sha256_file(question_path)},
                "contexts": {"primekg": {"file": "contexts/primekg.jsonl", "sha256": sha256_file(context_path)}}}
    write_json(directory / "manifest.json", manifest)
    return qs


class PromptTests(unittest.TestCase):
    def test_budget_drops_complete_paths_and_keeps_question(self):
        tokenizer = WordTokenizer()
        base = fit_prompt(tokenizer, "Specific original question?", "disease", [], 1000)
        paths = ["alpha -> beta", "gamma -> delta"]
        with_one = fit_prompt(tokenizer, "Specific original question?", "disease", paths[:1], 1000)
        fitted = fit_prompt(tokenizer, "Specific original question?", "disease", paths, with_one.input_tokens)
        self.assertEqual(fitted.evidence_paths_used, 1)
        self.assertEqual(fitted.evidence_paths_available, 2)
        self.assertIn("Specific original question?", fitted.text)
        self.assertIn(paths[0], fitted.text)
        self.assertNotIn("gamma", fitted.text)
        with self.assertRaisesRegex(ValueError, "instead of truncating"):
            fit_prompt(tokenizer, "Specific original question?", "disease", [], base.input_tokens - 1)

    def test_empty_evidence_matches_baseline(self):
        self.assertEqual(build_messages("Question", "disease"), build_messages("Question", "disease", []))

    def test_generation_decodes_suffix_and_retains_eos_defaults(self):
        class InputIDs:
            shape = (2, 7)

        class Encoding(dict):
            def to(self, device):
                return self

        class Output:
            def __getitem__(self, index):
                self.index = index
                return self

            def tolist(self):
                return [[101, 9, 0], [102, 103, 104]]

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, values, **kwargs):
                self.options = kwargs
                return Encoding(input_ids=InputIDs())

            def decode(self, values, **kwargs):
                return str(values)

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[8, 9])

            def generate(self, **kwargs):
                self.options = kwargs
                self.output = Output()
                return self.output

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        backend.device = "test"
        backend.tokenizer = Tokenizer()
        backend.model = Model()
        backend.max_new_tokens = 3
        result = backend.generate([SimpleNamespace(text="a"), SimpleNamespace(text="b")])
        self.assertEqual(backend.model.output.index[1].start, 7)
        self.assertFalse(backend.tokenizer.options["add_special_tokens"])
        self.assertNotIn("eos_token_id", backend.model.options)
        self.assertFalse(backend.model.options["do_sample"])
        self.assertEqual(result[0]["output_tokens"], 2)
        self.assertFalse(result[0]["output_truncated"])
        self.assertTrue(result[1]["output_truncated"])


class CheckpointTests(unittest.TestCase):
    def test_resume_does_not_repeat_answers_or_expose_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline/predictions.jsonl"
            qs = questions()
            contexts = {q["question_id"]: {"status": "no_candidates", "evidence": []} for q in qs}
            first = FakeBackend()
            generate_variant(qs[:1], contexts, path, first, 1)
            second = FakeBackend()
            generate_variant(qs, contexts, path, second, 2)
            self.assertEqual(len(first.calls), 1)
            self.assertEqual(len(second.calls), 1)
            self.assertNotIn("secret", first.calls[0] + second.calls[0])
            self.assertEqual(len(read_jsonl(path)), 2)
            self.assertEqual(read_jsonl(path)[1]["context_status"], "no_candidates")

    def test_failed_batch_is_retried_without_losing_completed_rows(self):
        class FailSecond(FakeBackend):
            def generate(self, prompts):
                if self.calls:
                    raise RuntimeError("simulated GPU failure")
                return super().generate(prompts)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            qs = questions()
            contexts = {q["question_id"]: {"status": "baseline", "evidence": []} for q in qs}
            with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                generate_variant(qs, contexts, path, FailSecond(), 1)
            self.assertEqual(len(read_jsonl(path)), 1)
            backend = FakeBackend()
            generate_variant(qs, contexts, path, backend, 1)
            self.assertEqual(len(backend.calls), 1)

    def test_incomplete_tail_is_preserved_and_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            valid = json.dumps({"question_id": "0:1", "status": "ok"}) + "\n"
            path.write_bytes(valid.encode() + b'{"question_id":')
            self.assertEqual(len(read_checkpoint(path)), 1)
            self.assertEqual(path.read_text(), valid)
            self.assertEqual(len(list(Path(tmp).glob("*.bin"))), 1)
            path.write_text("invalid\n" + valid, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corrupt completed"):
                read_checkpoint(path)

    def test_excluded_question_is_not_sent_to_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            qs = questions()
            qs[1].update(applicable=False, exclusion_reason="empty_targets")
            contexts = {q["question_id"]: {"status": "baseline", "evidence": []} for q in qs}
            backend = FakeBackend()
            path = Path(tmp) / "predictions.jsonl"
            generate_variant(qs, contexts, path, backend, 2)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(read_jsonl(path)[1]["status"], "excluded")


class RunIdentityTests(unittest.TestCase):
    def test_baseline_then_graph_resume_and_reject_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            parser = make_parser()
            base = ["--experiment-dir", str(directory), "--run-name", "nf4"]
            revision_calls = []

            def resolver(model, revision):
                revision_calls.append((model, revision))
                return "immutable-test-commit"

            run_experiment(parser.parse_args(base + ["--variants", "baseline"]), FakeBackend, resolver)
            metadata = json.loads((directory / "runs/nf4/run.json").read_text())
            self.assertEqual(metadata["status"], "partial")
            run_experiment(parser.parse_args(base + ["--variants", "primekg"]), FakeBackend, resolver)
            metadata = json.loads((directory / "runs/nf4/run.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(len(revision_calls), 1)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run_experiment(parser.parse_args(base + ["--max-new-tokens", "2"]), FakeBackend, resolver)

    def test_modified_prepared_data_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            with (directory / "questions.jsonl").open("a") as stream:
                stream.write(" ")
            args = make_parser().parse_args(["--experiment-dir", str(directory)])
            with self.assertRaisesRegex(ValueError, "data was modified"):
                run_experiment(args, lambda _: self.fail("Model must not load"))

    def test_complete_predictions_repair_interrupted_metadata_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            args = make_parser().parse_args(["--experiment-dir", str(directory)])
            run_experiment(args, FakeBackend, lambda *_: "test-commit")
            run_path = directory / "runs/nf4/run.json"
            metadata = json.loads(run_path.read_text())
            metadata.update(status="interrupted", completed_variants=[])
            write_json(run_path, metadata)
            run_experiment(args, lambda _: self.fail("Completed run must not load model"))
            repaired = json.loads(run_path.read_text())
            self.assertEqual(repaired["status"], "complete")
            self.assertEqual(repaired["completed_variants"], ["baseline", "primekg"])

    def test_complete_checkpoint_identity_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            args = make_parser().parse_args(["--experiment-dir", str(directory)])
            run_experiment(args, FakeBackend, lambda *_: "test-commit")
            path = directory / "runs/nf4/baseline/predictions.jsonl"
            rows = read_jsonl(path)
            rows[0]["sample_idx"] = 99
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match prepared question"):
                run_experiment(args, lambda _: self.fail("Invalid checkpoint must not load model"))


if __name__ == "__main__":
    unittest.main()
