"""Offline tests for leakage prevention, real graph paths, and preparation reuse."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.kg_eval.kg_loader import KnowledgeGraph
from .prepare import build_questions, prepare_experiment
from .retrieval import retrieve_context


def sample(**changes):
    row = {"hop1": "hidden bridge", "hop1_type": "gene/protein", "hop2": "Visible Drug",
           "hop2_type": "drug", "target_type": "disease", "answer": ["Correct disease"],
           "hop1_question": "Name a disease associated with hidden bridge.",
           "hop2_question": "Name a disease associated with a protein associated with Visible Drug.",
           "relation_hop1": "gene/protein:disease", "relation_hop2": "drug:gene/protein:disease"}
    row.update(changes)
    return row


def fixture_graph(reverse=False):
    graph = KnowledgeGraph()
    nodes = [("drug:1", "visible drug", "drug"), ("protein:1", "real bridge", "gene/protein"),
             ("protein:2", "real bridge", "gene/protein"), ("protein:3", "hidden bridge", "gene/protein"),
             ("disease:1", "correct disease", "disease"), ("disease:2", "other disease", "disease"),
             ("disease:3", "false shortcut", "disease")]
    for node in reversed(nodes) if reverse else nodes:
        graph.add_node(*node)
    edges = [("drug:1", "protein:1"), ("protein:1", "disease:1"),
             ("protein:1", "disease:2"), ("protein:2", "disease:3")]
    for edge in reversed(edges) if reverse else edges:
        graph.add_edge(*edge)
    return graph


class AnswerAccessForbidden(dict):
    """Fail if traversal tries to use gold answers, exclusions, or hidden fields."""
    forbidden = {"targets", "answer", "applicable", "exclusion_reason", "hop1"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Retrieval attempted to read {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"Retrieval attempted to read {key}")
        return super().get(key, default)


class RetrievalTests(unittest.TestCase):
    def test_two_hop_question_does_not_expose_hidden_bridge(self):
        question = build_questions(pd.DataFrame([sample()]))[1]
        self.assertEqual(question["question_id"], "0:2")
        self.assertEqual(question["source_name"], "visible drug")
        self.assertEqual(question["question"], sample()["hop2_question"])
        self.assertNotIn("hidden bridge", json.dumps(question))
        self.assertNotIn("hop1", question)

    def test_actual_two_hop_paths_no_cross_name_merge_or_hidden_bridge(self):
        question = AnswerAccessForbidden(build_questions(pd.DataFrame([sample()]))[1])
        result = retrieve_context(question, fixture_graph())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["n_candidates"], 2)
        self.assertEqual(len(result["evidence"]), 2)
        text = "\n".join(result["evidence"])
        self.assertIn('ID="protein:1"', text)
        self.assertNotIn('ID="protein:2"', text)
        self.assertNotIn("false shortcut", text)
        self.assertNotIn("hidden bridge", text)

    def test_targets_and_reference_availability_cannot_affect_context(self):
        question = build_questions(pd.DataFrame([sample()]))[1]
        original = retrieve_context(question, fixture_graph())
        for targets in (["false shortcut"], []):
            altered = dict(question, targets=targets, applicable=False, exclusion_reason="empty_targets")
            self.assertEqual(retrieve_context(altered, fixture_graph()), original)

    def test_deterministic_cap_independent_of_graph_insertion_order(self):
        question = build_questions(pd.DataFrame([sample()]))[1]
        result = retrieve_context(question, fixture_graph(), max_evidence_paths=1)
        self.assertEqual(result, retrieve_context(question, fixture_graph(reverse=True), max_evidence_paths=1))
        self.assertEqual(result["n_candidates"], 2)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertTrue(result["evidence_truncated"])

    def test_missing_node_and_no_candidates_distinguished(self):
        questions = build_questions(pd.DataFrame([sample()]))
        self.assertEqual(retrieve_context(questions[0], fixture_graph())["status"], "no_candidates")
        self.assertEqual(retrieve_context(dict(questions[1], source_name="absent"), fixture_graph())["status"],
                         "start_not_found")

    def test_unsupported_treatment_and_semantic_negative_edge_filter(self):
        question = dict(build_questions(pd.DataFrame([sample()]))[1], hop=1, intermediate_type="")
        graph = fixture_graph()
        graph.add_edge("drug:1", "disease:1", semantic=False)
        self.assertEqual(retrieve_context(question, graph)["status"], "no_candidates")
        graph.supported_constraints.discard("treatment")
        self.assertEqual(retrieve_context(question, graph)["status"], "unsupported_relation")
        result = retrieve_context(question, graph, relation_policy="type-only")
        self.assertEqual(result["n_candidates"], 1)
        self.assertEqual(result["status"], "ok")

    def test_inverse_treatment_filter_applies_on_second_hop(self):
        question = dict(build_questions(pd.DataFrame([sample()]))[1],
                        source_name="real bridge", source_type="gene/protein", intermediate_type="disease",
                        target_type="drug")
        graph = fixture_graph()
        graph.add_edge("drug:1", "disease:1", semantic=False)
        self.assertEqual(retrieve_context(question, graph)["status"], "no_candidates")
        graph.add_edge("drug:1", "disease:2", semantic=True)
        result = retrieve_context(question, graph)
        self.assertEqual(result["n_candidates"], 1)
        self.assertIn('ID="disease:2"', result["evidence"][0])

    def test_invalid_cap_and_missing_fields(self):
        question = build_questions(pd.DataFrame([sample()]))[1]
        with self.assertRaises(ValueError):
            retrieve_context(question, fixture_graph(), max_evidence_paths=0)
        self.assertEqual(retrieve_context(dict(question, intermediate_type=""), fixture_graph())["status"],
                         "missing_fields")
        self.assertEqual(retrieve_context(dict(question, question=""), fixture_graph())["status"],
                         "missing_question")

    def test_missing_questions_and_reference_answers_remain_in_question_file(self):
        questions = build_questions(pd.DataFrame([sample(answer=[], hop1_question=None)]))
        self.assertEqual(len(questions), 2)
        self.assertEqual([q["exclusion_reason"] for q in questions], ["missing_question", "empty_targets"])
        self.assertFalse(any(q["applicable"] for q in questions))


class PreparationTests(unittest.TestCase):
    def test_idempotence_settings_guard_and_generation_preservation(self):
        benchmark = pd.DataFrame([sample()])
        benchmark.attrs["provenance"] = {"dataset": "fixture", "sha256": "fixture-hash"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            with patch("src.medgemma_eval.prepare.load_benchmark", return_value=benchmark), \
                 patch("src.medgemma_eval.prepare.get_kg", side_effect=lambda name: fixture_graph()) as loader:
                manifest = prepare_experiment(output, kgs=["hetionet"], max_samples=1)
                self.assertEqual(manifest["n_questions"], 2)
                self.assertEqual(manifest["question_file"]["sha256"], manifest["questions_sha256"])
                self.assertEqual(loader.call_count, 1)
                generation = output / "generation-placeholder.jsonl"
                generation.write_text("preserved", encoding="utf-8")
                repeat = prepare_experiment(output, kgs=["hetionet"], max_samples=1)
                self.assertEqual(repeat, manifest)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(generation.read_text(encoding="utf-8"), "preserved")
                with self.assertRaisesRegex(ValueError, "different data, code, or retrieval settings"):
                    prepare_experiment(output, kgs=["hetionet"], max_samples=1, max_evidence_paths=2)
                self.assertEqual(generation.read_text(encoding="utf-8"), "preserved")
                with (output / "contexts" / "hetionet.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write("{}\n")
                with self.assertRaisesRegex(ValueError, "missing or modified"):
                    prepare_experiment(output, kgs=["hetionet"], max_samples=1)

    def test_does_not_overwrite_unrecognized_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with patch("src.medgemma_eval.prepare.load_benchmark", return_value=pd.DataFrame([sample()])):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    prepare_experiment(output, kgs=["hetionet"])
            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
