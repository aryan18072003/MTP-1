"""Offline regression tests: python -B -m unittest src.kg_eval.test_evaluation -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from .evaluate_kg import compute_metrics, evaluate_all, save_results, summarize
from .kg_loader import BioKG, DRKG, Hetionet, KnowledgeGraph, PrimeKG
from .utils import create_run_dir, download, parse_targets


def sample(**overrides):
    row = {"hop1": "bridge", "hop1_type": "gene/protein", "hop2": "start", "hop2_type": "drug",
           "target_type": "disease", "hop1_question": "Name a disease related to bridge",
           "hop2_question": "Name a disease related to a protein related to start",
           "relation_hop1": "gene/protein:disease", "relation_hop2": "drug:gene/protein:disease",
           "answer": ["answer"]}
    row.update(overrides)
    return row


def graph():
    kg = KnowledgeGraph()
    kg.add_node("s", "start", "drug")
    kg.add_node("b", "bridge", "gene/protein")
    kg.add_node("a", "answer", "disease")
    kg.add_edge("s", "b")
    kg.add_edge("b", "a")
    return kg


class EvaluationTests(unittest.TestCase):
    def test_metrics_normalization_and_deduplication(self):
        result = compute_metrics({"  ANSWER ", "wrong"}, ["answer", "ANSWER", "absent"])
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.5, 0.5, 0.5))
        self.assertEqual(result["n_ground_truth"], 2)
        self.assertEqual(compute_metrics(set(), ["answer"])["hit"], False)

    def test_target_formats_and_missing_values(self):
        self.assertEqual(parse_targets(np.array(["A", "a"])), ["a"])
        self.assertEqual(parse_targets('["a", "b"]'), ["a", "b"])
        self.assertEqual(parse_targets("['a', 'b']"), ["a", "b"])
        self.assertEqual(parse_targets("Disease, chronic"), ["disease, chronic"])
        self.assertEqual(parse_targets(float("nan")), [])
        self.assertEqual(parse_targets(None), [])
        with self.assertRaises(ValueError):
            parse_targets([123])

    def test_two_hop_does_not_merge_intermediates_with_same_name(self):
        kg = graph()
        kg.add_node("b2", "bridge", "gene/protein")
        kg.add_node("false", "false answer", "disease")
        kg.add_edge("b2", "false")
        self.assertEqual(kg.get_2hop_neighbors("start", "drug", "gene/protein", "disease"), {"answer"})
        self.assertEqual(len(kg.resolve("bridge", "gene/protein")), 2)

    def test_undirected_traversal_and_type_constraints(self):
        kg = graph()
        self.assertEqual(kg.get_neighbors("answer", "disease", "gene/protein"), {"bridge"})
        self.assertEqual(kg.get_2hop_neighbors("start", "drug", "disease", "disease"), set())

    def test_treatment_and_negative_phenotype_are_not_generic_associations(self):
        kg = graph()
        kg.add_node("contra", "contraindicated disease", "disease")
        kg.add_node("p", "absent phenotype", "effect/phenotype")
        kg.add_edge("s", "a", semantic=True)
        kg.add_edge("s", "contra", semantic=False)
        kg.add_edge("a", "p", semantic=False)
        self.assertEqual(kg.get_neighbors("start", "drug", "disease"), {"answer"})
        broad = kg.answer_names(kg.query_ids("start", "drug", "disease", relation_policy="type-only"))
        self.assertEqual(broad, {"answer", "contraindicated disease"})
        self.assertEqual(kg.get_neighbors("answer", "disease", "effect/phenotype"), set())
        # Inverse treatment wording (which drugs treat disease?) uses the same positive edge.
        self.assertEqual(kg.get_neighbors("answer", "disease", "drug"), {"start"})

    def test_unsupported_relation_is_not_claimed_as_treatment(self):
        kg = graph()
        kg.supported_constraints.discard("treatment")
        kg.add_edge("s", "a", semantic=False)
        row = sample(hop1="start", hop1_type="drug", hop1_question="Name a disease treated by drug start",
                     relation_hop1="drug:disease", relation_hop2="drug:drug:disease")
        result = evaluate_all(pd.DataFrame([row]), kg)
        self.assertEqual(result.iloc[0]["status"], "unsupported_relation")
        self.assertFalse(result.iloc[0]["hit"])
        self.assertEqual(summarize(result)["n_supported"], 0)

    def test_unlabelled_predictions_are_not_dropped(self):
        kg = graph()
        kg.add_node("unknown", "", "disease")
        kg.add_edge("b", "unknown")
        result = evaluate_all(pd.DataFrame([sample()]), kg)
        self.assertTrue(result["hit"].all())
        self.assertEqual(result.iloc[0]["precision"], 0.5)
        self.assertEqual(result.iloc[0]["n_unlabelled_predictions"], 1)

    def test_missing_questions_excluded_but_missing_starts_scored(self):
        rows = [sample(hop2_question=np.nan), sample(hop2="unknown")]
        result = evaluate_all(pd.DataFrame(rows), graph())
        hop2 = result[result["hop"] == 2]
        summary = summarize(hop2)
        self.assertEqual(summary["n_evaluated"], 1)
        self.assertEqual(summary["n_excluded"], 1)
        self.assertEqual(summary["hit_rate"], 0)
        self.assertEqual(list(hop2["status"]), ["missing_question", "start_not_found"])

    def test_empty_targets_and_fields_are_excluded(self):
        result = evaluate_all(pd.DataFrame([sample(answer=[]), sample(hop1=np.nan)]), graph())
        self.assertEqual(result.iloc[0]["status"], "empty_targets")
        self.assertEqual(result.iloc[2]["status"], "missing_fields")

    def test_unsupported_types_are_explicit_failures(self):
        row = sample(hop2_type="effect/phenotype", relation_hop2="effect/phenotype:gene/protein:disease")
        result = evaluate_all(pd.DataFrame([row]), graph())
        self.assertEqual(result.iloc[1]["status"], "unsupported_type")
        self.assertTrue(result.iloc[1]["applicable"])
        self.assertFalse(result.iloc[1]["hit"])

    def test_wrong_path_fails_instead_of_silently_evaluating(self):
        with self.assertRaises(ValueError):
            evaluate_all(pd.DataFrame([sample(relation_hop2="drug:disease:gene/protein")]), graph())

    def test_limit_and_separate_reproducible_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = create_run_dir("org/BioHopR", "TestKG", 1, tmp)
            b = create_run_dir("org/BioHopR", "TestKG", 1, tmp)
            self.assertNotEqual(a, b)
            frame = pd.DataFrame([sample(), sample()])
            kg = graph()
            result = evaluate_all(frame, kg, max_samples=1)
            summary = save_results(result, a, frame, kg, 1)
            self.assertEqual(summary["n_evaluated_samples"], 1)
            self.assertEqual(summary["scope"], "first-n")
            self.assertEqual(len(pd.read_csv(a / "per_question_metrics.csv")), 2)
            self.assertIn("per_question_metrics.csv", json.loads((a / "summary.json").read_text())["outputs"])
        with self.assertRaises(ValueError):
            evaluate_all(pd.DataFrame([sample()]), graph(), max_samples=0)

    def test_cached_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.tsv"
            path.write_text("source content")
            download("https://example.org/source.tsv", path)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                download("https://example.org/source.tsv", path)


class AdapterTests(unittest.TestCase):
    def test_primekg_filters_negative_relations(self):
        frame = pd.DataFrame([
            [0, "drug", "drug", 1, "treated", "disease", "indication"],
            [0, "drug", "drug", 2, "contra", "disease", "contraindication"],
            [1, "treated", "disease", 3, "absent", "effect/phenotype", "disease_phenotype_negative"],
        ], columns=["x_index", "x_name", "x_type", "y_index", "y_name", "y_type", "relation"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kg.parquet"
            path.touch()
            with patch("src.kg_eval.kg_loader.KG_PARQUET_PATH", path), \
                 patch("src.kg_eval.kg_loader.pd.read_parquet", return_value=frame):
                kg = PrimeKG()
                kg.load()
        self.assertEqual(kg.get_neighbors("drug", "drug", "disease"), {"treated"})
        self.assertEqual(kg.get_neighbors("treated", "disease", "effect/phenotype"), set())

    @patch("src.kg_eval.kg_loader.source_record", return_value={})
    @patch("src.kg_eval.kg_loader.download", return_value=Path("edges"))
    @patch("src.kg_eval.kg_loader.hetionet_nodes", return_value=Path("nodes"))
    def test_hetionet_preserves_isolates_and_native_ids(self, *_):
        nodes = [{"id": "Compound::D", "name": "drug", "kind": "Compound"},
                 {"id": "Side Effect::1", "name": "same", "kind": "Side Effect"},
                 {"id": "Symptom::2", "name": "same", "kind": "Symptom"},
                 {"id": "Disease::3", "name": "isolated", "kind": "Disease"}]
        edges = [{"source": "Compound::D", "target": "Side Effect::1", "metaedge": "CcSE"}]
        with patch("src.kg_eval.kg_loader.tsv_rows", side_effect=lambda p: iter(nodes if str(p) == "nodes" else edges)):
            kg = Hetionet()
            kg.load()
        self.assertTrue(kg.node_exists("isolated", "disease"))
        self.assertEqual(len(kg.resolve("same", "effect/phenotype")), 2)
        self.assertEqual(len(kg.query_ids("drug", "drug", "effect/phenotype")), 1)

    @patch("src.kg_eval.kg_loader.source_record", return_value={})
    @patch("src.kg_eval.kg_loader.download", return_value=Path("uniprot"))
    @patch("src.kg_eval.kg_loader.hetionet_nodes", return_value=Path("hetio"))
    @patch("src.kg_eval.kg_loader.biokg_archive", return_value=Path("bio"))
    def test_biokg_human_projection_and_authoritative_symbols(self, *_):
        members = {
            "biokg.metadata.protein.tsv": [("P1", "SPECIES", "HUMAN"), ("P1", "NAME", "NATIVE"),
                                           ("P2", "SPECIES", "MOUSE"), ("P2", "NAME", "MOUSE")],
            "biokg.metadata.drug.tsv": [("DB1", "NAME", "Drug_Name")],
            "biokg.metadata.disease.tsv": [("D1", "NAME", "disease")],
            "biokg.links.tsv": [("DB1", "DPI", "P1"), ("DB1", "DPI", "P2"),
                                ("P1", "PROTEIN_DISEASE_ASSOCIATION", "D1")],
            "biokg.properties.drug.tsv": [("DB1", "DRUG_SIDEEFFECT_ASSOCIATION", "C1")],
        }
        def rows(path):
            return iter([{"Entry": "P1", "Gene Names (primary)": "GENE1"}] if str(path) == "uniprot"
                        else [{"id": "Side Effect::C1", "name": "rash", "kind": "Side Effect"}])
        with patch("src.kg_eval.kg_loader.zip_triples", side_effect=lambda p, m: iter(members[m])), \
             patch("src.kg_eval.kg_loader.tsv_rows", side_effect=rows):
            kg = BioKG()
            kg.load()
        self.assertEqual(kg.get_2hop_neighbors("drug name", "drug", "gene/protein", "disease"), {"disease"})
        self.assertTrue(kg.node_exists("GENE1", "gene/protein"))
        self.assertNotIn("protein:P2", kg.ids)
        self.assertEqual(kg.get_neighbors("drug name", "drug", "effect/phenotype"), {"rash"})

    @patch("src.kg_eval.kg_loader.source_record", return_value={})
    @patch("src.kg_eval.kg_loader.download", return_value=Path("genes"))
    @patch("src.kg_eval.kg_loader.hetionet_nodes", return_value=Path("hetio"))
    @patch("src.kg_eval.kg_loader.biokg_archive", return_value=Path("bio"))
    def test_drkg_keeps_unmapped_connectivity(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drkg.tsv"
            path.write_text("Compound::DB1\tbinds\tGene::1\nGene::1\trelated\tDisease::MESH:D1\nGene::1\trelated\tDisease::UNKNOWN\n")
            def rows(p):
                return iter([{"GeneID": "1", "Symbol": "GENE1"}] if str(p) == "genes" else [])
            def triples(p, member):
                return iter([("DB1", "NAME", "drug")] if "drug.tsv" in member else [("D1", "NAME", "disease")])
            with patch("src.kg_eval.kg_loader.drkg_triples", return_value=path), \
                 patch("src.kg_eval.kg_loader.tsv_rows", side_effect=rows), \
                 patch("src.kg_eval.kg_loader.zip_triples", side_effect=triples):
                kg = DRKG()
                kg.load()
            self.assertEqual(kg.get_2hop_neighbors("drug", "drug", "gene/protein", "disease"),
                             {"disease", "unmapped:Disease::UNKNOWN"})


if __name__ == "__main__":
    unittest.main()
