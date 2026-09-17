"""Identity-preserving, undirected biomedical graph adapters.

Only graph edges supply connectivity. Label dictionaries never add edges.
Matching uses normalized preferred names; distinct source IDs stay distinct.
"""
import csv
from collections import Counter, defaultdict

import pandas as pd

try:
    from .utils import DATA_DIR, KG_CSV_PATH, KG_PARQUET_PATH, download, normalize
    from .sources import (HETIONET_EDGES, UNIPROT_URL, GENE_URL, biokg_archive,
                          drkg_triples, hetionet_nodes, source_record, tsv_rows, zip_triples)
except ImportError:
    from utils import DATA_DIR, KG_CSV_PATH, KG_PARQUET_PATH, download, normalize
    from sources import (HETIONET_EDGES, UNIPROT_URL, GENE_URL, biokg_archive,
                         drkg_triples, hetionet_nodes, source_record, tsv_rows, zip_triples)

BIO_TYPES = {"disease", "drug", "gene/protein", "effect/phenotype"}
HETIO_TYPES = {"Disease": "disease", "Compound": "drug", "Gene": "gene/protein",
               "Side Effect": "effect/phenotype", "Symptom": "effect/phenotype"}


class KnowledgeGraph:
    name = "base"

    def __init__(self):
        self.ids = {}
        self.external_ids = []
        self.names = []
        self.types = []
        self.name_index = defaultdict(set)
        self.adj = defaultdict(set)
        self.constrained_adj = {key: defaultdict(set) for key in ("treatment", "phenotype", "side_effect")}
        self.supported_constraints = {"treatment", "phenotype", "side_effect"}
        self.sources = []
        self.notes = []
        self.stats = Counter()
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def add_node(self, external_id, name, node_type):
        if node_type not in BIO_TYPES:
            return None
        external_id = str(external_id)
        name = normalize(name)
        if external_id in self.ids:
            idx = self.ids[external_id]
            if self.types[idx] != node_type or self.names[idx] != name:
                raise ValueError(f"Conflicting node metadata for {external_id}")
            return idx
        idx = len(self.names)
        self.ids[external_id] = idx
        self.external_ids.append(external_id)
        self.names.append(name)
        self.types.append(node_type)
        if name:
            self.name_index[(name, node_type)].add(idx)
        return idx

    def add_edge(self, source, target, semantic=True):
        self.stats["input_edges"] += 1
        if source not in self.ids or target not in self.ids:
            self.stats["edges_outside_projection"] += 1
            return
        a, b = self.ids[source], self.ids[target]
        self.adj[a].add(b)
        self.adj[b].add(a)
        constraint = self.constraint_for(self.types[a], self.types[b])
        if constraint and semantic:
            self.constrained_adj[constraint][a].add(b)
            self.constrained_adj[constraint][b].add(a)
        self.stats["retained_edge_rows"] += 1

    @staticmethod
    def constraint_for(source_type, target_type):
        pair = frozenset((source_type, target_type))
        return {frozenset(("drug", "disease")): "treatment",
                frozenset(("disease", "effect/phenotype")): "phenotype",
                frozenset(("drug", "effect/phenotype")): "side_effect"}.get(pair)

    def supports_path(self, type_path, relation_policy="semantic"):
        return relation_policy == "type-only" or all(
            not (constraint := self.constraint_for(a, b)) or constraint in self.supported_constraints
            for a, b in zip(type_path, type_path[1:]))

    @property
    def supported_types(self):
        return set(self.types)

    def resolve(self, name, node_type):
        return self.name_index.get((normalize(name), normalize(node_type)), set())

    def node_exists(self, node_name, node_type):
        return bool(self.resolve(node_name, node_type))

    def neighbor_ids(self, node_ids, target_type, relation_policy="semantic"):
        answer = set()
        for node in node_ids:
            constraint = self.constraint_for(self.types[node], target_type)
            adjacency = self.constrained_adj[constraint] if constraint and relation_policy == "semantic" else self.adj
            answer.update(n for n in adjacency.get(node, ()) if self.types[n] == target_type)
        return answer

    def query_ids(self, node_name, node_type, target_type, intermediate_type=None, relation_policy="semantic"):
        starts = self.resolve(node_name, node_type)
        if intermediate_type is not None:
            starts = self.neighbor_ids(starts, intermediate_type, relation_policy)
        return self.neighbor_ids(starts, target_type, relation_policy)

    def answer_names(self, node_ids):
        # Unknown labels remain explicit unmatched candidates, rather than
        # disappearing and artificially improving precision.
        return {self.names[i] or f"unmapped:{self.external_ids[i]}" for i in node_ids}

    def get_neighbors(self, node_name, node_type, target_type):
        return self.answer_names(self.query_ids(node_name, node_type, target_type))

    def get_2hop_neighbors(self, start_name, start_type, intermediate_type, target_type):
        return self.answer_names(self.query_ids(start_name, start_type, target_type, intermediate_type))

    def metadata(self):
        return {"name": self.name, "sources": self.sources, "notes": self.notes,
                "supported_constraints": sorted(self.supported_constraints),
                "node_count": len(self.ids), "node_types": dict(Counter(self.types)),
                "unlabelled_nodes": sum(not x for x in self.names),
                "ambiguous_name_type_pairs": sum(len(ids) > 1 for ids in self.name_index.values()),
                "adjacency_arcs": sum(len(v) for v in self.adj.values()), **dict(self.stats)}

    def finish(self):
        self._loaded = True
        print(f"{self.name}: {len(self.ids):,} nodes; "
              f"{self.stats['retained_edge_rows']:,} retained edge rows", flush=True)


class PrimeKG(KnowledgeGraph):
    name = "PrimeKG"

    def load(self):
        if self._loaded:
            return
        path = KG_PARQUET_PATH if KG_PARQUET_PATH.exists() else KG_CSV_PATH
        columns = ["x_index", "x_name", "x_type", "y_index", "y_name", "y_type", "relation"]
        frame = pd.read_parquet(path, columns=columns) if path.suffix == ".parquet" else pd.read_csv(path, usecols=columns)
        self.sources = [source_record(path)]
        for side in ("x", "y"):
            nodes = frame[[f"{side}_index", f"{side}_name", f"{side}_type"]].drop_duplicates()
            for idx, name, node_type in nodes.itertuples(index=False, name=None):
                self.add_node(str(idx), name, node_type)
        for a, b, relation in frame[["x_index", "y_index", "relation"]].itertuples(index=False, name=None):
            self.add_edge(str(a), str(b), semantic=relation not in {"contraindication", "disease_phenotype_negative"})
        self.notes = ["Local PrimeKG snapshot, restricted to BioHopR's four node types.",
                      "Benchmark is derived from PrimeKG; this is an in-source retrieval baseline.",
                      "Semantic mode excludes contraindications and negative phenotype annotations; indication and off-label use are treatment edges."]
        self.finish()


class Hetionet(KnowledgeGraph):
    name = "Hetionet"

    def load(self):
        if self._loaded:
            return
        nodes = hetionet_nodes()
        edges = download(HETIONET_EDGES, DATA_DIR / self.name / "hetionet-v1.0-edges.sif.gz")
        self.sources = [source_record(nodes), source_record(edges)]
        for row in tsv_rows(nodes):
            self.add_node(row["id"], row["name"], HETIO_TYPES.get(row["kind"]))
        for row in tsv_rows(edges):
            self.add_edge(row["source"], row["target"], semantic=row["metaedge"] != "CpD")
        self.notes = ["Official Hetionet v1.0, four-type projection.",
                      "Side Effect and Symptom share the benchmark phenotype type but retain distinct IDs.",
                      "All retained nodes count toward coverage, including nodes without retained edges.",
                      "Semantic treatment uses CtD (treats); CpD (palliates) is not treated as disease treatment."]
        self.finish()


class DRKG(KnowledgeGraph):
    name = "DRKG"

    def load(self):
        if self._loaded:
            return
        triples = drkg_triples()
        hetio = hetionet_nodes()
        bio = biokg_archive()
        genes = download(GENE_URL, DATA_DIR / "labels" / "Homo_sapiens.gene_info.gz")
        self.sources = [source_record(p) for p in (triples, hetio, bio, genes)]
        labels = {r["id"]: r["name"] for r in tsv_rows(hetio)}
        for row in tsv_rows(genes):
            labels["Gene::" + row["GeneID"]] = row["Symbol"]
        for member, prefix in [("biokg.metadata.drug.tsv", "Compound::"),
                               ("biokg.metadata.disease.tsv", "Disease::MESH:")]:
            for entity, predicate, label in zip_triples(bio, member):
                if predicate == "NAME":
                    labels.setdefault(prefix + entity, label.replace("_", " "))
        # First pass is unnecessary: nodes can be registered before each edge.
        with triples.open(encoding="utf-8", newline="") as stream:
            for source, relation, target in csv.reader(stream, delimiter="\t"):
                for entity in (source, target):
                    if entity not in self.ids:
                        kind = entity.split("::", 1)[0]
                        self.add_node(entity, labels.get(entity, ""), HETIO_TYPES.get(kind))
                is_drug_disease = {source.split("::", 1)[0], target.split("::", 1)[0]} == {"Compound", "Disease"}
                treatment = relation in {"DRUGBANK::treats::Compound:Disease", "Hetionet::CtD::Compound:Disease", "GNBR::T::Compound:Disease"}
                self.add_edge(source, target, semantic=not is_drug_disease or treatment)
        self.notes = [
            "Official DRKG triples; undirected projection of the four benchmark types.",
            "DRKG includes Hetionet as a source, so the graphs are not independent.",
            "Labels joined by exact source IDs from Hetionet, NCBI human GeneID, and BioKG DrugBank/MeSH metadata.",
            "Unmapped IDs remain in traversal and count as unmatched answer candidates.",
            "No PrimeKG topology or inferred entity mappings are used.",
            "Semantic treatment uses DrugBank treats, Hetionet CtD, and GNBR T; GNBR is literature-mined evidence, not clinical validation.",
        ]
        self.finish()


class BioKG(KnowledgeGraph):
    name = "BioKG"

    def load(self):
        if self._loaded:
            return
        bio = biokg_archive()
        hetio = hetionet_nodes()
        uniprot = download(UNIPROT_URL, DATA_DIR / "labels" / "uniprot-reviewed-human-gene-names.tsv")
        self.sources = [source_record(p) for p in (bio, hetio, uniprot)]
        self.supported_constraints.discard("treatment")
        self.supported_constraints.discard("phenotype")
        symbols = {r["Entry"]: r["Gene Names (primary)"] for r in tsv_rows(uniprot)}
        human = set()
        protein_names = {}
        for entity, predicate, value in zip_triples(bio, "biokg.metadata.protein.tsv"):
            if predicate == "SPECIES" and value == "HUMAN":
                human.add(entity)
            elif predicate == "NAME":
                protein_names[entity] = value
        for entity in sorted(human):
            self.add_node("protein:" + entity, symbols.get(entity) or protein_names.get(entity, ""), "gene/protein")
        for member, prefix, node_type in [("biokg.metadata.drug.tsv", "drug:", "drug"),
                                          ("biokg.metadata.disease.tsv", "disease:", "disease")]:
            for entity, predicate, label in zip_triples(bio, member):
                if predicate == "NAME":
                    self.add_node(prefix + entity, label.replace("_", " "), node_type)
        phenotype_labels = {}
        for row in tsv_rows(hetio):
            if row["kind"] == "Side Effect":
                phenotype_labels[row["id"].split("::", 1)[1]] = row["name"]
        relation_types = {
            "PPI": ("protein:", "protein:"), "DPI": ("drug:", "protein:"),
            "DRUG_TARGET": ("drug:", "protein:"), "DRUG_CARRIER": ("drug:", "protein:"),
            "DRUG_ENZYME": ("drug:", "protein:"), "DRUG_TRANSPORTER": ("drug:", "protein:"),
            "DDI": ("drug:", "drug:"), "DRUG_DISEASE_ASSOCIATION": ("drug:", "disease:"),
            "PROTEIN_DISEASE_ASSOCIATION": ("protein:", "disease:"),
            "RELATED_GENETIC_DISORDER": ("protein:", "disease:"),
            "DISEASE_GENETIC_DISORDER": ("disease:", "disease:"),
        }
        for source, relation, target in zip_triples(bio, "biokg.links.tsv"):
            if relation not in relation_types:
                self.stats["edge_rows_other_relations"] += 1
                continue
            a_prefix, b_prefix = relation_types[relation]
            for entity, prefix in ((source, a_prefix), (target, b_prefix)):
                if prefix == "disease:" and prefix + entity not in self.ids:
                    self.add_node(prefix + entity, "", "disease")
            self.add_edge(a_prefix + source, b_prefix + target, semantic=relation != "DRUG_DISEASE_ASSOCIATION")
        for source, relation, target in zip_triples(bio, "biokg.properties.drug.tsv"):
            if relation == "DRUG_SIDEEFFECT_ASSOCIATION":
                self.add_node("phenotype:" + target, phenotype_labels.get(target, ""), "effect/phenotype")
                self.add_edge("drug:" + source, "phenotype:" + target)
        self.notes = [
            "BioKG v1.0.0 main links plus explicit drug-side-effect property edges.",
            "Protein projection includes only nodes tagged HUMAN in the release metadata.",
            "Human protein labels use exact UniProt accession to primary gene-symbol mappings; native NAME is fallback.",
            "Drug/MeSH labels are native metadata; phenotype CUI labels are exact-ID joins to Hetionet Side Effect nodes.",
            "Unlabelled disease/phenotype nodes remain as unmatched candidates; no guessed names or PrimeKG topology.",
            "Other property edges (including indication and ontology annotations) are outside this projection.",
            "Drug-disease associations have no treatment predicate in this projection: semantic treatment queries are explicitly unsupported.",
            "No disease-phenotype relation is supplied by this projection.",
        ]
        self.finish()


KG_REGISTRY = {"primekg": PrimeKG, "hetionet": Hetionet, "drkg": DRKG, "biokg": BioKG}


def get_kg(kg_name):
    key = kg_name.casefold()
    if key == "biographfusion":
        raise ValueError("BioGraphFusion's legacy mapping was inferred from PrimeKG gene overlap. "
                         "It has no reproducible authoritative identity mapping and is excluded from validated runs. "
                         "Existing files are preserved. Use drkg or biokg.")
    if key not in KG_REGISTRY:
        raise ValueError(f"Unknown KG {kg_name!r}; choose from {', '.join(KG_REGISTRY)}")
    kg = KG_REGISTRY[key]()
    kg.load()
    return kg

