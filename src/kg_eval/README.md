# BioHopR knowledge graph evaluation

This folder evaluates the released BioHopR questions against PrimeKG, Hetionet,
DRKG and BioKG. It retrieves graph neighbors; it does not run an LLM. The existing
folder is `src/kg_eval/` (there was no `kg-csv/` folder in the workspace).

See [VALIDATION.md](VALIDATION.md) for the review of the original implementation,
dataset limitations and the completed evaluation results.

## Run

From the repository root, using Python 3.10 or newer:

```powershell
python -m pip install -r src/kg_eval/requirements.txt
python -B -m unittest src.kg_eval.test_evaluation -v
python -B -m src.kg_eval.evaluate_kg --kg primekg hetionet drkg biokg --plots
```

Running `python src/kg_eval/evaluate_kg.py` directly also works.

```powershell
# Evaluate one graph; default output is three small tabular/metadata files.
python -B -m src.kg_eval.evaluate_kg --kg drkg

# A prefix smoke run gets its own folder and cannot overwrite a full run.
python -B -m src.kg_eval.evaluate_kg --kg hetionet --max-samples 50

# Explicitly reproduce the broad, relation-agnostic baseline for comparison.
python -B -m src.kg_eval.evaluate_kg --kg primekg --relation-policy type-only

# Candidate IDs and names are optional because two-hop answer lists can be large.
python -B -m src.kg_eval.evaluate_kg --kg biokg --save-predictions
```

`--kg all` selects the four validated adapters. `--results-dir` overrides the
results root. `--revision` changes the benchmark revision; BioHopR defaults to
commit `08f06692c3900347e7405ba55ff6a6d330f55ddb`. Other datasets must provide the
same schema, and a dataset with multiple splits requires `--split`.

## Outputs

```text
src/data/
  kg.csv                              # Existing PrimeKG source
  kg.parquet                          # Existing PrimeKG snapshot used when present
  benchmarks/BioHopR/                  # Pinned benchmark and download manifest
  knowledge_graphs/
    Hetionet/                         # Official node and edge tables
    DRKG/                             # Only triples; no pretrained embeddings
    BioKG/                            # Official v1.0.0 archive
    labels/                           # NCBI and UniProt label snapshots

src/kg_eval/results/BioHopR/
  PrimeKG/<UTC timestamp>_semantic_full/
  Hetionet/<UTC timestamp>_semantic_full/
  DRKG/<UTC timestamp>_semantic_full/
  BioKG/<UTC timestamp>_semantic_full/
    per_question_metrics.csv           # One row per sample and hop
    metrics_by_type_path.csv           # Aggregates for each typed path
    summary.json                       # Metrics, coverage, source/code hashes, settings
    plots/retrieval_dashboard.png       # Only with --plots
    predictions.jsonl.gz               # Only with --save-predictions
  comparison/<UTC timestamp>_semantic_full/
    graph_comparison.csv               # Created for multi-graph invocations
```

The three tabular/metadata files are the default necessary results. No raw graph
data is placed in the results folder. Download caches are ignored by Git; results
remain reviewable. Earlier CSVs/plots directly under PrimeKG, Hetionet and
BioGraphFusion are legacy outputs and are preserved. New runs never resume from
or overwrite them. A failed run has `failed.json` and must not be used in a
comparison. A successful run has `summary.json` with output hashes.

## Evaluation protocol

For one hop, the start is the named `hop1` entity. For two hops, the start is
`hop2`, followed by **any** intermediate node of `hop1_type`, then a node of
`target_type`. The provided `hop1` name is not supplied as an oracle bridge for
two-hop traversal. All intermediates remain distinct native node IDs.

Semantic mode (the default) maps the released question templates to these
positive relationships. Inverse traversal is allowed for inverse wording such
as "drugs that treat this disease" versus "diseases treated by this drug".

| Graph | Drug–disease treatment | Disease–phenotype | Drug–side effect |
| --- | --- | --- | --- |
| PrimeKG | `indication`, `off-label use`; excludes `contraindication` | Positive annotations; excludes `disease_phenotype_negative` | `drug_effect` |
| Hetionet | `CtD`; excludes `CpD` (palliation) | Disease–Symptom edges | Compound–Side Effect edges |
| DRKG | DrugBank `treats`, Hetionet `CtD`, GNBR `T` | Inherited Disease–Symptom edges | Inherited Compound–Side Effect edges |
| BioKG projection | Unsupported: generic associations are not asserted to be treatment | Unsupported in this projection | `DRUG_SIDEEFFECT_ASSOCIATION` property edges |

Gene/protein associations are pooled within the requested type pair. DRKG's GNBR
treatment evidence is literature-mined; it is not an assertion of clinical
approval. The optional `type-only` policy pools all relations, including negative
ones, and is **only a broad connectivity baseline**.

Entity matching uses casefolding and whitespace normalization of preferred names.
It does not infer equivalence from gene overlap or fuzzy similarity. If a start
name resolves to several native IDs, their candidate sets are united and the
ambiguity count is recorded. Intermediate nodes with the same name are never
merged. Answer candidates are deduplicated by normalized preferred name to match
BioHopR's text-answer representation. Unlabelled candidate IDs remain unmatched
candidates rather than disappearing from the precision denominator.

The reported metrics are exact-name reference-answer hit rate (at least one
overlap), recall, precision and F1. Rates in CSV/JSON are fractions, not percentages.
Primary aggregates weight each applicable benchmark row equally. The summary
also reports distinct question/reference-set pairs, whose hit rate avoids
weighting identical question–answer pairs repeatedly. This is not the BioHopR
paper's embedding-based scoring protocol and is not directly comparable to LLM
accuracy from `src/pipeline/`.

Missing questions, required fields or reference answers are excluded explicitly.
Missing start entities and unsupported type/relation paths are counted as failed
retrievals. Coverage, status counts and supported-only hit rates make those
different failure causes visible. Two-hop plots label both the actual source
type and intermediate type. The benchmark answers are never silently repaired.

## Sources and label mappings

- [BioHopR release](https://huggingface.co/datasets/knowlab-research/BioHopR) and
  [paper](https://aclanthology.org/2025.findings-acl.668/).
- [PrimeKG](https://github.com/mims-harvard/PrimeKG): local snapshot; the benchmark
  was derived from this graph, so this is an in-source reference baseline.
- [Hetionet official TSV format](https://github.com/hetio/hetionet/tree/master/hetnet/tsv):
  node/edge downloads pinned to commit `af668689a81952058df8b1321cc6c744b93d83e2`.
  Side Effect and Symptom map to the benchmark phenotype type without merging IDs.
- [DRKG](https://github.com/gnn4dr/DRKG) and its
  [treatment-relation documentation](https://www.dgl.ai/news/2020/06/09/covid.html):
  official triples. Labels join exact DRKG source IDs to Hetionet labels,
  NCBI human GeneIDs, and BioKG DrugBank/MeSH metadata. No edges from these label
  dictionaries are added to DRKG. DRKG already contains Hetionet as an upstream
  source, so the graphs are not independent.
- [BioKG v1.0.0](https://github.com/dsi-bdi/biokg/releases/tag/v1.0.0): native
  drug/disease names; only release-tagged HUMAN protein nodes. Human UniProt
  accessions are joined to current reviewed primary gene symbols, with native
  protein NAME as fallback. Side-effect CUI labels join exact Hetionet IDs.
  Other ontology/property edges are not part of this projection. This is a
  documented BioHopR projection, not a claim to evaluate every BioKG relation.
- Label services: [NCBI human gene information](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz)
  and [UniProt API](https://www.uniprot.org/help/programmatic_access).
  These label feeds are mutable: the downloaded files, retrieval times and SHA-256
  hashes are recorded. Retain the local cache to reproduce the identical labels.

OpenBioLink was considered, but its identifiers need additional ontology label
resolution. CKG requires a substantially larger database build. DRKG and BioKG
were selected because their public snapshots and identifier metadata can be
evaluated locally without a database service. See the respective primary
[OpenBioLink](https://github.com/OpenBioLink/OpenBioLink) and
[CKG](https://github.com/MannLabs/CKG) repositories.

The legacy BioGraphFusion adapter is excluded from the registry: its supplied
name mappings were inferred from PrimeKG gene overlap, and the mapping builder
and authoritative identity evidence are absent. Its mapping CSV and earlier
outputs remain untouched for inspection.
