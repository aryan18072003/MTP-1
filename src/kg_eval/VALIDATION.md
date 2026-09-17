# Implementation review and completed evaluation

Reviewed and run on 2026-09-07. Scope: `src/kg_eval/`. The LLM pipeline was not
modified. The original implementation was **a useful type-only connectivity
baseline, but not fully correct for the semantics of BioHopR's questions**.

## What the original code got right

- One-hop starts from `hop1`; two-hop starts from `hop2`, traverses all matching
  intermediate types, then reaches the requested target type. Supplying the named
  `hop1` bridge to two-hop traversal would leak an intermediate answer; neither
  the original nor corrected evaluator does that.
- Hit, precision, recall and F1 were valid set-overlap formulas for the candidate
  sets generated and the reference answer lists supplied.
- The Hetionet type mapping (Compound/Drug, Gene/Protein,
  Side Effect or Symptom/Phenotype) is a reasonable benchmark projection.

## Problems corrected

1. **Predicate semantics were ignored.** Questions include "can treat" and "is
   treated by", not only generic "related to" wording. The old adjacency pooled
   indications, off-label use and contraindications. The local PrimeKG contains
   61,350 directed contraindication rows. It also distinguishes positive and
   negative disease–phenotype annotations. The corrected default excludes
   contraindications and negative phenotype edges, and uses explicit treatment
   mappings for Hetionet and DRKG. `relation_hop1` and `relation_hop2` encode
   *type paths*, not the original graph predicate names.
2. **Names were graph identities.** Distinct native IDs with the same lowercase
   name/type were collapsed. In two-hop traversal this could join edges through
   different entities. Native IDs now remain distinct throughout traversal;
   ambiguous starting names are counted explicitly. Hetionet's retained node
   table has 151 ambiguous normalized name/type pairs.
3. **Coverage depended on retained edges.** Hetionet only counted nodes after
   finding a retained edge. Nodes in the official node table now count as present
   even if no usable outgoing edge exists. Coverage and successful retrieval are
   reported separately.
4. **Missing data was scored as incorrect.** Missing values could become the
   string `nan`; absent hop questions were not excluded. Missing questions,
   required fields and answer sets now have explicit excluded statuses. Missing
   KG nodes and unsupported paths remain failures in the primary denominator.
   All 7,633 rows in the pinned release have both questions, so the missing-question
   fix does not change the denominator for this particular full run.
5. **Two-hop source labels were misleading.** Plots grouped by `hop1_type` and
   called it the source. They now show `hop2_type via hop1_type`.
6. **Runs were not independently reproducible.** Files could be overwritten and
   the first dataset split was selected implicitly. New results have separate
   timestamp/policy/scope folders, pinned BioHopR and Hetionet revisions, input
   hashes, code hashes, environment versions and recorded scoring assumptions.

This treatment mapping follows the distinction in the
[PrimeKG paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9893183/) and the
[DRKG authors' treatment-relation documentation](https://www.dgl.ai/news/2020/06/09/covid.html).
It evaluates positive relationships or their inverse wording; it does not treat
all graph predicates as semantically interchangeable.

## Benchmark limitations verified locally

The pinned [BioHopR release](https://huggingface.co/datasets/knowlab-research/BioHopR/tree/08f06692c3900347e7405ba55ff6a6d330f55ddb)
has 7,633 paired rows, 2,494 distinct one-hop question texts, and 7,633 distinct
two-hop question texts. **107 one-hop question texts occur with differing
reference answer sets.** For example, the Hydrocortisone side-effect question
has 42 distinct reference sets across its paired rows.

The evaluator keeps these released answers unchanged. Therefore precision and
recall measure agreement with the supplied reference set, which should not be
assumed to be exhaustive or consistent across repeated one-hop questions.
Primary metrics are row-weighted. The summary additionally reports unique
question/reference-set pairs, not a misleading count of distinct question texts.

Exact preferred-name scoring is sensitive to naming conventions and ontology
coverage. It is not the embedding-based scorer from the BioHopR paper or the
LLM pipeline. PrimeKG is the benchmark's source graph and has an inherent
in-source advantage. DRKG includes Hetionet and is not an independent graph.

## Completed results

Each graph evaluated all 7,633 samples, giving 15,266 per-question metric rows.
All figures below use **semantic mode** and the full applicable denominator.
Unsupported graph relations count as failures.

| Graph | 1-hop hit | 2-hop hit | 1-hop recall | 2-hop recall | 1-hop precision | 2-hop precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PrimeKG | 99.87% | 99.87% | 95.14% | 99.53% | 95.98% | 92.68% |
| Hetionet | 46.50% | 3.50% | 12.21% | 1.02% | 17.29% | 0.90% |
| DRKG | 52.64% | 21.09% | 21.86% | 7.53% | 9.54% | 1.24% |
| BioKG | 57.40% | 12.92% | 23.62% | 3.26% | 25.14% | 3.99% |

PrimeKG's hit rate remains high even after filtering the wrong relations, but
its precision improves from the legacy 88.98% to 95.98% for one hop and from
80.74% to 92.68% for two hops. Hit rate alone would conceal that semantic error.

The BioKG projection supports 5,615 one-hop and 3,928 two-hop relation paths.
Its generic drug–disease associations are not asserted to be treatment, and its
disease–phenotype path is unsupported. These limitations are explicit in the
CSV statuses and supported-only metrics, rather than hidden by dropping rows.

- [Machine-readable comparison](results/BioHopR/comparison/20260907T135016235269Z_semantic_full/graph_comparison.csv)
- [PrimeKG run](results/BioHopR/PrimeKG/20260907T134920511421Z_semantic_full/summary.json)
- [Hetionet run](results/BioHopR/Hetionet/20260907T134932420651Z_semantic_full/summary.json)
- [DRKG run](results/BioHopR/DRKG/20260907T134956829501Z_semantic_full/summary.json)
- [BioKG run](results/BioHopR/BioKG/20260907T135011886173Z_semantic_full/summary.json)

Each run stores three necessary tabular/metadata files plus one requested
dashboard in `plots/`. Candidate dumps are optional and were not saved for these
full runs. Pre-existing legacy outputs and the BioGraphFusion mapping are
preserved. Intermediate audit outputs created during this implementation were
moved outside the results tree to `archive/type-only-development/` and labelled
as development artifacts, so they are not mixed into the final comparison.

## Validation performed

- 17 offline regression tests passed: graph identity, inverse traversal, type and
  relation constraints, contraindications, negative phenotype annotations,
  missing values, unsupported relations, scoring, all four adapter formats,
  native identifier label joins, source integrity and isolated result paths.
- Full actual-data runs completed for all four graphs.
- All 61,064 stored question rows were checked for sample/hop uniqueness,
  denominators, hit/precision/recall arithmetic and complete sample coverage.
- Every saved output hash and recorded code hash matched the current files.
- Comparison links were resolved and the generated dashboard was visually
  inspected for correct source/intermediate labels.
- The direct-file CLI, separate prefix-run output, and optional compressed
  candidate evidence were checked on five real Hetionet benchmark samples in a
  temporary directory. Saved overlaps matched the recorded candidate names.

See [README.md](README.md) for commands, source links, graph projections and the
folder layout. The legacy BioGraphFusion mapping is not a validated identity
mapping and that graph is excluded from the new comparison.
