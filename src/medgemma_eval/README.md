# BioHopR with MedGemma 27B

This package evaluates `google/medgemma-27b-text-it` in five separate settings:
MedGemma alone, and MedGemma with evidence from PrimeKG, Hetionet, DRKG or BioKG.
It performs inference without fine-tuning. See [RUN_REMOTE.md](RUN_REMOTE.md) for
the copy, installation and execution commands for the remote H100 server.

## Workflow

1. `prepare.py` loads the pinned BioHopR release and retrieves evidence through the
   existing `src/kg_eval` adapters. Each dataset row becomes one 1-hop and one
   2-hop question. It uses the visible source entity and requested node types;
   it never uses the reference answers or hidden intermediate entity name to
   find or rank evidence. Each graph has a separate evidence file.
2. `run.py` loads MedGemma once per invocation, processes the baseline first when
   selected, then generates a single answer per question for each selected graph.
   `model.py` applies the model's chat template. It keeps at most 24 prepared paths
   by default and removes whole paths when the input exceeds 4,096 tokens. The
   question and instructions remain intact. Empty evidence produces the same
   prompt as the baseline, so uncovered questions still receive an LLM answer.
3. `scoring.py` compares the full generated answer to the reference names. It
   reports exact name accuracy and, optionally, maximum BioLORD cosine similarity
   strictly greater than 0.9. The latter is an approximate semantic score.
4. `compare.py` reports 1-hop, 2-hop and overall accuracy and percentage-point
   changes from the baseline. It also groups scores by the requested node types
   and optionally writes a comparison plot.

Retrieval applies the adapters' semantic filters and preserves native node IDs.
Evidence says “associated with” because the common adapter adjacency does not
retain every original predicate. Unsupported relations are recorded as missing
evidence. No disease-code crosswalk is introduced: name matching does not equate
ICD, MeSH, MONDO, DOID or other ontology identifiers. This experiment also supplies
structured query types, so it does not evaluate free-text entity extraction.

The original answer list is retained for both questions in each dataset row.
Repeated question texts can have different reference sets in the release; this
workflow does not rewrite those labels. These single-answer LLM scores answer a
different question from graph-only retrieval hit rate over all reachable nodes.
PrimeKG is the benchmark's source graph and therefore has a coverage advantage;
see the existing [dataset validation notes](../kg_eval/VALIDATION.md).

## Outputs and reproducibility

Results are created under the directory passed to `--experiment-dir`, recommended
as `experiments/medgemma27b/biohopr_full/`. The separate `biohopr_smoke/` directory
holds small trial runs. Each experiment contains a preparation manifest, questions
and bounded context files; `runs/nf4/` contains a run manifest and these folders:

```text
baseline/    primekg/    hetionet/    drkg/    biokg/
  predictions.jsonl
  metrics.csv
  summary.json
comparison/
  comparison.csv
  by_type.csv
  comparison.json
  comparison.png       # only with --plots
```

Generation checkpoints completed answers after each batch. Repeating the command
resumes pending questions and recovers incomplete final writes, preserving their
bytes. A file lock prevents simultaneous writers to the same run. Model commit,
quantization, prompt and generation settings, package versions, implementation
hashes, and input hashes prevent incompatible predictions from being mixed.
These settings aid reproduction; identical floating-point results across different
GPU hardware or software stacks are not guaranteed.

The default comparison requires successful inference for every applicable
question in every declared variant. `--allow-incomplete` explicitly compares only
the shared successful subset and reports omissions; it can introduce selection
bias. After resumed generation, rerun scoring and comparison to refresh hashes.

## Local verification

```bash
python -B -m unittest discover -s src/medgemma_eval -t . -p 'test_*.py'
python -B -m unittest src.kg_eval.test_evaluation
```

Tests use synthetic predictions and embedding vectors. They cover answer leakage,
real native-ID paths, evidence budgets, strict scoring, comparison denominators,
data/configuration mismatches and interruption recovery. Actual MedGemma 27B
inference and BioLORD model inference must still be checked on the target server;
no fabricated model-performance results are included.

Validation on 2026-09-13: 38 new tests and 17 existing KG tests passed. A two-row
BioHopR integration check loaded all four real graphs and produced the five-way
CSV/PNG comparison using a test generation backend in a temporary directory.
A tiny, randomly initialized Gemma3 text model also exercised batched generation
and output decoding on CPU. The published core dependency pins were checked for
mutual compatibility against package metadata; the remote environment still needs
the documented installation and pretrained-model smoke test.
