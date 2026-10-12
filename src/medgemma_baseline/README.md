# MedGemma 27B model-only evaluation on BioHopR

This folder runs `google/medgemma-27b-text-it` on the BioHopR question text,
without graph evidence or KG downloads. It shares the tested Hugging Face
backend, prompt, checkpointing, and scorers in `src/medgemma_eval` so its baseline
protocol stays consistent with the existing KG-assisted experiments.

## Protocol

- Use the project's pinned BioHopR revision
  `08f06692c3900347e7405ba55ff6a6d330f55ddb`, split `train` (the release's split
  name; no model training or fine-tuning happens).
- Each of 7,633 rows yields one 1-hop question and one 2-hop question:
  **15,266 evaluations**. Preserve repeated questions and each row's references.
- The model receives only the original question and requested answer type.
  Structured entity fields, hidden bridge entities, and reference answers are
  excluded from the model API. References remain in the prepared data for scoring.
- Ask for one entity name, or `UNKNOWN`. Greedy generation, seed 42, BF16 weights,
  batch size 4, maximum 4,096 input tokens and 64 generated tokens by default.
  Questions are never silently truncated. These are project evaluation settings,
  not a claim to reproduce another publication's MedGemma results.
- Exact accuracy: the complete generated answer must match any reference after
  case/whitespace normalization. No substring matching, punctuation stripping,
  synonym expansion, or ICD/ontology crosswalk is applied.
- Optional BioLORD accuracy: maximum cosine similarity between the entire answer
  and a reference must be **strictly greater than 0.9**. This is approximate name
  similarity, not verified biomedical concept equivalence.
- Report overall and separate 1-hop/2-hop accuracy. Excluded rows and statuses
  remain visible. These single-answer accuracies differ from the KG-only
  all-candidate precision/recall metrics.

## Remote server installation

Run from the repository root after copying this updated repository to the server.
Use an existing **Python 3.10 or 3.11** environment, or create one:

```bash
conda create -n medgemma_biohopr python=3.11 -y
conda activate medgemma_biohopr
```

Install the pinned inference dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r src/medgemma_baseline/requirements.txt
python -m pip check
```

This uses the official CUDA 12.8 PyTorch wheel. The CUDA version shown by
`nvidia-smi` describes driver capability; a separate CUDA toolkit installation is
not required for this wheel. The default run uses one H100 NVL GPU in BF16;
it does not require splitting the model across both GPUs.

Accept access conditions for
[MedGemma 27B](https://huggingface.co/google/medgemma-27b-text-it) using your Hugging
Face account, then authenticate on this server:

```bash
hf auth login
hf download google/medgemma-27b-text-it config.json
```

The second command checks model access. Model weights download automatically at
the first inference run. Existing authentication can be reused.

## Run

Optional smoke test, five dataset rows / ten questions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_baseline.evaluate --max-samples 5
```

Full evaluation, preparation → generation → exact and BioLORD scoring:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_baseline.evaluate
```

Use GPU 1 by changing `CUDA_VISIBLE_DEVICES=0` to `CUDA_VISIBLE_DEVICES=1`.
The selected physical GPU is still addressed as `cuda:0` inside the process.
BioLORD scoring defaults to CPU; use `--embedder-device cuda:0` if desired.

Exact scoring only (skips the BioLORD model download):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_baseline.evaluate --scorer exact
```

Prepare questions only, with no model access or GPU required:

```bash
python -m src.medgemma_baseline.prepare
```

Inspect two prompts without loading the model:

```bash
python -m src.medgemma_baseline.evaluate --max-samples 5 --preview 2
```

Generation checkpoints every completed batch. Rerun the **same command** after
interruption to resume. Data, code, resolved model revision, and generation
settings are checked before reuse. Changed generation settings need a new run
name; changed preparation needs a new experiment directory. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_baseline.evaluate --batch-size 1 --run-name bf16_batch1
```

To repeat only scoring on completed predictions:

```bash
python -m src.medgemma_eval.scoring \
  --experiment-dir src/medgemma_baseline/results/BioHopR/full \
  --run-name bf16 --variants baseline --scorer both --embedder-device cpu
```

## Output folders

```text
src/medgemma_baseline/
  prepare.py
  evaluate.py
  requirements.txt
  test_baseline.py
  results/BioHopR/
    first_5_rows/                 # Smoke experiment
    full/
      manifest.json              # Dataset identity, selection, code and file hashes
      questions.jsonl            # Both hops and their evaluation references
      runs/bf16/
        run.json                 # Model commit, settings, hardware, completion
        baseline/
          predictions.jsonl      # Generated answers, token counts, truncation flags
          metrics.csv            # Per-question exact/cosine scores
          summary.json           # Overall and per-hop accuracy, scoring provenance
```

The generated results directory is ignored by Git. No fabricated predictions or
accuracy values are shipped. Review `output_truncated` in predictions if answers
reach the generation limit; increasing that limit requires a new run name.

Offline checks (use test doubles, not actual MedGemma predictions):

```bash
python -B -m unittest src.medgemma_baseline.test_baseline src.medgemma_eval.test_generation src.medgemma_eval.test_scoring
```

References: [official model usage](https://huggingface.co/google/medgemma-27b-text-it),
[official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/#v280).
