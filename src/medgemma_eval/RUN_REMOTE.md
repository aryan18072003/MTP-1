# Run BioHopR with MedGemma 27B on the remote GPU server

This experiment compares five systems on identical BioHopR questions:

| Variant | Information available to MedGemma |
| --- | --- |
| `baseline` | Question only |
| `primekg` | Question and paths retrieved from PrimeKG |
| `hetionet` | Question and paths retrieved from Hetionet |
| `drkg` | Question and paths retrieved from DRKG |
| `biokg` | Question and paths retrieved from BioKG |

The model is **`google/medgemma-27b-text-it`**, Google's instruction-tuned text model. The different `google/medgemma-27b-it` checkpoint is multimodal; use the text checkpoint for this workflow. Access requires accepting the conditions on the [official Hugging Face model page](https://huggingface.co/google/medgemma-27b-text-it).

Your screenshot shows an H100 NVL with 95,830 MiB total and 43,785 MiB already allocated to `VLLM::EngineCore`: approximately **50.8 GiB free** at that moment. The commands use **4-bit NF4, BF16 computation, batch size 1**. Approximately 27 billion BF16 parameters alone need 50.3 GiB, before generation memory; BF16 needs more available memory than the screenshot shows. Check current capacity with `nvidia-smi`. These commands do not manage the existing vLLM process.

Before loading, inference checks for at least **22 GiB free for NF4** or **65 GiB for BF16**. These are initial estimates, not guarantees against running out of memory; actual use depends on sequence length, batch size and other processes. `--min-free-gib` changes the check, not the available memory.

The workflow uses Hugging Face Transformers directly. Create a separate environment so its dependencies do not conflict with the existing vLLM environment. No MedGemma performance results are supplied: generation and model scoring must run on your server.

**1. Copy the current files to the server.**

Run every Python command below from the remote repository root, for example `/home/your-user/MTP-1`. Transfer the complete `src/medgemma_eval/` directory, the Python files in `src/kg_eval/`, and **one** local PrimeKG file: `src/data/kg.parquet` or `src/data/kg.csv`. Keep that directory structure. A Git clone/pull only receives changes that have actually been committed and pushed; new local files are not included automatically.

For example, after replacing `your-user@your-server` and the remote path, these commands can be run from the local repository root with OpenSSH installed:

```bash
ssh your-user@your-server "mkdir -p /home/your-user/MTP-1/src/kg_eval /home/your-user/MTP-1/src/data"
scp -r src/medgemma_eval your-user@your-server:/home/your-user/MTP-1/src/
scp src/kg_eval/*.py your-user@your-server:/home/your-user/MTP-1/src/kg_eval/
scp src/data/kg.parquet your-user@your-server:/home/your-user/MTP-1/src/data/
```

Use `kg.csv` in the last command if that is the available PrimeKG file. Hetionet, DRKG, BioKG, their label tables, and the public BioHopR release download automatically when needed. Graphs and label tables are cached in `src/data/knowledge_graphs/`; copying an existing cache is optional. MedGemma and BioLORD use the Hugging Face cache. Allow space for the original model download even when inference uses 4-bit weights, plus graph downloads, extracted files and experiment outputs. Graph preparation needs CPU RAM independently of GPU memory; it loads one graph at a time.

Interrupted HTTP downloads, including UniProt's `ChunkedEncodingError: Response ended prematurely`, retry up to five attempts with increasing delays. Each attempt restarts the incomplete file; only completed downloads enter the cache. If preparation still exits, repeat the same prepare command. Completed source downloads are reused, while evidence preparation is recomputed. To get this retry behavior on an older server copy, transfer the updated `src/kg_eval/utils.py` first. No package reinstall is needed for this change.

**2. Create the environment and authenticate.**

On the Linux server:

```bash
cd /home/your-user/MTP-1
python3.11 -m venv .venv-medgemma
source .venv-medgemma/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r src/medgemma_eval/requirements.txt
python -m pip check
hf auth login
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print('Free/total GiB:', [round(x / 2**30, 2) for x in torch.cuda.mem_get_info()])"
python -m unittest discover -s src/medgemma_eval -t . -p 'test_*.py'
```

Accept the model conditions in your browser first, then use a Hugging Face read token for that same account at the login prompt. Credentials come from the Hugging Face login cache or `HF_TOKEN`; keep tokens out of source files and shell command history. The [HF CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) describes authentication.

The `CUDA Version: 13.0` in `nvidia-smi` describes driver support; it does not require a CUDA 13 PyTorch wheel. PyTorch publishes the [2.8.0 CUDA 12.8 wheel](https://pytorch.org/get-started/previous-versions/), and NVIDIA documents [backward compatibility with newer drivers](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Bitsandbytes 0.48.2 provides CUDA 12.8 support for the H100 architecture; see its [installation requirements](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation). The pinned package combination follows these supported interfaces; it still needs the smoke run on your hardware.

**3. Run a five-sample smoke experiment.**

Preparation retrieves evidence from all four graphs without loading MedGemma. Even this small selection requires loading the graphs initially.

```bash
python -m src.medgemma_eval.prepare \
  --experiment-dir experiments/medgemma27b/biohopr_smoke --max-samples 5

python -m src.medgemma_eval.run \
  --experiment-dir experiments/medgemma27b/biohopr_smoke --preview 1

CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_eval.run \
  --experiment-dir experiments/medgemma27b/biohopr_smoke --run-name nf4 \
  --quantization nf4 --batch-size 1 --max-input-tokens 4096 --max-new-tokens 64

python -m src.medgemma_eval.scoring \
  --experiment-dir experiments/medgemma27b/biohopr_smoke --run-name nf4 \
  --scorer both --embedder-device cpu

python -m src.medgemma_eval.compare \
  --experiment-dir experiments/medgemma27b/biohopr_smoke --run-name nf4 --plots
```

`--preview 1` prints one prompt per variant without loading the model or writing results; it shows the evidence before token-budget fitting. Omitting `--variants` for generation runs all five systems and loads MedGemma once for that command. Inspect smoke predictions, completion status, `output_truncated` flags and comparison before the full run. If 64 output tokens regularly cut answers short, choose a larger `--max-new-tokens` budget in a new run and apply it to all five systems. CPU embedding keeps scoring from competing for GPU memory; `--embedder-device cuda` is available when the GPU has capacity after generation exits.

**4. Evaluate the full dataset, starting with MedGemma alone.**

Use a different experiment directory for the full selection. `--max-samples` belongs to preparation; omit it to use every row. Run long jobs inside your usual persistent server session, for example `tmux`.

```bash
python -m src.medgemma_eval.prepare \
  --experiment-dir experiments/medgemma27b/biohopr_full

CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_eval.run \
  --experiment-dir experiments/medgemma27b/biohopr_full --run-name nf4 \
  --variants baseline --quantization nf4 --batch-size 1 \
  --max-input-tokens 4096 --max-new-tokens 64

python -m src.medgemma_eval.scoring \
  --experiment-dir experiments/medgemma27b/biohopr_full --run-name nf4 \
  --variants baseline --scorer both --embedder-device cpu

CUDA_VISIBLE_DEVICES=0 python -m src.medgemma_eval.run \
  --experiment-dir experiments/medgemma27b/biohopr_full --run-name nf4 \
  --variants primekg hetionet drkg biokg --quantization nf4 --batch-size 1 \
  --max-input-tokens 4096 --max-new-tokens 64

python -m src.medgemma_eval.scoring \
  --experiment-dir experiments/medgemma27b/biohopr_full --run-name nf4 \
  --scorer both --embedder-device cpu

python -m src.medgemma_eval.compare \
  --experiment-dir experiments/medgemma27b/biohopr_full --run-name nf4 --plots
```

Alternatively, run the full generation command once without `--variants`; it processes baseline followed by the four graph variants in one model session. The released 7,633 rows contain two questions each: up to **76,330 question/variant generations** across five systems. Use observed smoke timings to plan server time; preparation and model loading make the small run's wall time a poor direct throughput estimate.

**5. Resume and locate results.**

After an interruption, repeat the same generation command. Completed records are reused. Changing model, generation settings, code, recorded package versions or prepared data requires a new run name or experiment directory; the resume checks prevent mixing incompatible outputs. Rerun scoring and comparison after generation finishes. Comparison requires completed and scored predictions for every variant declared in the experiment. To evaluate BF16 when enough GPU memory is available, use a separate `--run-name bf16 --quantization bf16` consistently, and score/compare that run separately.

```text
experiments/medgemma27b/
  biohopr_smoke/                      # five rows only
  biohopr_full/
    manifest.json                    # benchmark and retrieval provenance
    questions.jsonl                  # selected questions and scoring references
    contexts/
      primekg.jsonl
      hetionet.jsonl
      drkg.jsonl
      biokg.jsonl
    runs/nf4/
      run.json                       # model revision, settings and completion status
      baseline/                      # each variant contains:
        predictions.jsonl            # generated answers and generation details
        metrics.csv                  # one row per question
        summary.json                 # aggregate scores and scoring provenance
      primekg/
      hetionet/
      drkg/
      biokg/
      comparison/
        comparison.csv               # hop scores and change from baseline
        by_type.csv                  # scores grouped by the requested type path
        comparison.json              # comparison provenance and coverage
        comparison.png               # written when --plots is selected
```

The prepare command uses the benchmark's starting entity and requested node types to query each graph. For two-hop questions it discovers intermediate entities from graph connections: the row's hidden intermediate name and answer list are not supplied to the model or used to choose evidence. The default evidence budget is 24 paths, chosen deterministically without consulting reference answers; generation may further reduce evidence to meet the token budget. This is evaluation with structured benchmark queries, so it does not measure free-text entity extraction.

Each system generates **one entity name**. Exact accuracy compares the whole response to any reference name after case-folding and collapsing whitespace. BioLORD accuracy uses the maximum cosine similarity between the whole generated response and reference names, accepting **strictly greater than 0.9**. `--scorer both` reports both measures using [BioLORD-2023-C](https://huggingface.co/FremyCompany/BioLORD-2023-C); `--scorer exact` avoids the embedding-model download. These answer-generation scores differ from the existing KG evaluator's hit rate over all reachable candidates. Do not combine them into a single ranking.

Disease lookup still uses normalized preferred names, without shared ICD/MONDO/DOID/MeSH alignment. BioLORD similarity is an approximate additional score, not a verified disease-code mapping. Unsupported or missing graph evidence is recorded; the hybrid model can then answer from its own knowledge. The same benchmark answer list scores both hops, retaining the original dataset's reference-set limitations. Compare the five new variants under identical quantization, generation and scoring settings.
