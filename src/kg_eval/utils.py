"""Data caching, normalization, and reproducible result paths (no LLM imports)."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SRC_DIR = Path(__file__).resolve().parents[1]
KG_EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = KG_EVAL_DIR / 'results'
DATA_DIR = SRC_DIR / 'data' / 'knowledge_graphs'
KG_CSV_PATH = SRC_DIR / 'data' / 'kg.csv'
KG_PARQUET_PATH = SRC_DIR / 'data' / 'kg.parquet'
BIOHOPR_REVISION = '08f06692c3900347e7405ba55ff6a6d330f55ddb'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize(value):
    if value is None or (not isinstance(value, (str, list, np.ndarray)) and pd.isna(value)):
        return ''
    return ' '.join(str(value).split()).casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def download(url, path):
    """Cache a public source atomically and record its content hash and URL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f'Downloading {path.name}...', flush=True)
        temporary = path.with_suffix(path.suffix + '.part')
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(path)
    metadata_path = path.with_suffix(path.suffix + '.source.json')
    digest = sha256_file(path)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['sha256'] != digest or metadata['url'] != url:
            raise ValueError(f'Cached source differs from its manifest: {path}')
    else:
        write_json(metadata_path, {'url': url, 'sha256': digest, 'bytes': path.stat().st_size,
                                   'recorded_at': utc_now()})
    return path


def parse_targets(targets):
    """Accept native arrays or JSON/Python lists; do not split names on commas."""
    if isinstance(targets, str):
        if not targets.strip():
            return []
        if targets.lstrip().startswith('['):
            try:
                targets = json.loads(targets)
            except json.JSONDecodeError:
                targets = ast.literal_eval(targets)
            if not isinstance(targets, list):
                raise ValueError('Serialized answers must be a list')
        else:
            targets = [targets]
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()
    if targets is None or (not isinstance(targets, list) and pd.isna(targets)):
        return []
    if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
        raise ValueError('Answers must be a list of strings')
    return sorted({normalize(t) for t in targets if normalize(t)})


def load_benchmark(dataset_name='knowlab-research/BioHopR', split=None, revision=None):
    if dataset_name == 'knowlab-research/BioHopR':
        if split not in (None, 'train'):
            raise ValueError('The BioHopR JSON has a single split: train')
        revision = revision or BIOHOPR_REVISION
        folder = SRC_DIR / 'data' / 'benchmarks' / 'BioHopR'
        filename = 'BioHopR.json' if revision == BIOHOPR_REVISION else f'BioHopR-{revision}.json'
        url = f'https://huggingface.co/datasets/{dataset_name}/resolve/{revision}/BioHopR.json'
        path = download(url, folder / filename)
        df = pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
        provenance = {'dataset': dataset_name, 'split': 'train', 'revision': revision,
                      'sha256': sha256_file(path), 'url': url}
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, revision=revision)
        if split is None and len(ds) != 1:
            raise ValueError('Specify --split for a dataset with multiple splits')
        split = split or next(iter(ds))
        df = ds[split].to_pandas()
        provenance = {'dataset': dataset_name, 'split': split, 'revision': revision,
                      'fingerprint': ds[split]._fingerprint}
    required = {'hop1', 'hop1_type', 'hop2', 'hop2_type', 'target_type', 'answer',
                'hop1_question', 'hop2_question', 'relation_hop1', 'relation_hop2'}
    if required - set(df):
        raise ValueError(f'Missing benchmark columns: {sorted(required - set(df))}')
    df.attrs['provenance'] = provenance
    print(f'Benchmark: {len(df):,} rows', flush=True)
    return df


def create_run_dir(dataset_name, kg_name, max_samples=None, results_dir=None, relation_policy='semantic'):
    scope = 'full' if max_samples is None else f'first-{max_samples}'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(results_dir or RESULTS_DIR) / dataset_name.split('/')[-1] / kg_name / f'{stamp}_{relation_policy}_{scope}'
    path.mkdir(parents=True, exist_ok=False)
    return path
