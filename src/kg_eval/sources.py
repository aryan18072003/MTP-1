"""Public, versioned graph downloads and authoritative identifier labels."""
import csv
import gzip
import io
import json
import shutil
import tarfile
import zipfile

import requests

try:
    from .utils import DATA_DIR, download, sha256_file, utc_now, write_json
except ImportError:
    from utils import DATA_DIR, download, sha256_file, utc_now, write_json

HETIONET_REVISION = "af668689a81952058df8b1321cc6c744b93d83e2"
HETIONET_NODES = f"https://raw.githubusercontent.com/hetio/hetionet/{HETIONET_REVISION}/hetnet/tsv/hetionet-v1.0-nodes.tsv"
HETIONET_EDGES = f"https://media.githubusercontent.com/media/hetio/hetionet/{HETIONET_REVISION}/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
BIOKG_URL = "https://github.com/dsi-bdi/biokg/releases/download/v1.0.0/biokg.zip"
DRKG_URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream?query=%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29&format=tsv&fields=accession%2Cgene_primary"
GENE_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"


def hetionet_nodes():
    return download(HETIONET_NODES, DATA_DIR / "Hetionet" / "hetionet-v1.0-nodes.tsv")


def biokg_archive():
    return download(BIOKG_URL, DATA_DIR / "BioKG" / "biokg-v1.0.0.zip")


def tsv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def zip_triples(archive, member):
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) != 3:
                    raise ValueError(f"Expected 3 columns in {member}: {row!r}")
                yield row


def drkg_triples():
    """Read only drkg.tsv from the release stream; never unpack embeddings/paths."""
    path = DATA_DIR / "DRKG" / "drkg.tsv"
    manifest = path.with_suffix(".tsv.source.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Downloading DRKG triples (excluding pretrained embeddings)...", flush=True)
        temporary = path.with_suffix(".tsv.part")
        with requests.get(DRKG_URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    if member.name.lstrip("./") == "drkg.tsv" and member.isfile():
                        with archive.extractfile(member) as source, temporary.open("wb") as dest:
                            shutil.copyfileobj(source, dest, length=1024 * 1024)
                        if temporary.stat().st_size != member.size:
                            raise ValueError("Truncated DRKG download")
                        temporary.replace(path)
                        break
                else:
                    raise ValueError("drkg.tsv missing from DRKG release")
    digest = sha256_file(path)
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["sha256"] != digest or saved["url"] != DRKG_URL:
            raise ValueError("DRKG cached triples do not match their manifest")
    else:
        write_json(manifest, {"url": DRKG_URL, "member": "drkg.tsv", "sha256": digest,
                              "bytes": path.stat().st_size, "recorded_at": utc_now()})
    return path


def source_record(path):
    return {"file": str(path), "sha256": sha256_file(path),
            **(json.loads(path.with_suffix(path.suffix + ".source.json").read_text(encoding="utf-8"))
               if path.with_suffix(path.suffix + ".source.json").exists() else {})}
