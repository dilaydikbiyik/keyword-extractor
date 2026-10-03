#!/usr/bin/env python3
"""What each system costs per document, measured the same way for all of them.

    python -m experiments.benchmark            # every system, into results/benchmark.json
    python -m experiments.benchmark --system embed-minilm --raw   # one system, to stdout

Accuracy says which system to prefer; cost says whether it can be run. This
measures the second on the documents of the evaluation set, with the same
interpreter, the same machine and the same inputs for every system.

What is measured, per system:

* **cold start**: building the system once, which loads the model or the frozen
  statistics and encodes the class texts. Class vectors are computed once and
  reused, so this cost is paid per process, not per document.
* **per document, batch 1**: warm latency, repeated, reported as median and 95th
  percentile. This is the number a request-per-document service would see.
* **throughput in batch**: documents per second when the documents are handed
  over together, which is what a batch job gets.
* **peak memory**: the maximum resident set size of the process that ran the
  system, and the size of the model on disk.

Each system runs in its own process, so its memory is its own. Embedding caches
are bypassed: a cache would make the second repetition meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.data import load_labeled_samples, load_taxonomy

RESULT = RESULTS_DIR / "benchmark.json"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
WARMUP = 5
DOCUMENTS = 100
REPEATS = 3
LLM_DOCUMENTS = 30
LLM_REPEATS = 1
BATCH = 32


def directory_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    # The Hugging Face cache stores each file once as a blob and links to it from
    # the snapshot; counting both would double every model.
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
    return round(total / 1024 / 1024, 1)


def model_disk_mb(name: str) -> float:
    """Size on disk of a Hugging Face model, or of the committed statistics."""
    if name.startswith("data/"):
        return round((ROOT / name).stat().st_size / 1024 / 1024, 1)
    return directory_mb(HF_CACHE / ("models--" + name.replace("/", "--")))


def cpu_name() -> str:
    """The processor as run.py records it, so both reports name the same machine."""
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return round(peak / (1024 if sys.platform.startswith("linux") else 1024 * 1024), 1)


# ── the systems, each as (build once) -> (classify one document) ─────────────

def tfidf_baseline_system():
    """The paper's lexical baseline: word TF-IDF over the committed corpus statistics."""
    from experiments.systems import TfidfRanker

    ranker = TfidfRanker()
    ranker.fit([])  # no raw corpus: the committed vocabulary and IDF weights

    def classify(text: str) -> str:
        return ranker.rank(text)[0][0]

    return classify, lambda texts: [classify(t) for t in texts]


def char_system():
    """The stronger lexical space: character 3-5-grams, frozen vocabulary and IDF."""
    from experiments.run_references import load_space, sector_documents

    taxonomy = load_taxonomy()
    codes = sorted(taxonomy)
    space = load_space("tfidf-char")
    sectors = np.vstack([space.transform(d) for d in sector_documents(taxonomy, codes)])

    def classify(text: str) -> str:
        return codes[int(np.argmax(space.transform(text) @ sectors.T))]

    return classify, lambda texts: [classify(t) for t in texts]


def embedding_system(model_name: str, device: str = "cpu"):
    from experiments.systems import get_embedder

    embedder = get_embedder(model_name, device)
    taxonomy = load_taxonomy()
    codes = sorted(taxonomy)
    texts = [f"{taxonomy[c].get('name', '')}. {taxonomy[c].get('description', '')}" for c in codes]
    sectors = np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts])
    sectors /= np.linalg.norm(sectors, axis=1, keepdims=True)

    def vector(text: str) -> np.ndarray:
        v = np.asarray(embedder.embed_text(text, use_cache=False), dtype=np.float64)
        return v / np.linalg.norm(v)

    def classify(text: str) -> str:
        return codes[int(np.argmax(vector(text) @ sectors.T))]

    def classify_batch(texts: Sequence[str]) -> List[str]:
        matrix = np.vstack([np.asarray(v, dtype=np.float64)
                            for v in embedder.embed_texts(list(texts), batch_size=BATCH)])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return [codes[i] for i in np.argmax(matrix @ sectors.T, axis=1)]

    return classify, classify_batch


def llm_system():
    from experiments.systems import LOCAL_LLM, LocalLLMRanker

    ranker = LocalLLMRanker(LOCAL_LLM)

    def classify(text: str) -> str:
        return ranker.rank(text)[0][0]

    return classify, None


SYSTEMS: Dict[str, Dict] = {
    "tfidf-word": {"label": "TF-IDF, words", "device": "cpu",
                   "model": "data/derived/tfidf_corpus_stats.json",
                   "build": tfidf_baseline_system},
    "tfidf-char": {"label": "TF-IDF, character 3-5-grams", "device": "cpu",
                   "model": "data/derived/tfidf_char_corpus_stats.json",
                   "build": char_system},
    "embed-minilm": {"label": "MiniLM embeddings (118M)", "device": "cpu",
                     "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     "build": lambda device="cpu": embedding_system(
                         "paraphrase-multilingual-MiniLM-L12-v2", device)},
    "embed-mpnet": {"label": "mpnet embeddings (278M)", "device": "cpu",
                    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "build": lambda device="cpu": embedding_system(
                        "paraphrase-multilingual-mpnet-base-v2", device)},
    "llm-qwen": {"label": "Qwen2.5-7B-Instruct, asked directly", "device": "mps/cuda/cpu, whichever is present",
                 "model": "Qwen/Qwen2.5-7B-Instruct",
                 "build": llm_system},
}


def measure(key: str, device: str = "cpu") -> Dict:
    """Cold start, warm per-document latency, batch throughput and peak memory."""
    spec = SYSTEMS[key]
    samples = load_labeled_samples()
    n_docs = LLM_DOCUMENTS if key == "llm-qwen" else DOCUMENTS
    repeats = LLM_REPEATS if key == "llm-qwen" else REPEATS
    texts = [s.purpose for s in samples][:n_docs]

    start = time.perf_counter()
    try:
        classify, classify_batch = spec["build"](device)
    except TypeError:
        # The lexical and LLM systems take no device: the first runs on the CPU
        # by construction, the second chooses its own.
        classify, classify_batch = spec["build"]()
    cold_start = time.perf_counter() - start

    for text in texts[:WARMUP]:
        classify(text)

    per_document: List[float] = []
    for _ in range(repeats):
        for text in texts:
            t0 = time.perf_counter()
            classify(text)
            per_document.append(1000 * (time.perf_counter() - t0))

    batch_docs_per_second = None
    if classify_batch is not None:
        classify_batch(texts[:WARMUP])
        t0 = time.perf_counter()
        classify_batch(texts)
        batch_docs_per_second = round(len(texts) / (time.perf_counter() - t0), 1)

    per_document.sort()
    return {
        "key": key, "label": spec["label"],
        "device": device if "embed" in key else spec["device"],
        "n_documents": len(texts), "repeats": repeats, "n_timings": len(per_document),
        "cold_start_seconds": round(cold_start, 2),
        "median_ms": round(statistics.median(per_document), 2),
        "p95_ms": round(per_document[min(len(per_document) - 1, int(0.95 * len(per_document)))], 2),
        "docs_per_second_single": round(1000 / statistics.median(per_document), 1),
        "docs_per_second_batch": batch_docs_per_second,
        "peak_rss_mb": peak_rss_mb(),
        "model_disk_mb": model_disk_mb(spec["model"]),
    }


def run_in_subprocess(key: str, device: str = "cpu") -> Dict:
    """Each system in its own process, so peak memory is its own."""
    out = subprocess.run([sys.executable, "-m", "experiments.benchmark", "--system", key,
                          "--device", device, "--raw"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{key} failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=list(SYSTEMS), help="Measure one system.")
    parser.add_argument("--raw", action="store_true", help="Print the measurement as JSON and stop.")
    parser.add_argument("--skip-llm", action="store_true", help="Leave the 7B model out (it needs 14 GB on disk).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Where the encoders run. The paper's table is the cpu run; another "
                             "device is written beside it, to show whether the ordering moves.")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    if args.system:
        report = measure(args.system, args.device)
        if args.raw:
            print(json.dumps(report))
        return 0

    keys = [k for k in SYSTEMS if not (args.skip_llm and k == "llm-qwen")]
    measurements = []
    for key in keys:
        print(f"  {key} ...", flush=True)
        measurements.append(run_in_subprocess(key, args.device))
    payload = {
        "machine": {"cpu": cpu_name(), "cpu_count": os.cpu_count(),
                    "platform": platform.platform(terse=True), "python": platform.python_version()},
        "protocol": {"documents": DOCUMENTS, "repeats": REPEATS, "warmup": WARMUP, "batch_size": BATCH,
                     "llm_documents": LLM_DOCUMENTS, "seed": SEED,
                     "note": "Warm latency at batch 1, median and p95 over all timings; embedding caches "
                             "bypassed; class vectors built once in cold start, as a service would; each "
                             "system in its own process."},
        "encoder_device": args.device,
        "systems": measurements,
    }
    result = RESULT if args.device == "cpu" else RESULT.with_name(f"benchmark_{args.device}.json")
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for m in measurements:
        print(f"  {m['label']:42s} {m['median_ms']:8.2f} ms  p95 {m['p95_ms']:8.2f}  "
              f"{str(m['docs_per_second_batch'] or '-'):>7s} docs/s batch  {m['peak_rss_mb']:7.1f} MB")
    print(f"Wrote {result.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
