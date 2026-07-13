"""Deterministic benchmark harness for the hierarchical language model core."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.inference import LLMEngine
from hierarchical_lm.config import ModelConfig
from hierarchical_lm.dataset import HierarchicalTextDataset
from hierarchical_lm.model import HierarchicalLM
from hierarchical_lm.tokenizer import HierarchicalTokenizer


def measure_latency(function: Callable[[], object], iterations: int) -> dict[str, float | int]:
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        durations.append((time.perf_counter() - started) * 1000)

    sorted_durations = sorted(durations)
    return {
        "iterations": iterations,
        "mean_ms": round(statistics.mean(durations), 6),
        "median_ms": round(statistics.median(durations), 6),
        "p95_ms": round(sorted_durations[int(iterations * 0.95) - 1], 6),
        "min_ms": round(min(durations), 6),
        "max_ms": round(max(durations), 6),
    }


def run_benchmarks(iterations: int) -> dict:
    torch.manual_seed(7)
    config = ModelConfig(vocab_size=128, embed_dim=16, token_hidden=24, doc_hidden=24)
    tokenizer = HierarchicalTokenizer(config)
    model = HierarchicalLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        token_hidden=config.token_hidden,
        doc_hidden=config.doc_hidden,
        num_classes=config.num_classes,
    )
    model.eval()
    sample_text = "The system is scalable. Traffic is dense. The pipeline is efficient."
    input_tensor = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.max_sentences, config.max_seq_len),
    )
    dataset_docs = [sample_text, "The backend system is efficient."]
    dataset_labels = [1, 0]
    fallback_engine = LLMEngine()

    with torch.no_grad():
        return {
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "iterations": iterations,
                "harness": "benchmarks/benchmark_hlm.py",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            "benchmarks": {
                "tokenizer_encode_document": measure_latency(
                    lambda: tokenizer.encode_document(sample_text),
                    iterations,
                ),
                "dataset_materialization": measure_latency(
                    lambda: HierarchicalTextDataset(dataset_docs, dataset_labels, config),
                    iterations,
                ),
                "model_forward_pass": measure_latency(
                    lambda: model(input_tensor),
                    iterations,
                ),
                "api_safe_fallback_generation": measure_latency(
                    lambda: fallback_engine.generate("Explain hierarchical reasoning.", 16),
                    iterations,
                ),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()

    if args.iterations < 20:
        raise SystemExit("--iterations must be at least 20 for stable p95 output")

    results = run_benchmarks(args.iterations)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
