# Research Metrics

Benchmark command:

```bash
python benchmarks/benchmark_hlm.py --iterations 100 --output benchmark-results.json
python -m json.tool benchmark-results.json
```

Environment: CPU execution, deterministic synthetic tensor inputs, live Hugging Face
model loading disabled by default.

## Recorded Benchmark Results

| Benchmark | Mean latency | Median latency | p95 latency | Evidence |
|---|---:|---:|---:|---|
| Tokenizer document encoding | 0.007402 ms | 0.006550 ms | 0.008000 ms | `benchmark-results.json` |
| Dataset materialization | 0.028561 ms | 0.025100 ms | 0.044400 ms | `benchmark-results.json` |
| Model forward pass | 2.207161 ms | 2.189750 ms | 2.790500 ms | `benchmark-results.json` |
| API safe fallback generation | 0.000217 ms | 0.000200 ms | 0.000200 ms | `benchmark-results.json` |

## Quality Metrics

| Dimension | Recorded value | Notes |
|---|---:|---|
| Test count | 13 passing tests | Unit, API contract, tokenizer/config/dataset, and encoder tests |
| Runtime package coverage | 87% | `pytest --cov=api --cov=hierarchical_lm --cov-report=term-missing` |
| Benchmark JSON validity | 100% for current run | Validated with `python -m json.tool` |
| Live model requirement in CI | 0 gated downloads | Live HF loading is opt-in with `HLM_ENABLE_LIVE_MODEL=true` |
| Cross-platform log collision | Resolved in branch | Consolidated to lowercase `dailylog.md` |

## Production Target Metrics

These are engineering targets for future live infrastructure validation, not claims from
the deterministic local benchmark.

| Capability | Target | Validation path |
|---|---:|---|
| API health endpoint p95 | < 50 ms | FastAPI integration benchmark |
| CPU forward-pass p95 | < 10 ms for compact config | `benchmarks/benchmark_hlm.py` |
| Benchmark JSON validity | 100% of CI runs | Tier 8 hygiene gate |
| CI masked failures | 0 | No `|| echo` bypasses in hygiene workflow |
| Verified secret findings | 0 | Security scan tier plus GitHub secret scanning |
