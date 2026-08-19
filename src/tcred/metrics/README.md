# Metrics

This package evaluates answer correctness, temporal grounding, evidence quality,
and correction quality over T-CRED records.

It contains:

- deterministic lexical and structural metrics;
- optional neural workers with isolated dependency/runtime handling;
- diagnostic-case builders and meta-evaluation reports;
- a task-matched judge adapter;
- the composite T-CRED metric suite;
- source-disjoint comparison, evaluation, and post-hoc audit utilities.

Metric inputs are validated against the final gold and dataset schemas. Hosted
judge calls require a configured provider; neural workers require the
`metrics-neural` extra. Generated predictions, caches, and reports are not source
files and should remain below ignored artifact paths.

Start with:

```bash
uv run tcred evaluate-current-metrics --help
uv run tcred evaluate-tcred-metric-suite --help
uv run tcred evaluate-metric-diagnostics --help
```
