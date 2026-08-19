# T-CRED

T-CRED is a Python toolkit and public data release for evaluating temporal
correctness in retrieval-augmented question answering over changing knowledge
graphs. It provides deterministic synthetic-data generation, QA baselines,
human-evaluation data models, automatic metrics, source-disjoint validation,
and trainable metric tooling.

## Structure

- `src/tcred/dataset`: temporal graph models, generation, validation, auditing,
  release construction, and source-disjoint workflows.
- `src/tcred/qa`: lexical, vector, graph, and temporal QA baselines.
- `src/tcred/human_eval`: sampling, assignment, import, quality-control, final
  gold schema, and system-scoring utilities.
- `src/tcred/metrics`: deterministic, neural, task-judge, diagnostic, and T-CRED
  metric suites.
- `src/tcred/trainable_metrics`: the T-CRED-SL data, training, packaging, and
  inference pipeline.
- `src/tcred/external_evaluations`: adapters for external evaluation suites.
- `data`: the public synthetic release plus de-identified raw and final gold
  evaluation data.

The annotation web application and administration interface are maintained
separately in the
[T-CRED annotation tool](https://github.com/Muradmustafayev-03/tcred_annotation_tool).

## Requirements and installation

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked base environment:

```bash
uv sync --frozen
```

Install optional neural and trainable-metric dependencies when those modules are
needed:

```bash
uv sync --frozen --all-extras
```

The command-line entry points are `tcred` and `tcred-sl`:

```bash
uv run tcred --help
uv run tcred-sl --help
```

Commands that call hosted language models read provider credentials from local
environment variables. Copy `.env.example` to `.env` and populate only the
provider variables you use. `.env` files are ignored by Git.

## Public data

The repository includes three reviewed data layers:

1. `data/generated/tcred_release/tcred_synth`: 600 synthetic temporal scenarios,
   2,400 questions, and their graph, evidence, answer-variant, split, and audit
   records.
2. `data/human_eval/tcred_release/raw/2026-08-13T011632Z`: 210 de-identified raw
   label rows over 138 synthetic evaluation units.
3. `data/human_eval/tcred_release/gold/2026-08-13T011632Z`: a final gold layer for
   the frozen public snapshot, with 65 included units and 283 resolved fields.

The raw release contains release-only rater pseudonyms and no timestamps or
free-text comments. The final gold layer is hybrid: 172 fields were resolved by
matching human judgments and 111 fields by the adjudicator.
The source collection was incomplete, so this dataset must not be described as
a completed 36-annotator study or as purely human-expert gold. Each published
snapshot includes a manifest with byte counts, SHA-256 hashes, scope, and
limitations.

PAT-Questions and HoH-QAs content is not bundled. Their converters remain
available for users who obtain those sources under their upstream terms.

## Common workflows

Score QA-system rows represented in a compatible final gold directory:

```bash
uv run tcred evaluate-systems-on-human-gold \
  --gold-dir data/human_eval/tcred_release/gold/2026-08-13T011632Z \
  --output-dir artifacts/gold-system-performance
```

Generate synthetic data from a locally frozen source collection:

```bash
uv run tcred generate-synthetic --help
uv run tcred build-release --help
```
