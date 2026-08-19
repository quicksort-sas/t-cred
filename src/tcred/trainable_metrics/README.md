# T-CRED-SL trainable metric

This package implements the source acquisition, corpus construction, exclusion,
near-duplicate audit, preprocessing, training, calibration, evaluation,
packaging, and inference pipeline for the trainable T-CRED-SL metric.

The repository contains pipeline code and configuration only. Downloaded
corpora, processed examples, tokenized shards, model weights, checkpoints, and
evaluation results are excluded. Source-specific licensing rules are encoded in
the acquisition and exclusion configuration and must be reviewed before building
or publishing any derived corpus.

Install the optional dependencies and inspect the staged commands:

```bash
uv sync --frozen --extra metrics-trainable
uv run tcred-sl --help
uv run tcred-sl readiness --help
uv run tcred-sl train --help
uv run tcred-sl validate-export --help
```

GPU packaging helpers produce checksum-manifested bundles for an external worker.
Those bundles and their checkpoints belong in ignored artifact storage, not in
Git.
