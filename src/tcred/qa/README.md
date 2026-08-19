# QA baselines

The QA package implements four comparable systems over a shared corpus:

- vector retrieval;
- vector retrieval with temporal filtering;
- graph retrieval without temporal constraints;
- temporal graph retrieval.

All systems emit the same typed output and trace models, allowing downstream
human and automatic metrics to compare answer and evidence behavior. Checkpoint
manifests capture configuration and input identity for restartable runs.

Use `uv run tcred run-qa-systems --help` for local execution and
`uv run tcred run-qa-systems-batch --help` for batch processing. System outputs
are generated artifacts and are intentionally not bundled in this repository.
