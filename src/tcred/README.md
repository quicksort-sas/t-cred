# `tcred` package

The package is organized by stable responsibility:

- `dataset`: temporal graph schemas, generation, release, validation, and audit.
- `qa`: baseline retrieval and question-answering systems.
- `human_eval`: evaluation sampling, imports, QC, final gold schemas, and scoring.
- `metrics`: automatic and model-assisted evaluation metrics.
- `trainable_metrics`: T-CRED-SL data preparation, training, and inference.
- `external`: converters for user-supplied third-party datasets.
- `external_evaluations`: adapters for external benchmark implementations.
- `llm`: hosted-model providers, paraphrasing, and batch-job helpers.

`cli.py` exposes the `tcred` command. The annotation web application and
administration service intentionally live in the separate
[T-CRED annotation tool](https://github.com/Muradmustafayev-03/tcred_annotation_tool).

Serialized models use explicit schema/protocol versions. Prefer the public
Pydantic models and writer functions over constructing rows by hand.
