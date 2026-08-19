# SABET-TKGQA evaluation adapter

This package provides reproducible adapters for auditing and evaluating an
external SABET-TKGQA implementation without bundling its repository, checkpoints,
or run logs.

It includes schema validation, label-bundle handling, matrix recovery, batch
calibration, neural-shard checks, metric inputs, and artifact auditing. Inputs
are always explicit paths; generated results belong in ignored artifact
directories.

The default configuration is
`configs/external_evaluation/sabet_tkgqa.json`. The adapter does not require
redistribution of the external project.
