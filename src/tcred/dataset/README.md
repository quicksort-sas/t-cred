# Dataset toolkit

This package defines T-CRED temporal graph records and the deterministic
generation pipeline.

The main layers are:

- `models.py`, `graph.py`, and `intervals.py`: typed records and temporal logic.
- `generator.py` and `source_grounded.py`: synthetic scenario construction.
- `solver.py`, `verbalize.py`, and `text.py`: question solving and rendering.
- `writer.py`, `io.py`, and `validate.py`: atomic serialization and invariants.
- `audit.py` and `reporting.py`: structural coverage and generation summaries.
- `release.py`: checksum-manifested multi-family release assembly.
- `source_disjoint_validation.py`: source-disjoint challenge construction.

Generation is seed-controlled, but a release is identified by its manifest and
file hashes rather than by seed alone. Preserve JSONL ordering and verify the
manifest when moving artifacts between systems.

Use `uv run tcred generate-synthetic --help`, `build-release --help`, and
`audit-dataset --help` for the corresponding command contracts.
