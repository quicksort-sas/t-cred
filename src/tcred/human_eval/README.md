# Human-evaluation utilities

This package contains reusable data logic for preparing and consuming T-CRED
evaluation records. The annotation user interface and administration service are
maintained in the separate
[T-CRED annotation tool](https://github.com/Muradmustafayev-03/tcred_annotation_tool).

Published modules cover:

- controlled sampling and balanced assignment plans;
- blinded public-unit construction and label import;
- protocol validation and quality-control summaries;
- response presentation and complexity features;
- validation of the published final gold schema;
- descriptive QA-system scoring against compatible final gold rows.

Use `uv run tcred import-human-labels --help` for frozen imports and
`uv run tcred evaluate-systems-on-human-gold --help` for the scoring interface.
Write generated reports beneath an ignored output directory.
