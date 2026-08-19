# External dataset converters

This package converts user-supplied PAT-Questions and HoH-QAs sources into T-CRED
records and validates the resulting family bundles.

No upstream dataset is bundled here. Users are responsible for obtaining inputs
and complying with upstream licenses and terms. Converted outputs can retain
upstream questions, answers, entity labels, and evidence, so they must not be
redistributed merely because the converter code is available.

Inspect `uv run tcred convert-pat --help` and
`uv run tcred convert-hoh --help` for input and output contracts.
