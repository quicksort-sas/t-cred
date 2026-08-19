# LLM helpers

The `llm` package provides provider selection, retry-aware requests,
paraphrasing, and asynchronous batch-job helpers used by data-generation and
evaluation workflows.

Credentials are read from environment variables shown in the root
`.env.example`; they are never accepted as dataset fields or written to release
manifests. Batch inputs and results can contain prompt content and should be
written only to ignored artifact directories.

Use `uv run tcred list-models` to inspect locally configured providers and the
`prepare-paraphrase-batch`, `submit-batch-job`, and
`import-paraphrase-results` command help for batch contracts.
