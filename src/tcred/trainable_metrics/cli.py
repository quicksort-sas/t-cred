from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import orjson
import typer
from rich.console import Console

from tcred.trainable_metrics.config import (
    DataBuildConfig,
    TrainingConfig,
    load_yaml_model,
)

app = typer.Typer(no_args_is_help=True, help="T-CRED-SL data preparation and training.")
console = Console()


@app.command("acquire-local")
def acquire_local(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = Path("data/trainable_metrics/raw"),
    backbone_root: Annotated[Path, typer.Option("--backbone-root")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384"
    ),
) -> None:
    """Prepare checksum-pinned local sources and the safe MiniLM checkpoint."""
    from tcred.trainable_metrics.acquire import (
        prepare_attribution_bench,
        prepare_backbone,
        prepare_fever,
        prepare_mocha,
    )

    prepare_mocha(raw_root=raw_root)
    prepare_fever(raw_root=raw_root)
    prepare_attribution_bench(raw_root=raw_root)
    prepare_backbone(
        model_id="microsoft/MiniLM-L12-H384-uncased",
        revision="44acabbec0ef496f6dbc93adadea57f376b7c0ec",
        output_dir=backbone_root,
    )
    console.print("[green]Local acquisition artifacts are ready.[/green]")


@app.command("build-exclusion-ledger")
def build_exclusion_ledger_command(
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "data/trainable_metrics/raw/project_formal/exclusion_ledger.json"
    ),
) -> None:
    from tcred.trainable_metrics.source_exclusions import build_source_exclusion_ledger

    ledger = build_source_exclusion_ledger(data_root=data_root)
    _write_json(output_path, ledger.as_dict())
    console.print(
        f"[green]Protected {len(ledger.source_ids)} source series and "
        f"{len(ledger.entity_ids)} Wikidata entities.[/green]"
    )


@app.command("extract-project-sources")
def extract_project_sources(
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "data/trainable_metrics/raw/project_formal/fresh_sources.jsonl"
    ),
    target: Annotated[int, typer.Option("--target", min=4)] = 4_000,
    candidate_rows_per_property: Annotated[
        int, typer.Option("--candidate-rows-per-property", min=1_000)
    ] = 30_000,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(
        "data/trainable_metrics/raw/project_formal/wikidata_cache"
    ),
) -> None:
    """Extract fresh, protected-artifact-disjoint Wikidata temporal series."""
    from tcred.dataset.extracted_source import extract_wikidata_temporal_sources
    from tcred.trainable_metrics.source_exclusions import build_source_exclusion_ledger

    ledger = build_source_exclusion_ledger(data_root=data_root)
    ledger_path = output_path.parent / "exclusion_ledger.json"
    _write_json(ledger_path, ledger.as_dict())
    sources = extract_wikidata_temporal_sources(
        output_path=output_path,
        target=target,
        candidate_rows_per_property=candidate_rows_per_property,
        cache_dir=cache_dir,
        sampling_salt="tcred-temporal-series-v1",
        hash_buckets=tuple("123456789abcdef"),
        excluded_source_ids=set(ledger.source_ids),
        excluded_entity_ids=set(ledger.entity_ids),
    )
    console.print(f"[green]Extracted {len(sources)} fresh disjoint source series.[/green]")


@app.command("build-project-formal")
def build_project_formal_command(
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    source_path: Annotated[Path, typer.Option("--source-path")] = Path(
        "data/trainable_metrics/raw/project_formal/fresh_sources.jsonl"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/trainable_metrics/raw/project_formal"
    ),
    scenarios: Annotated[int, typer.Option("--scenarios", min=4)] = 4_000,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    from tcred.dataset.extracted_source import load_extracted_sources
    from tcred.trainable_metrics.project_formal import build_project_formal_dataset
    from tcred.trainable_metrics.source_exclusions import build_source_exclusion_ledger

    ledger = build_source_exclusion_ledger(data_root=data_root)
    manifest = build_project_formal_dataset(
        sources=load_extracted_sources(source_path),
        output_dir=output_dir,
        exclusion_ledger=ledger,
        scenario_count=scenarios,
        overwrite=overwrite,
    )
    console.print(f"[green]Built {manifest['record_count']} formal semantic records.[/green]")


@app.command("build-corpus")
def build_corpus_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path(
        "configs/trainable_metrics/data.yaml"
    ),
    raw_root: Annotated[Path, typer.Option("--raw-root")] = Path("data/trainable_metrics/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/trainable_metrics/processed/corpus_v1"
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    from tcred.trainable_metrics.builder import build_semantic_corpus

    config = cast(DataBuildConfig, load_yaml_model(config_path, DataBuildConfig))
    manifest = build_semantic_corpus(
        config=config,
        raw_root=raw_root,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Built {manifest['integrity']['selected_rows']} canonical corpus rows.[/green]"
    )


@app.command("audit-near-duplicates")
def audit_near_duplicates_command(
    corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
        "data/trainable_metrics/processed/corpus_v1"
    ),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "data/trainable_metrics/processed/corpus_v1/near_duplicate_audit.json"
    ),
    threshold: Annotated[float, typer.Option("--threshold", min=0.5, max=1.0)] = 0.90,
    candidate_threshold: Annotated[
        float, typer.Option("--candidate-threshold", min=0.1, max=1.0)
    ] = 0.60,
) -> None:
    from tcred.trainable_metrics.audit import audit_cross_partition_near_duplicates

    report = audit_cross_partition_near_duplicates(
        corpus_dir=corpus_dir,
        output_path=output_path,
        threshold=threshold,
        candidate_threshold=candidate_threshold,
        num_perm=128,
        seed=20260817,
    )
    color = "green" if report["status"] == "passed" else "red"
    console.print(
        f"[{color}]{report['status']}: {report['cross_partition_collisions']} "
        f"cross-partition near duplicates.[/{color}]"
    )
    if report["status"] != "passed":
        raise typer.Exit(2)


@app.command("pretokenize")
def pretokenize_command(
    corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
        "data/trainable_metrics/processed/corpus_v1"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    max_length: Annotated[int, typer.Option("--max-length", min=32, max=512)] = 256,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    from tcred.trainable_metrics.preprocessing import pretokenize_corpus

    manifest = pretokenize_corpus(
        corpus_dir=corpus_dir,
        backbone_dir=backbone_dir,
        output_dir=output_dir,
        max_length=max_length,
        overwrite=overwrite,
    )
    rows = sum(item["rows"] for item in manifest["artifacts"].values())
    console.print(f"[green]Pretokenized {rows} rows.[/green]")


@app.command("readiness")
def readiness_command(
    data_config_path: Annotated[Path, typer.Option("--data-config")] = Path(
        "configs/trainable_metrics/data.yaml"
    ),
    training_config_path: Annotated[Path, typer.Option("--training-config")] = Path(
        "configs/trainable_metrics/training.yaml"
    ),
    corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
        "data/trainable_metrics/processed/corpus_v1"
    ),
    tokenized_dir: Annotated[Path, typer.Option("--tokenized-dir")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "data/trainable_metrics/processed/gpu_readiness.json"
    ),
) -> None:
    from tcred.trainable_metrics.readiness import validate_gpu_readiness

    report = validate_gpu_readiness(
        data_config=cast(DataBuildConfig, load_yaml_model(data_config_path, DataBuildConfig)),
        training_config=cast(
            TrainingConfig,
            load_yaml_model(training_config_path, TrainingConfig),
        ),
        corpus_dir=corpus_dir,
        tokenized_dir=tokenized_dir,
        backbone_dir=backbone_dir,
        near_duplicate_report=corpus_dir / "near_duplicate_audit.json",
    )
    _write_json(output_path, report)
    color = "green" if report["status"] == "ready" else "red"
    console.print(f"[{color}]GPU readiness: {report['status']}[/{color}]")
    if report["status"] != "ready":
        for check in report["checks"]:
            if check["status"] == "failed":
                console.print(f"- {check['name']}: {check['detail']}")
        raise typer.Exit(2)


@app.command("package-gpu")
def package_gpu_command(
    workspace_root: Annotated[Path, typer.Option("--workspace-root")] = Path("."),
    corpus_dir: Annotated[Path, typer.Option("--corpus-dir")] = Path(
        "data/trainable_metrics/processed/corpus_v1"
    ),
    tokenized_dir: Annotated[Path, typer.Option("--tokenized-dir")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    backbone_root: Annotated[Path, typer.Option("--backbone-root")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384"
    ),
    readiness_path: Annotated[Path, typer.Option("--readiness-path")] = Path(
        "data/trainable_metrics/processed/gpu_readiness.json"
    ),
    config_dir: Annotated[Path, typer.Option("--config-dir")] = Path(
        "configs/trainable_metrics"
    ),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "build/tcred-sl-private-gpu-bundle.tar.gz"
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Package only GPU-required artifacts after the readiness gate passes."""
    from tcred.trainable_metrics.package import create_private_gpu_bundle

    result = create_private_gpu_bundle(
        workspace_root=workspace_root,
        output_path=output_path,
        corpus_dir=corpus_dir,
        tokenized_dir=tokenized_dir,
        backbone_root=backbone_root,
        readiness_path=readiness_path,
        config_dir=config_dir,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Private GPU bundle verified: {result['archive']['path']} "
        f"({result['archive']['sha256']}).[/green]"
    )


@app.command("verify-gpu-bundle")
def verify_gpu_bundle_command(
    path: Annotated[Path, typer.Argument()] = Path(
        "build/tcred-sl-private-gpu-bundle.tar.gz"
    ),
) -> None:
    from tcred.trainable_metrics.package import verify_private_gpu_bundle

    result = verify_private_gpu_bundle(path)
    console.print(
        f"[green]GPU bundle verified: {result['payload_files']} files, "
        f"{result['payload_bytes']} bytes.[/green]"
    )


@app.command("train")
def train_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path(
        "configs/trainable_metrics/training.yaml"
    ),
    tokenized_dir: Annotated[Path, typer.Option("--tokenized-dir")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "runs/tcred-sl-minilm-l12-seed42"
    ),
    resume_from: Annotated[Path | None, typer.Option("--resume-from")] = None,
) -> None:
    from tcred.trainable_metrics.trainer import train_semantic_metric

    config = cast(TrainingConfig, load_yaml_model(config_path, TrainingConfig))
    summary = train_semantic_metric(
        config=config,
        tokenized_dir=tokenized_dir,
        backbone_dir=backbone_dir,
        output_dir=output_dir,
        resume_from=resume_from,
    )
    console.print(
        f"[green]Training finished at step {summary.get('completed_steps', 'worker')}.[/green]"
    )


@app.command("validate-export")
def validate_export_command(
    export_root: Annotated[Path, typer.Argument()] = Path(
        "data/trainable_metrics/models/tcred-sl-minilm-l12-a100-seed42"
    ),
    output_path: Annotated[Path | None, typer.Option("--output-path", "-o")] = None,
) -> None:
    """Verify a downloaded training export, including archive and model hashes."""
    from tcred.trainable_metrics.artifacts import validate_training_export

    report = validate_training_export(export_root, output_path=output_path)
    console.print(
        f"[green]Export verified: {report['run']['completed_steps']} steps, "
        f"weight {report['model']['weight_sha256']}.[/green]"
    )
    for warning in report["warnings"]:
        console.print(f"[yellow]Warning: {warning['kind']}[/yellow]")


@app.command("compare-checkpoint")
def compare_checkpoint_command(
    export_root: Annotated[Path, typer.Option("--export-root")] = Path(
        "data/trainable_metrics/models/tcred-sl-minilm-l12-a100-seed42"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    population_dir: Annotated[Path, typer.Option("--population-dir")] = Path(
        "data/metrics/tcred_suite/population-v1.4"
    ),
    source_disjoint_root: Annotated[Path, typer.Option("--source-disjoint-root")] = Path(
        "data/validation/tcred_v1_4_source_disjoint"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/tcred_sl/seed42-2026-08-16"
    ),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
    bootstrap_samples: Annotated[
        int, typer.Option("--bootstrap-samples", min=100)
    ] = 2_000,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Score frozen evaluation artifacts and compare T-CRED-SL with T-CRED v1.4."""
    from tcred.trainable_metrics.comparison import evaluate_checkpoint_against_tcred_v14

    result = evaluate_checkpoint_against_tcred_v14(
        repository_root=Path.cwd(),
        export_root=export_root,
        backbone_dir=backbone_dir,
        population_dir=population_dir,
        source_disjoint_root=source_disjoint_root,
        output_dir=output_dir,
        batch_size=batch_size,
        bootstrap_samples=bootstrap_samples,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Comparison complete: {result['human_gold']['units']} human-gold units and "
        f"{result['source_disjoint_diagnostics']['pairs']} formal pairs.[/green]"
    )


@app.command("gpu-smoke")
def gpu_smoke_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path(
        "configs/trainable_metrics/training.a100-80gb.yaml"
    ),
    tokenized_dir: Annotated[Path, typer.Option("--tokenized-dir")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "runs/gpu_training_smoke.json"
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Verify one full CUDA optimization step before the paid training run."""
    from tcred.trainable_metrics.gpu_smoke import run_gpu_training_smoke

    report = run_gpu_training_smoke(
        config=cast(TrainingConfig, load_yaml_model(config_path, TrainingConfig)),
        tokenized_dir=tokenized_dir,
        backbone_dir=backbone_dir,
        output_path=output_path,
        overwrite=overwrite,
    )
    console.print(
        f"[green]GPU smoke passed: loss={report['step']['loss']:.6f}, "
        f"peak_reserved={report['memory']['peak_reserved_bytes']} bytes.[/green]"
    )


@app.command("verify-gpu-smoke")
def verify_gpu_smoke_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path(
        "configs/trainable_metrics/training.a100-80gb.yaml"
    ),
    tokenized_dir: Annotated[Path, typer.Option("--tokenized-dir")] = Path(
        "data/trainable_metrics/processed/tokenized_v1"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    report_path: Annotated[Path, typer.Option("--report-path")] = Path(
        "runs/gpu_training_smoke.json"
    ),
) -> None:
    """Verify that the GPU smoke report belongs to the current run inputs."""
    from tcred.trainable_metrics.gpu_smoke import validate_gpu_training_smoke

    validate_gpu_training_smoke(
        config=cast(TrainingConfig, load_yaml_model(config_path, TrainingConfig)),
        tokenized_dir=tokenized_dir,
        backbone_dir=backbone_dir,
        report_path=report_path,
    )
    console.print("[green]GPU smoke report matches the current run inputs and environment.[/green]")


@app.command("smoke")
def smoke_command(
    workspace: Annotated[Path, typer.Option("--workspace")] = Path(
        ".tmp/tcred-sl-cpu-smoke"
    ),
    backbone_dir: Annotated[Path, typer.Option("--backbone-dir")] = Path(
        "data/trainable_metrics/backbone/minilm-l12-h384/safe_checkpoint"
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    from tcred.trainable_metrics.smoke import run_cpu_pipeline_smoke

    summary = run_cpu_pipeline_smoke(
        workspace=workspace,
        backbone_dir=backbone_dir,
        overwrite=overwrite,
    )
    console.print(
        f"[green]CPU pipeline smoke passed at step {summary['completed_steps']}.[/green]"
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


if __name__ == "__main__":
    app()
