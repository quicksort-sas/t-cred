from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import orjson
import typer
from rich.console import Console
from rich.table import Table

from tcred.dataset.audit import audit_dataset_dir, write_audit_report
from tcred.dataset.extracted_source import (
    ExtractedSourceDatasetGenerator,
    extract_wikidata_temporal_sources,
    load_extracted_sources,
)
from tcred.dataset.generator import SyntheticDatasetGenerator
from tcred.dataset.release import ReleaseBuildConfig, build_dataset_release
from tcred.dataset.reporting import estimate_generation_runtime, write_generation_report
from tcred.dataset.source_disjoint_validation import prepare_source_disjoint_validation
from tcred.dataset.source_grounded import SourceGroundedDatasetGenerator, load_source_subgraphs
from tcred.dataset.validate import validate_bundle
from tcred.dataset.writer import DatasetWriter
from tcred.external.converters import convert_hoh_dataset, convert_pat_dataset
from tcred.human_eval.assignments import DEFAULT_ASSIGNMENT_SEED
from tcred.human_eval.augmented import export_augmented_human_eval
from tcred.human_eval.export import export_human_eval
from tcred.human_eval.import_labels import import_human_labels
from tcred.human_eval.reassign import reassign_human_eval_package
from tcred.human_eval.system_performance import analyze_system_performance
from tcred.llm.batch import (
    build_paraphrase_tasks,
    import_paraphrase_results,
    write_provider_batch,
)
from tcred.llm.batch_jobs import BatchJobClient
from tcred.llm.paraphrase import LLMParaphraseClient
from tcred.llm.providers import list_available_models
from tcred.metrics.config import ALIGNSCORE_BATCH_SIZE
from tcred.metrics.diagnostic_runner import run_diagnostic_meta_evaluation
from tcred.metrics.reserve_response_addendum import run_reserve_response_addendum
from tcred.metrics.runner import run_current_metrics
from tcred.metrics.source_disjoint_comparators import run_source_disjoint_comparators
from tcred.metrics.source_disjoint_evaluation import run_source_disjoint_tcred_evaluation
from tcred.metrics.source_disjoint_posthoc import run_source_disjoint_posthoc_audit
from tcred.metrics.task_judge_runner import run_task_judge_experiment
from tcred.metrics.tcred_diagnostic_runner import run_tcred_diagnostic_evaluation
from tcred.metrics.tcred_population_runner import run_tcred_population_evaluation
from tcred.qa.batch import run_qa_systems_batch
from tcred.qa.models import QARunConfig, QASystemName
from tcred.qa.runner import run_qa_systems

app = typer.Typer(no_args_is_help=True, help="T-CRED dataset generation utilities.")
console = Console()


@app.command("extract-wikidata-sources")
def extract_wikidata_sources_command(
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = Path(
        "data/external/wikidata/temporal_subgraphs.jsonl"
    ),
    target: Annotated[int, typer.Option("--target", min=1)] = 700,
    candidate_rows_per_property: Annotated[
        int, typer.Option("--candidate-rows-per-property", min=100)
    ] = 5000,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("data/cache/source_extraction"),
) -> None:
    """Freeze provenance-complete Wikidata temporal series for certified generation."""
    sources = extract_wikidata_temporal_sources(
        output_path=output_path,
        target=target,
        candidate_rows_per_property=candidate_rows_per_property,
        cache_dir=cache_dir,
    )
    console.print(f"[green]Extracted {len(sources)} source series:[/green] {output_path}")


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/generated/tcred_synth"
    ),
    scenarios: Annotated[int, typer.Option("--scenarios", "-n", min=1)] = 24,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    questions_per_scenario: Annotated[int, typer.Option("--questions-per-scenario", min=1)] = 4,
    generator_mode: Annotated[
        Literal["source_extracted", "source_grounded", "legacy_template"],
        typer.Option("--generator-mode"),
    ] = "source_extracted",
    source_subgraphs_path: Annotated[Path | None, typer.Option("--source-subgraphs")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Generate from an extracted catalog, pattern fixture, or legacy synthetic template."""
    if generator_mode == "source_extracted":
        if source_subgraphs_path is None:
            raise typer.BadParameter("source_extracted requires --source-subgraphs")
        generator = ExtractedSourceDatasetGenerator(
            seed=seed,
            sources=load_extracted_sources(source_subgraphs_path),
        )
    elif generator_mode == "source_grounded":
        source_subgraphs = (
            load_source_subgraphs(source_subgraphs_path) if source_subgraphs_path else None
        )
        generator = SourceGroundedDatasetGenerator(
            seed=seed,
            source_subgraphs=source_subgraphs,
        )
    else:
        generator = SyntheticDatasetGenerator(seed=seed)
    bundle = generator.generate(
        scenario_count=scenarios,
        questions_per_scenario=questions_per_scenario,
    )
    warnings = validate_bundle(bundle)
    writer = DatasetWriter(output_dir=output_dir, overwrite=overwrite)
    written = writer.write_bundle(bundle)
    report_path = write_generation_report(bundle=bundle, output_dir=output_dir)

    table = Table(title="Synthetic dataset generated")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    table.add_row("generation_report", str(report_path))
    console.print(table)
    if warnings:
        console.print("[yellow]Validation warnings:[/yellow]")
        for warning in warnings:
            console.print(f"- {warning}")
    else:
        console.print("[green]Validation passed with no warnings.[/green]")


@app.command("convert-pat")
def convert_pat(
    pat_data_dir: Annotated[Path, typer.Option("--pat-data-dir")] = Path(
        "data/external/PAT-Questions/PAT-data"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/generated/tcred_pat"
    ),
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Convert PAT-Questions snapshot JSON files into T-CRED artifacts."""
    written = convert_pat_dataset(
        pat_data_dir=pat_data_dir,
        output_dir=output_dir,
        limit=limit,
        overwrite=overwrite,
    )
    _print_artifact_table("PAT conversion complete", written)


@app.command("convert-hoh")
def convert_hoh(
    input_path: Annotated[Path | None, typer.Option("--input-path")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/generated/tcred_hoh"
    ),
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = 200,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Convert HoH JSON/JSONL rows, or fetch rows from Hugging Face if no input is given."""
    written = convert_hoh_dataset(
        input_path=input_path,
        output_dir=output_dir,
        limit=limit,
        overwrite=overwrite,
    )
    _print_artifact_table("HoH conversion complete", written)


@app.command("build-release")
def build_release(
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path(
        "data/generated/tcred_release"
    ),
    human_eval_output_dir: Annotated[Path, typer.Option("--human-eval-output-dir")] = Path(
        "data/human_eval/tcred_release"
    ),
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("data"),
    synthetic_scenarios: Annotated[int, typer.Option("--synthetic-scenarios", min=1)] = 600,
    questions_per_scenario: Annotated[int, typer.Option("--questions-per-scenario", min=1)] = 4,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    pat_data_dir: Annotated[Path, typer.Option("--pat-data-dir")] = Path(
        "data/external/PAT-Questions/PAT-data"
    ),
    pat_limit: Annotated[int, typer.Option("--pat-limit", min=1)] = 300,
    hoh_input_path: Annotated[Path | None, typer.Option("--hoh-input-path")] = None,
    hoh_limit: Annotated[int, typer.Option("--hoh-limit", min=1)] = 200,
    synthetic_generator: Annotated[
        Literal["source_extracted", "source_grounded", "legacy_template"],
        typer.Option("--synthetic-generator"),
    ] = "source_extracted",
    source_subgraphs_path: Annotated[Path | None, typer.Option("--source-subgraphs")] = None,
    synthetic_human_units: Annotated[int, typer.Option("--synthetic-human-units", min=0)] = 160,
    pat_human_units: Annotated[int, typer.Option("--pat-human-units", min=0)] = 40,
    hoh_human_units: Annotated[int, typer.Option("--hoh-human-units", min=0)] = 40,
    annotators: Annotated[int, typer.Option("--annotators", min=1)] = 36,
    assignments_per_annotator: Annotated[
        int,
        typer.Option("--assignments-per-annotator", min=1),
    ] = 20,
    assignment_seed: Annotated[int, typer.Option("--assignment-seed")] = DEFAULT_ASSIGNMENT_SEED,
    include_human_eval: Annotated[bool, typer.Option("--human-eval/--no-human-eval")] = False,
    clean_data: Annotated[bool, typer.Option("--clean-data/--keep-data")] = False,
    strict_audit: Annotated[bool, typer.Option("--strict-audit/--allow-warnings")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Build the full T-CRED release layout: synthetic, PAT, HoH, audits, and manifest."""
    manifest = build_dataset_release(
        ReleaseBuildConfig(
            output_root=output_root,
            synthetic_scenarios=synthetic_scenarios,
            questions_per_scenario=questions_per_scenario,
            seed=seed,
            synthetic_generator=synthetic_generator,
            source_subgraphs_path=source_subgraphs_path,
            pat_data_dir=pat_data_dir,
            pat_limit=pat_limit,
            hoh_input_path=hoh_input_path,
            hoh_limit=hoh_limit,
            human_eval_output_dir=human_eval_output_dir,
            data_root=data_root,
            synthetic_human_units=synthetic_human_units,
            pat_human_units=pat_human_units,
            hoh_human_units=hoh_human_units,
            annotators=annotators,
            assignments_per_annotator=assignments_per_annotator,
            assignment_seed=assignment_seed,
            include_human_eval=include_human_eval,
            clean_data=clean_data,
            strict_audit=strict_audit,
            overwrite=overwrite,
        )
    )

    table = Table(title="T-CRED release built")
    table.add_column("Family")
    table.add_column("Dataset dir")
    table.add_column("Questions")
    table.add_column("Facts")
    table.add_column("Answer variants")
    table.add_column("Warnings")
    for summary in manifest.datasets:
        table.add_row(
            summary.family,
            str(summary.dataset_dir),
            str(summary.question_count),
            str(summary.fact_count),
            str(summary.answer_variant_count),
            str(len(summary.warnings)),
        )
    console.print(table)
    console.print(f"[green]Release manifest:[/green] {output_root / 'release_manifest.json'}")
    if manifest.human_eval:
        console.print(
            f"[green]Human evaluation export:[/green] {manifest.human_eval['output_dir']}"
        )


@app.command("audit-dataset")
def audit_dataset(
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")] = Path(
        "data/generated/tcred_synth"
    ),
    output_path: Annotated[Path | None, typer.Option("--output-path", "-o")] = None,
) -> None:
    """Audit a written dataset directory for diversity and structural coverage."""
    report = audit_dataset_dir(dataset_dir)
    path = write_audit_report(dataset_dir, output_path)
    table = Table(title="Dataset audit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("dataset_dir", str(dataset_dir))
    table.add_row("audit_report", str(path))
    table.add_row("scenarios", str(report.scenario_count))
    table.add_row("questions", str(report.question_count))
    table.add_row("facts", str(report.fact_count))
    table.add_row("answer_variants", str(report.answer_variant_count))
    table.add_row("warnings", str(len(report.warnings)))
    console.print(table)
    if report.warnings:
        console.print("[yellow]Audit warnings:[/yellow]")
        for warning in report.warnings:
            console.print(f"- {warning}")
    else:
        console.print("[green]Audit passed with no warnings.[/green]")


@app.command("estimate-runtime")
def estimate_runtime(
    scenarios: Annotated[int, typer.Option("--scenarios", "-n", min=1)] = 600,
    sample_scenarios: Annotated[int, typer.Option("--sample-scenarios", min=1)] = 12,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    generator_mode: Annotated[
        Literal["source_grounded", "legacy_template"],
        typer.Option("--generator-mode"),
    ] = "source_grounded",
) -> None:
    """Run a tiny local sample and extrapolate full deterministic generation time."""
    estimate = estimate_generation_runtime(
        target_scenarios=scenarios,
        sample_scenarios=sample_scenarios,
        seed=seed,
        generator_mode=generator_mode,
    )
    table = Table(title="Generation runtime estimate")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in estimate.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("list-models")
def list_models() -> None:
    """List API models visible to configured providers without printing secrets."""
    results = asyncio.run(list_available_models())
    table = Table(title="Configured provider model visibility")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Models")
    for result in results:
        models = ", ".join(result.models[:12])
        if len(result.models) > 12:
            models += f", ... ({len(result.models)} total)"
        table.add_row(result.provider, result.status, models or result.message)
    console.print(table)


@app.command("run-qa-systems")
def run_qa_systems_command(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path(
        "data/system_outputs/tcred_release"
    ),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("data/cache/qa"),
    systems: Annotated[
        str,
        typer.Option(
            "--systems",
            help="Comma-separated QA systems; defaults to all four.",
        ),
    ] = ",".join(system.value for system in QASystemName),
    families: Annotated[
        str,
        typer.Option("--families", help="Comma-separated dataset family directories."),
    ] = "tcred_synth,tcred_pat,tcred_hoh",
    embedding_provider: Annotated[
        Literal["mistral", "openai"],
        typer.Option("--embedding-provider"),
    ] = "mistral",
    embedding_model: Annotated[str, typer.Option("--embedding-model")] = "mistral-embed",
    embedding_dimensions: Annotated[
        int,
        typer.Option("--embedding-dimensions", min=256),
    ] = 1024,
    generator_provider: Annotated[
        Literal["groq", "openai", "mistral"],
        typer.Option("--generator-provider"),
    ] = "groq",
    generator_model: Annotated[str, typer.Option("--generator-model")] = ("openai/gpt-oss-20b"),
    reasoning_effort: Annotated[
        Literal["low", "medium", "high"],
        typer.Option("--reasoning-effort"),
    ] = "low",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 10,
    candidate_k: Annotated[int, typer.Option("--candidate-k", min=1)] = 80,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 12,
    limit_per_family: Annotated[int | None, typer.Option("--limit-per-family", min=1)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    revalidate_checkpoints: Annotated[
        bool,
        typer.Option(
            "--revalidate-checkpoints",
            help=(
                "Replay retrieval and prompt construction before adopting checkpoints whose "
                "provenance fingerprint changed."
            ),
        ),
    ] = False,
) -> None:
    """Run the four controlled QA baselines with resumable output checkpoints."""
    selected_systems = [
        QASystemName(value.strip()) for value in systems.split(",") if value.strip()
    ]
    selected_families = [value.strip() for value in families.split(",") if value.strip()]
    manifest = asyncio.run(
        run_qa_systems(
            QARunConfig(
                dataset_root=dataset_root,
                output_root=output_root,
                cache_dir=cache_dir,
                systems=selected_systems,
                families=selected_families,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                generator_provider=generator_provider,
                generator_model=generator_model,
                reasoning_effort=reasoning_effort,
                top_k=top_k,
                candidate_k=candidate_k,
                concurrency=concurrency,
                limit_per_family=limit_per_family,
                seed=seed,
                resume=resume,
                overwrite=overwrite,
            ),
            revalidate_checkpoints=revalidate_checkpoints,
        )
    )
    table = Table(title="QA systems run")
    table.add_column("Family")
    table.add_column("System")
    table.add_column("Succeeded")
    table.add_column("Failed")
    table.add_column("Resumed")
    table.add_column("Output")
    for summary in manifest.summaries:
        table.add_row(
            summary.family,
            str(summary.system_name),
            str(summary.succeeded),
            str(summary.failed),
            str(summary.resumed),
            str(summary.output_path),
        )
    console.print(table)
    console.print(f"[green]Run manifest:[/green] {output_root / 'run_manifest.json'}")
    console.print(f"[green]Diagnostics:[/green] {manifest.diagnostics_path}")


@app.command("run-qa-systems-batch")
def run_qa_systems_batch_command(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path(
        "data/system_outputs/tcred_release"
    ),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("data/cache/qa"),
    systems: Annotated[str, typer.Option("--systems")] = ",".join(
        system.value for system in QASystemName
    ),
    families: Annotated[str, typer.Option("--families")] = ("tcred_synth,tcred_pat,tcred_hoh"),
    generator_model: Annotated[str, typer.Option("--generator-model")] = ("ministral-3b-2512"),
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 10,
    candidate_k: Annotated[int, typer.Option("--candidate-k", min=1)] = 80,
    limit_per_family: Annotated[int | None, typer.Option("--limit-per-family", min=1)] = None,
    poll_seconds: Annotated[float, typer.Option("--poll-seconds", min=1)] = 10.0,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Run the full QA workload through the resumable Mistral Batch API."""
    selected_systems = [
        QASystemName(value.strip()) for value in systems.split(",") if value.strip()
    ]
    selected_families = [value.strip() for value in families.split(",") if value.strip()]
    manifest = asyncio.run(
        run_qa_systems_batch(
            QARunConfig(
                dataset_root=dataset_root,
                output_root=output_root,
                cache_dir=cache_dir,
                systems=selected_systems,
                families=selected_families,
                embedding_provider="mistral",
                embedding_model="mistral-embed",
                embedding_dimensions=1024,
                generator_provider="mistral",
                generator_model=generator_model,
                reasoning_effort="low",
                top_k=top_k,
                candidate_k=candidate_k,
                limit_per_family=limit_per_family,
                seed=seed,
                resume=resume,
                overwrite=overwrite,
            ),
            poll_seconds=poll_seconds,
        )
    )
    table = Table(title="QA batch run")
    table.add_column("Family")
    table.add_column("System")
    table.add_column("Succeeded")
    table.add_column("Failed")
    table.add_column("Output")
    for summary in manifest.summaries:
        table.add_row(
            summary.family,
            str(summary.system_name),
            str(summary.succeeded),
            str(summary.failed),
            str(summary.output_path),
        )
    console.print(table)
    console.print(f"[green]Run manifest:[/green] {output_root / 'run_manifest.json'}")
    console.print(f"[green]Diagnostics:[/green] {manifest.diagnostics_path}")


@app.command("export-augmented-human-eval")
def export_augmented_human_eval_command(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    system_output_root: Annotated[Path, typer.Option("--system-output-root")] = Path(
        "data/system_outputs/tcred_release"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/human_eval/tcred_release"
    ),
    synthetic_units: Annotated[int, typer.Option("--synthetic-units", min=0)] = 160,
    pat_units: Annotated[int, typer.Option("--pat-units", min=0)] = 80,
    hoh_units: Annotated[int, typer.Option("--hoh-units", min=0)] = 80,
    annotators: Annotated[int, typer.Option("--annotators", min=1)] = 36,
    assignments_per_annotator: Annotated[
        int,
        typer.Option("--assignments-per-annotator", min=1),
    ] = 20,
    assignment_seed: Annotated[int, typer.Option("--assignment-seed")] = DEFAULT_ASSIGNMENT_SEED,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Export the preregistered 50/50 controlled and actual-system human sample."""
    dataset_dirs = {
        family: dataset_root / family for family in ("tcred_synth", "tcred_pat", "tcred_hoh")
    }
    written = export_augmented_human_eval(
        dataset_dirs=dataset_dirs,
        system_output_root=system_output_root,
        output_dir=output_dir,
        target_units_by_family={
            "tcred_synth": synthetic_units,
            "tcred_pat": pat_units,
            "tcred_hoh": hoh_units,
        },
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        assignment_seed=assignment_seed,
        overwrite=overwrite,
    )
    table = Table(title="Augmented human evaluation export")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("export-human-eval")
def export_human_eval_command(
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")] = Path(
        "data/generated/tcred_synth"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/human_eval/tcred_synth"
    ),
    target_units: Annotated[int, typer.Option("--target-units", min=1)] = 160,
    annotators: Annotated[int, typer.Option("--annotators", min=1)] = 24,
    assignments_per_annotator: Annotated[
        int,
        typer.Option("--assignments-per-annotator", min=1),
    ] = 10,
    assignment_seed: Annotated[int, typer.Option("--assignment-seed")] = DEFAULT_ASSIGNMENT_SEED,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Export blind human-evaluation units and per-annotator assignments."""
    written = export_human_eval(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        target_units=target_units,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        assignment_seed=assignment_seed,
        overwrite=overwrite,
    )
    table = Table(title="Human evaluation export")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    table.add_row("assignments_dir", str(output_dir / "assignments"))
    console.print(table)


@app.command("reassign-human-eval")
def reassign_human_eval_command(
    package_dir: Annotated[Path, typer.Option("--package-dir")] = Path(
        "data/human_eval/tcred_release"
    ),
    annotators: Annotated[int, typer.Option("--annotators", min=1)] = 36,
    assignments_per_annotator: Annotated[
        int,
        typer.Option("--assignments-per-annotator", min=1),
    ] = 20,
    assignment_seed: Annotated[int, typer.Option("--assignment-seed")] = DEFAULT_ASSIGNMENT_SEED,
) -> None:
    """Rebuild only assignment files for an existing frozen human-evaluation sample."""
    written = reassign_human_eval_package(
        package_dir=package_dir,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        assignment_seed=assignment_seed,
    )
    table = Table(title="Human-evaluation assignments rebuilt")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("import-human-labels")
def import_human_labels_command(
    assignment_dir: Annotated[Path, typer.Option("--assignment-dir")] = Path(
        "data/human_eval/tcred_synth/assignments"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/human_eval/tcred_synth/imported"
    ),
    manifest_path: Annotated[Path | None, typer.Option("--manifest-path")] = None,
    allow_unfrozen: Annotated[
        bool,
        typer.Option("--allow-unfrozen/--require-frozen"),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Import completed labels and compute agreement, QC, and sensitivity diagnostics."""
    written = import_human_labels(
        assignment_dir=assignment_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        allow_unfrozen=allow_unfrozen,
        overwrite=overwrite,
    )
    table = Table(title="Human labels imported")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("evaluate-systems-on-human-gold")
def evaluate_systems_on_human_gold_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Evaluate the four QA systems against the published final gold judgments."""
    resolved_output = output_dir or gold_dir / "system_performance"
    written = analyze_system_performance(
        gold_dir=gold_dir,
        output_dir=resolved_output,
        overwrite=overwrite,
    )
    table = Table(title="QA-system performance on final gold")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("evaluate-current-metrics")
def evaluate_current_metrics_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    system_output_root: Annotated[Path, typer.Option("--system-output-root")] = Path(
        "data/system_outputs/tcred_release"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/current_sota/2026-08-13"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    judge_provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--judge-provider"),
    ] = "mistral",
    judge_model: Annotated[str, typer.Option("--judge-model")] = "mistral-large-2512",
    judge_concurrency: Annotated[int, typer.Option("--judge-concurrency", min=1)] = 12,
    judge_requests_per_second: Annotated[
        float, typer.Option("--judge-requests-per-second", min=0.01)
    ] = 0.06,
    judge_transport: Annotated[
        Literal["direct", "batch"], typer.Option("--judge-transport")
    ] = "direct",
    judge_scope: Annotated[
        Literal["human_gold", "all"], typer.Option("--judge-scope")
    ] = "human_gold",
    judge_stability_sample: Annotated[int, typer.Option("--judge-stability-sample", min=0)] = 40,
    minicheck_scope: Annotated[
        Literal["human_gold", "all"], typer.Option("--minicheck-scope")
    ] = "human_gold",
    alignscore_batch_size: Annotated[
        int, typer.Option("--alignscore-batch-size", min=1)
    ] = ALIGNSCORE_BATCH_SIZE,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 2000,
    skip_judge: Annotated[bool, typer.Option("--skip-judge/--run-judge")] = False,
    skip_neural: Annotated[bool, typer.Option("--skip-neural/--run-neural")] = False,
) -> None:
    """Meta-evaluate automatic metrics and score all four QA systems."""
    written = run_current_metrics(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
        output_dir=output_dir,
        metric_python=metric_python,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_concurrency=judge_concurrency,
        judge_requests_per_second=judge_requests_per_second,
        judge_transport=judge_transport,
        judge_scope=judge_scope,
        judge_stability_sample=judge_stability_sample,
        minicheck_scope=minicheck_scope,
        alignscore_batch_size=alignscore_batch_size,
        bootstrap_samples=bootstrap_samples,
        skip_judge=skip_judge,
        skip_neural=skip_neural,
    )
    table = Table(title="Current automatic metric evaluation")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("evaluate-metric-diagnostics")
def evaluate_metric_diagnostics_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/diagnostic_meta_evaluation/2026-08-15"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    source_split: Annotated[str, typer.Option("--source-split")] = "test_auto",
    pair_cap: Annotated[int, typer.Option("--pair-cap", min=10)] = 40,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 2000,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=8)] = 8,
    judge_transport: Annotated[
        Literal["direct", "batch"], typer.Option("--judge-transport")
    ] = "batch",
    requests_per_second: Annotated[
        float, typer.Option("--requests-per-second", min=0.01, max=0.24)
    ] = 0.24,
    skip_claim_judge: Annotated[bool, typer.Option("--skip-claim-judge/--run-claim-judge")] = False,
    skip_task_judge: Annotated[bool, typer.Option("--skip-task-judge/--run-task-judge")] = False,
    skip_neural: Annotated[bool, typer.Option("--skip-neural/--run-neural")] = False,
) -> None:
    """Evaluate metrics with frozen formal contrastive and invariance tests."""
    written = run_diagnostic_meta_evaluation(
        dataset_root=dataset_root,
        gold_dir=gold_dir,
        output_dir=output_dir,
        metric_python=metric_python,
        source_split=source_split,
        pair_cap_per_phenomenon=pair_cap,
        bootstrap_samples=bootstrap_samples,
        concurrency=concurrency,
        judge_transport=judge_transport,
        requests_per_second=requests_per_second,
        skip_claim_judge=skip_claim_judge,
        skip_task_judge=skip_task_judge,
        skip_neural=skip_neural,
    )
    table = Table(title="Controlled metric diagnostic meta-evaluation")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("evaluate-tcred-metric-suite")
def evaluate_tcred_metric_suite_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    baseline_dir: Annotated[Path, typer.Option("--baseline-dir")] = Path(
        "data/metrics/diagnostic_meta_evaluation/2026-08-15"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/tcred_suite/development-2026-08-15"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    source_split: Annotated[str, typer.Option("--source-split")] = "test_auto",
    pair_cap: Annotated[int, typer.Option("--pair-cap", min=10)] = 40,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 2000,
    semantic_batch_size: Annotated[int, typer.Option("--semantic-batch-size", min=1)] = 16,
) -> None:
    """Evaluate the preregistered T-CRED suite against a checksum-bound comparator run."""
    written = run_tcred_diagnostic_evaluation(
        dataset_root=dataset_root,
        gold_dir=gold_dir,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        metric_python=metric_python,
        source_split=source_split,
        pair_cap_per_phenomenon=pair_cap,
        bootstrap_samples=bootstrap_samples,
        semantic_batch_size=semantic_batch_size,
    )
    table = Table(title="T-CRED metric-suite meta-evaluation")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("prepare-source-disjoint-validation")
def prepare_source_disjoint_validation_command(
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path(
        "data/validation/tcred_v1_4_source_disjoint"
    ),
) -> None:
    """Build, audit, and checksum-lock the preregistered score-blind challenge."""
    written = prepare_source_disjoint_validation(
        repository_root=Path.cwd(),
        output_root=output_root,
    )
    _print_artifact_table("Source-disjoint validation preflight", written)


@app.command("score-source-disjoint-comparators")
def score_source_disjoint_comparators_command(
    study_root: Annotated[Path, typer.Option("--study-root")] = Path(
        "data/validation/tcred_v1_4_source_disjoint"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    judge_transport: Annotated[
        Literal["direct", "batch"], typer.Option("--judge-transport")
    ] = "batch",
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 8,
    requests_per_second: Annotated[
        float, typer.Option("--requests-per-second", min=0.01)
    ] = 0.24,
    alignscore_batch_size: Annotated[
        int, typer.Option("--alignscore-batch-size", min=1)
    ] = 32,
) -> None:
    """Run pinned local comparators and the scoped RAGChecker-style response comparator."""
    written = run_source_disjoint_comparators(
        repository_root=Path.cwd(),
        study_root=study_root,
        metric_python=metric_python,
        judge_transport=judge_transport,
        concurrency=concurrency,
        requests_per_second=requests_per_second,
        alignscore_batch_size=alignscore_batch_size,
    )
    _print_artifact_table("Source-disjoint comparators", written)


@app.command("evaluate-source-disjoint-tcred")
def evaluate_source_disjoint_tcred_command(
    study_root: Annotated[Path, typer.Option("--study-root")] = Path(
        "data/validation/tcred_v1_4_source_disjoint"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    semantic_batch_size: Annotated[int, typer.Option("--semantic-batch-size", min=1)] = 32,
) -> None:
    """Run frozen T-CRED and the 10,000-replicate preregistered inference."""
    written = run_source_disjoint_tcred_evaluation(
        repository_root=Path.cwd(),
        study_root=study_root,
        metric_python=metric_python,
        gold_dir=gold_dir,
        semantic_batch_size=semantic_batch_size,
    )
    _print_artifact_table("Source-disjoint T-CRED validation", written)


@app.command("score-opened-reserve-response-addendum")
def score_opened_reserve_response_addendum_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/tcred_suite/reserve-v1.4-response-addendum-2026-08-16"
    ),
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    judge_transport: Annotated[
        Literal["direct", "batch"], typer.Option("--judge-transport")
    ] = "batch",
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=8)] = 8,
    requests_per_second: Annotated[
        float, typer.Option("--requests-per-second", min=0.01, max=0.24)
    ] = 0.24,
) -> None:
    """Run the missing response comparator as a non-confirmatory reserve addendum."""
    written = run_reserve_response_addendum(
        repository_root=Path.cwd(),
        output_dir=output_dir,
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        judge_transport=judge_transport,
        concurrency=concurrency,
        requests_per_second=requests_per_second,
    )
    _print_artifact_table("Opened-reserve response addendum", written)


@app.command("audit-source-disjoint-labels")
def audit_source_disjoint_labels_command(
    study_root: Annotated[Path, typer.Option("--study-root")] = Path(
        "data/validation/tcred_v1_4_source_disjoint"
    ),
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
) -> None:
    """Run the immutable post-hoc intervention-label audit and sensitivity analysis."""
    written = run_source_disjoint_posthoc_audit(
        repository_root=Path.cwd(),
        study_root=study_root,
        gold_dir=gold_dir,
    )
    _print_artifact_table("Source-disjoint post-hoc label audit", written)


@app.command("evaluate-tcred-population")
def evaluate_tcred_population_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    system_output_root: Annotated[Path, typer.Option("--system-output-root")] = Path(
        "data/system_outputs/tcred_release"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/tcred_suite/population-2026-08-15"
    ),
    metric_python: Annotated[Path, typer.Option("--metric-python")] = Path(
        ".venv-metrics/Scripts/python.exe"
    ),
    current_sota_scores: Annotated[Path, typer.Option("--current-sota-scores")] = Path(
        "data/metrics/current_sota/2026-08-13/metric_scores.jsonl"
    ),
    non_llm_scores: Annotated[Path, typer.Option("--non-llm-scores")] = Path(
        "data/metrics/non_llm_expansion/2026-08-14/metric_scores.jsonl"
    ),
    task_judge_scores: Annotated[Path, typer.Option("--task-judge-scores")] = Path(
        "data/metrics/tcred_task_judge/2026-08-14/metric_scores.jsonl"
    ),
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 2000,
    semantic_batch_size: Annotated[int, typer.Option("--semantic-batch-size", min=1)] = 16,
) -> None:
    """Evaluate T-CRED on human gold and every available QA-system response."""

    written = run_tcred_population_evaluation(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
        output_dir=output_dir,
        metric_python=metric_python,
        comparator_score_paths=(
            current_sota_scores,
            non_llm_scores,
            task_judge_scores,
        ),
        bootstrap_samples=bootstrap_samples,
        semantic_batch_size=semantic_batch_size,
    )
    table = Table(title="T-CRED population evaluation")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("evaluate-task-judge")
def evaluate_task_judge_command(
    gold_dir: Annotated[Path, typer.Option("--gold-dir")] = Path(
        "data/human_eval/tcred_release/gold/2026-08-13T011632Z"
    ),
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "data/generated/tcred_release"
    ),
    system_output_root: Annotated[Path, typer.Option("--system-output-root")] = Path(
        "data/system_outputs/tcred_release"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/metrics/tcred_task_judge/2026-08-14"
    ),
    model: Annotated[str, typer.Option("--model")] = "mistral-large-2512",
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=8)] = 4,
    requests_per_second: Annotated[
        float, typer.Option("--requests-per-second", min=0.01, max=0.24)
    ] = 0.24,
    stability_sample: Annotated[int, typer.Option("--stability-sample", min=0)] = 40,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=100)] = 2000,
) -> None:
    """Calibrate and evaluate the blinded, task-matched Mistral judge."""

    written = run_task_judge_experiment(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
        output_dir=output_dir,
        model=model,
        concurrency=concurrency,
        requests_per_second=requests_per_second,
        stability_sample=stability_sample,
        bootstrap_samples=bootstrap_samples,
    )
    table = Table(title="T-CRED task-matched LLM judge")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("prepare-paraphrase-batch")
def prepare_paraphrase_batch(
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")] = Path(
        "data/generated/tcred_synth"
    ),
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--provider"),
    ] = "openai",
    model: Annotated[str, typer.Option("--model")] = "gpt-5-mini",
    include_answers: Annotated[bool, typer.Option("--include-answers/--no-answers")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Prepare provider-native batch files for LLM paraphrasing."""
    batch_dir = output_dir or dataset_dir / "llm_batches" / provider
    tasks = build_paraphrase_tasks(
        dataset_dir,
        include_questions=True,
        include_evidence=True,
        include_answers=include_answers,
        limit=limit,
    )
    written = write_provider_batch(
        tasks=tasks,
        provider=provider,
        model=model,
        output_dir=batch_dir,
        overwrite=overwrite,
    )

    table = Table(title="Paraphrase batch prepared")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("provider", provider)
    table.add_row("model", model)
    table.add_row("tasks", str(len(tasks)))
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("submit-batch-job")
def submit_batch_job(
    request_file: Annotated[Path, typer.Option("--request-file")],
    provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--provider"),
    ] = "openai",
    model: Annotated[str | None, typer.Option("--model")] = None,
    endpoint: Annotated[str | None, typer.Option("--endpoint")] = None,
    output_path: Annotated[Path | None, typer.Option("--output-path", "-o")] = None,
) -> None:
    """Submit a prepared provider batch request file."""
    client = BatchJobClient(provider=provider)
    result = asyncio.run(
        client.submit(
            request_file=request_file,
            model=model,
            endpoint=endpoint,
        )
    )
    path = output_path or request_file.parent / "batch_submission.json"
    path.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
    table = Table(title="Batch job submitted")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("provider", provider)
    table.add_row("batch_id", result.batch_id)
    table.add_row("uploaded_file_id", result.uploaded_file_id or "")
    table.add_row("status", result.status or "")
    table.add_row("submission_record", str(path))
    console.print(table)


@app.command("batch-job-status")
def batch_job_status(
    batch_id: Annotated[str, typer.Option("--batch-id")],
    provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--provider"),
    ] = "openai",
    output_path: Annotated[Path | None, typer.Option("--output-path", "-o")] = None,
) -> None:
    """Retrieve a provider batch job status."""
    client = BatchJobClient(provider=provider)
    result = asyncio.run(client.retrieve(batch_id))
    path = output_path or Path("data/cache") / f"{provider}_{batch_id}_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
    table = Table(title="Batch job status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("provider", provider)
    table.add_row("batch_id", result.batch_id)
    table.add_row("status", result.status or "")
    table.add_row("output_file_id", result.output_file_id or "")
    table.add_row("results_url", result.results_url or "")
    table.add_row("status_record", str(path))
    console.print(table)


@app.command("download-batch-results")
def download_batch_results(
    output_path: Annotated[Path, typer.Option("--output-path", "-o")],
    provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--provider"),
    ] = "openai",
    file_id: Annotated[str | None, typer.Option("--file-id")] = None,
    results_url: Annotated[str | None, typer.Option("--results-url")] = None,
) -> None:
    """Download a provider batch output file."""
    client = BatchJobClient(provider=provider)
    path = asyncio.run(
        client.download_results(
            output_path=output_path,
            file_id=file_id,
            results_url=results_url,
        )
    )
    console.print(f"[green]Downloaded batch results:[/green] {path}")


@app.command("import-paraphrase-results")
def import_paraphrase_results_command(
    task_manifest: Annotated[Path, typer.Option("--task-manifest")],
    result_file: Annotated[Path, typer.Option("--result-file")],
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")] = Path(
        "data/generated/tcred_synth"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/generated/tcred_synth_paraphrased"
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Import provider batch outputs and write a paraphrased dataset copy."""
    written = import_paraphrase_results(
        dataset_dir=dataset_dir,
        task_manifest=task_manifest,
        result_file=result_file,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    table = Table(title="Paraphrase results imported")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


def _print_artifact_table(title: str, written: dict[str, Path]) -> None:
    table = Table(title=title)
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in written.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("paraphrase-sample")
def paraphrase_sample(
    provider: Annotated[
        Literal["openai", "anthropic", "mistral", "groq"],
        typer.Option("--provider"),
    ] = "openai",
    model: Annotated[str, typer.Option("--model")] = "gpt-5-mini",
    kind: Annotated[Literal["question", "evidence"], typer.Option("--kind")] = "question",
    text: Annotated[
        str,
        typer.Option("--text"),
    ] = "Who was Director of Orion Labs on June 1, 2020?",
) -> None:
    """Run one async paraphrase request for prompt/model smoke testing."""
    client = LLMParaphraseClient(provider=provider, model=model)
    if kind == "question":
        result = asyncio.run(client.paraphrase_question(text))
    else:
        result = asyncio.run(client.paraphrase_evidence(text))

    table = Table(title="Paraphrase sample")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("provider", result.provider)
    table.add_row("model", result.model)
    table.add_row("prompt", result.prompt_name)
    table.add_row("input", result.input_text)
    table.add_row("output", result.output_text)
    console.print(table)
