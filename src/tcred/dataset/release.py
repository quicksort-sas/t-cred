from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from typing import Literal
from uuid import uuid4

import orjson
from pydantic import BaseModel, ConfigDict

from tcred.dataset.audit import DatasetAuditReport, audit_dataset_dir, write_audit_report
from tcred.dataset.extracted_source import (
    ExtractedSourceDatasetGenerator,
    load_extracted_sources,
)
from tcred.dataset.generator import SyntheticDatasetGenerator
from tcred.dataset.reporting import write_generation_report
from tcred.dataset.source_grounded import SourceGroundedDatasetGenerator, load_source_subgraphs
from tcred.dataset.validate import validate_bundle
from tcred.dataset.writer import DatasetWriter
from tcred.external.converters import convert_hoh_dataset, convert_pat_dataset
from tcred.human_eval.assignments import DEFAULT_ASSIGNMENT_SEED
from tcred.human_eval.export import export_multi_dataset_human_eval


class ReleaseBuildConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    output_root: Path
    synthetic_scenarios: int = 600
    questions_per_scenario: int = 4
    seed: int = 7
    synthetic_generator: Literal["source_extracted", "source_grounded", "legacy_template"] = (
        "source_extracted"
    )
    source_subgraphs_path: Path | None = None
    pat_data_dir: Path
    pat_limit: int = 300
    hoh_input_path: Path | None = None
    hoh_limit: int = 200
    human_eval_output_dir: Path
    data_root: Path = Path("data")
    synthetic_human_units: int = 160
    pat_human_units: int = 40
    hoh_human_units: int = 40
    annotators: int = 36
    assignments_per_annotator: int = 20
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED
    overwrite: bool = False
    include_human_eval: bool = False
    clean_data: bool = False
    strict_audit: bool = True


class ReleaseDatasetSummary(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    family: str
    dataset_dir: Path
    scenario_count: int
    question_count: int
    fact_count: int
    graph_path_count: int
    context_pack_count: int
    answer_variant_count: int
    source_fidelity_counts: dict[str, int]
    audit_report: Path
    warnings: list[str]
    warning_waivers: list[str]
    audit_status: Literal["passed", "waived"]
    dataset_manifest: Path
    private_payload_sha256: str
    runtime_payload_sha256: str


class ReleaseBuildManifest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    release_version: str = "3.0"
    release_status: Literal["certified", "development"]
    created_at: datetime
    config: ReleaseBuildConfig
    implementation_sha256: str
    environment_sha256: str
    source_artifacts: dict[str, object]
    license_notes: dict[str, str]
    datasets: list[ReleaseDatasetSummary]
    totals: dict[str, int]
    human_eval: dict[str, object] | None = None


def build_dataset_release(config: ReleaseBuildConfig) -> ReleaseBuildManifest:
    _preflight_release_inputs(config)
    if config.clean_data:
        clean_generated_data(config.data_root)
    _guard_release_destinations(config)
    staging_output = _staging_path(config.output_root)
    staging_human = _staging_path(config.human_eval_output_dir)
    build_config = config.model_copy(
        update={
            "output_root": staging_output,
            "human_eval_output_dir": staging_human,
            "clean_data": False,
            "overwrite": True,
        }
    )
    try:
        manifest = _build_staged_release(
            build_config=build_config,
            published_config=config,
        )
        promotion_pairs = [(staging_output, config.output_root)]
        if config.include_human_eval:
            promotion_pairs.append((staging_human, config.human_eval_output_dir))
        _promote_roots(promotion_pairs)
    except Exception:
        for path in (staging_output, staging_human):
            if path.exists():
                rmtree(path)
        raise
    return manifest


def _build_staged_release(
    *,
    build_config: ReleaseBuildConfig,
    published_config: ReleaseBuildConfig,
) -> ReleaseBuildManifest:
    build_config.output_root.mkdir(parents=True, exist_ok=True)
    dataset_dirs = {
        "tcred_synth": build_config.output_root / "tcred_synth",
        "tcred_pat": build_config.output_root / "tcred_pat",
        "tcred_hoh": build_config.output_root / "tcred_hoh",
    }

    _write_synthetic_dataset(config=build_config, output_dir=dataset_dirs["tcred_synth"])
    convert_pat_dataset(
        pat_data_dir=build_config.pat_data_dir,
        output_dir=dataset_dirs["tcred_pat"],
        limit=build_config.pat_limit,
        overwrite=True,
    )
    convert_hoh_dataset(
        input_path=build_config.hoh_input_path,
        output_dir=dataset_dirs["tcred_hoh"],
        limit=build_config.hoh_limit,
        overwrite=True,
    )

    staged_summaries = [
        _audit_dataset(
            family=family,
            dataset_dir=dataset_dir,
            strict=build_config.strict_audit,
        )
        for family, dataset_dir in dataset_dirs.items()
    ]
    staged_human_eval = None
    if build_config.include_human_eval:
        staged_human_eval = _write_human_eval(
            config=build_config,
            dataset_dirs=dataset_dirs,
        )
        assignment_manifest = Path(str(staged_human_eval["artifacts"]["assignment_manifest"]))
        _rewrite_json_paths(
            assignment_manifest,
            replacements=(
                (build_config.output_root, published_config.output_root),
                (
                    build_config.human_eval_output_dir,
                    published_config.human_eval_output_dir,
                ),
            ),
        )

    summaries = [
        _published_summary(
            summary,
            staging_root=build_config.output_root,
            published_root=published_config.output_root,
        )
        for summary in staged_summaries
    ]
    human_eval = _replace_path_root(
        staged_human_eval,
        replacements=(
            (build_config.output_root, published_config.output_root),
            (build_config.human_eval_output_dir, published_config.human_eval_output_dir),
        ),
    )

    manifest = ReleaseBuildManifest(
        release_status="certified" if published_config.strict_audit else "development",
        created_at=datetime.now(UTC),
        config=published_config,
        implementation_sha256=_implementation_hash(),
        environment_sha256=_environment_hash(),
        source_artifacts=_source_artifacts(published_config),
        license_notes={
            "tcred_synth": (
                "Questions and interventions are generated from frozen Wikidata statements; "
                "Wikidata source data is available under CC0 and record-level provenance is "
                "retained."
            ),
            "tcred_pat": (
                "Derived from PAT-Questions; redistribution must follow the upstream license."
            ),
            "tcred_hoh": "Derived from HoH-QAs; redistribution must follow the upstream license.",
        },
        datasets=summaries,
        totals=_totals(summaries),
        human_eval=human_eval,
    )
    _write_release_manifest(
        build_config.output_root / "release_manifest.json",
        manifest,
        build_config,
    )
    return manifest


def _preflight_release_inputs(config: ReleaseBuildConfig) -> None:
    """Fail before writing release artifacts if required source inputs are missing."""
    if config.source_subgraphs_path and not config.source_subgraphs_path.exists():
        raise FileNotFoundError(f"Source subgraphs file not found: {config.source_subgraphs_path}")
    if not config.pat_data_dir.exists() or not config.pat_data_dir.is_dir():
        raise FileNotFoundError(f"PAT source directory not found: {config.pat_data_dir}")
    pat_files = list(config.pat_data_dir.glob("*/PAT-singlehop.json")) + list(
        config.pat_data_dir.glob("*/PAT-multihop.json")
    )
    if not pat_files:
        raise FileNotFoundError(
            f"PAT source directory contains no PAT snapshot JSON files: {config.pat_data_dir}"
        )
    if config.hoh_input_path is None:
        raise ValueError(
            "Release builds require --hoh-input-path so the exact source file can be hashed; "
            "live downloads are allowed only for exploratory standalone conversion"
        )
    if not config.hoh_input_path.exists():
        raise FileNotFoundError(f"HoH input file not found: {config.hoh_input_path}")
    if config.strict_audit:
        if config.synthetic_generator != "source_extracted":
            raise ValueError("Strict releases require the source-extracted synthetic generator")
        if config.questions_per_scenario != 4:
            raise ValueError(
                "Strict source-extracted releases require exactly four questions per scenario"
            )
        if config.source_subgraphs_path is None:
            raise ValueError(
                "Strict releases require --source-subgraphs with source_extracted records; "
                "use --allow-warnings only for development builds"
            )
        source_subgraphs = load_extracted_sources(config.source_subgraphs_path)
        extraction_manifest_path = config.source_subgraphs_path.with_suffix(".manifest.json")
        if not extraction_manifest_path.is_file():
            raise FileNotFoundError(
                "Strict releases require the frozen source extraction manifest: "
                f"{extraction_manifest_path}"
            )
        extraction_manifest = orjson.loads(extraction_manifest_path.read_bytes())
        expected_source_hash = str(extraction_manifest.get("source_file_sha256") or "")
        if expected_source_hash != _file_hash(config.source_subgraphs_path):
            raise ValueError("Frozen source file does not match its extraction manifest")
        if len(source_subgraphs) < config.synthetic_scenarios:
            raise ValueError(
                "Strict releases require one distinct extracted source series per scenario: "
                f"need {config.synthetic_scenarios}, found {len(source_subgraphs)}"
            )


def clean_generated_data(data_root: Path = Path("data")) -> list[Path]:
    """Remove reproducible generated/cache artifacts and preserve human work."""
    root = data_root.resolve()
    cleaned: list[Path] = []
    for name in ("generated", "cache"):
        target = (data_root / name).resolve()
        if not _is_within(target, root):
            raise ValueError(f"Refusing to clean path outside data root: {target}")
        if target.exists():
            rmtree(target)
            cleaned.append(target)
        target.mkdir(parents=True, exist_ok=True)
    return cleaned


def _guard_release_destinations(config: ReleaseBuildConfig) -> None:
    output = config.output_root.resolve()
    human = config.human_eval_output_dir.resolve()
    if _is_within(output, human) or _is_within(human, output):
        raise ValueError("Release and human-evaluation roots must not contain one another")
    destinations = [config.output_root]
    if config.include_human_eval:
        destinations.append(config.human_eval_output_dir)
    for path in destinations:
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"Release destination is not a directory: {path}")
        if path.exists() and any(path.iterdir()) and not config.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _staging_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{path.name}.staging-{uuid4().hex}"


def _promote_roots(pairs: list[tuple[Path, Path]]) -> None:
    token = uuid4().hex
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for staging, published in pairs:
            if not staging.exists():
                raise FileNotFoundError(f"Staged release root is missing: {staging}")
            published.parent.mkdir(parents=True, exist_ok=True)
            if published.exists():
                backup = published.parent / f".{published.name}.backup-{token}"
                published.replace(backup)
                backups.append((backup, published))
            staging.replace(published)
            promoted.append(published)
    except Exception:
        for published in reversed(promoted):
            if published.exists():
                rmtree(published)
        for backup, published in reversed(backups):
            if backup.exists():
                backup.replace(published)
        raise
    for backup, _ in backups:
        if backup.exists():
            rmtree(backup)


def _published_summary(
    summary: ReleaseDatasetSummary,
    *,
    staging_root: Path,
    published_root: Path,
) -> ReleaseDatasetSummary:
    relative = summary.dataset_dir.relative_to(staging_root)
    dataset_dir = published_root / relative
    return summary.model_copy(
        update={
            "dataset_dir": dataset_dir,
            "audit_report": dataset_dir / summary.audit_report.name,
            "dataset_manifest": dataset_dir / summary.dataset_manifest.name,
        }
    )


def _replace_path_root(
    value: object,
    *,
    replacements: tuple[tuple[Path, Path], ...],
) -> object:
    if isinstance(value, dict):
        return {
            key: _replace_path_root(item, replacements=replacements) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_root(item, replacements=replacements) for item in value]
    if isinstance(value, str):
        for staging, published in replacements:
            try:
                relative = Path(value).relative_to(staging)
            except ValueError:
                continue
            return str(published / relative)
    return value


def _rewrite_json_paths(
    path: Path,
    *,
    replacements: tuple[tuple[Path, Path], ...],
) -> None:
    value = orjson.loads(path.read_bytes())
    remapped = _replace_path_root(value, replacements=replacements)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(remapped, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _write_synthetic_dataset(*, config: ReleaseBuildConfig, output_dir: Path) -> None:
    if config.synthetic_generator == "source_extracted":
        if config.source_subgraphs_path is None:
            raise ValueError("Source-extracted generation requires --source-subgraphs")
        generator = ExtractedSourceDatasetGenerator(
            seed=config.seed,
            sources=load_extracted_sources(config.source_subgraphs_path),
        )
    elif config.synthetic_generator == "source_grounded":
        source_subgraphs = (
            load_source_subgraphs(config.source_subgraphs_path)
            if config.source_subgraphs_path
            else None
        )
        generator = SourceGroundedDatasetGenerator(
            seed=config.seed,
            source_subgraphs=source_subgraphs,
        )
    else:
        generator = SyntheticDatasetGenerator(seed=config.seed)
    bundle = generator.generate(
        scenario_count=config.synthetic_scenarios,
        questions_per_scenario=config.questions_per_scenario,
    )
    validate_bundle(bundle)
    DatasetWriter(output_dir=output_dir, overwrite=config.overwrite).write_bundle(bundle)
    write_generation_report(bundle=bundle, output_dir=output_dir)


def _audit_dataset(
    *,
    family: str,
    dataset_dir: Path,
    strict: bool,
) -> ReleaseDatasetSummary:
    report = audit_dataset_dir(dataset_dir)
    audit_path = write_audit_report(dataset_dir)
    waivers = _warning_waivers(family=family, warnings=report.warnings, strict=strict)
    return _summary_from_audit(
        family=family,
        dataset_dir=dataset_dir,
        audit_path=audit_path,
        report=report,
        warning_waivers=waivers,
    )


def _summary_from_audit(
    *,
    family: str,
    dataset_dir: Path,
    audit_path: Path,
    report: DatasetAuditReport,
    warning_waivers: list[str],
) -> ReleaseDatasetSummary:
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = orjson.loads(manifest_path.read_bytes())
    return ReleaseDatasetSummary(
        family=family,
        dataset_dir=dataset_dir,
        scenario_count=report.scenario_count,
        question_count=report.question_count,
        fact_count=report.fact_count,
        graph_path_count=report.graph_path_count,
        context_pack_count=report.context_pack_count,
        answer_variant_count=report.answer_variant_count,
        source_fidelity_counts=report.source_fidelity_counts,
        audit_report=audit_path,
        warnings=report.warnings,
        warning_waivers=warning_waivers,
        audit_status="waived" if report.warnings else "passed",
        dataset_manifest=manifest_path,
        private_payload_sha256=str(dataset_manifest["private_payload_sha256"]),
        runtime_payload_sha256=str(dataset_manifest["runtime_payload_sha256"]),
    )


def _write_human_eval(
    *,
    config: ReleaseBuildConfig,
    dataset_dirs: dict[str, Path],
) -> dict[str, object]:
    written = export_multi_dataset_human_eval(
        dataset_dirs=dataset_dirs,
        output_dir=config.human_eval_output_dir,
        target_units_by_family={
            "tcred_synth": config.synthetic_human_units,
            "tcred_pat": config.pat_human_units,
            "tcred_hoh": config.hoh_human_units,
        },
        annotators=config.annotators,
        assignments_per_annotator=config.assignments_per_annotator,
        assignment_seed=config.assignment_seed,
        overwrite=config.overwrite,
    )
    return {
        "output_dir": str(config.human_eval_output_dir),
        "target_units_by_family": {
            "tcred_synth": config.synthetic_human_units,
            "tcred_pat": config.pat_human_units,
            "tcred_hoh": config.hoh_human_units,
        },
        "annotators": config.annotators,
        "assignments_per_annotator": config.assignments_per_annotator,
        "assignment_seed": config.assignment_seed,
        "artifacts": {name: str(path) for name, path in written.items()},
    }


def _totals(summaries: list[ReleaseDatasetSummary]) -> dict[str, int]:
    return {
        "scenarios": sum(summary.scenario_count for summary in summaries),
        "questions": sum(summary.question_count for summary in summaries),
        "facts": sum(summary.fact_count for summary in summaries),
        "graph_paths": sum(summary.graph_path_count for summary in summaries),
        "context_packs": sum(summary.context_pack_count for summary in summaries),
        "answer_variants": sum(summary.answer_variant_count for summary in summaries),
    }


def _write_release_manifest(
    path: Path,
    manifest: ReleaseBuildManifest,
    config: ReleaseBuildConfig,
) -> None:
    if path.exists() and not config.overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(
            manifest.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
    )
    temporary.replace(path)


_EXTERNAL_COVERAGE_WARNING_PREFIXES = (
    "Missing temporal operators:",
    "Missing context pack types:",
    "No abstention-required questions found",
    "No invalid graph paths found",
    "Missing expected answer variant types:",
    "One temporal operator exceeds 35% of questions:",
)


def _warning_waivers(*, family: str, warnings: list[str], strict: bool) -> list[str]:
    if not warnings:
        return []
    if not strict:
        return [f"development build: strict audit disabled: {warning}" for warning in warnings]
    if family not in {"tcred_pat", "tcred_hoh"}:
        raise RuntimeError(f"Strict release audit failed for {family}: {warnings}")
    unexpected = [
        warning
        for warning in warnings
        if not warning.startswith(_EXTERNAL_COVERAGE_WARNING_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"Strict release audit failed for {family}: {unexpected}")
    return [
        f"source-family coverage waiver ({family} is not a synthetic schema-coverage set): "
        f"{warning}"
        for warning in warnings
    ]


def _source_artifacts(config: ReleaseBuildConfig) -> dict[str, object]:
    pat_files = sorted(config.pat_data_dir.glob("*/PAT-*.json"))
    source_subgraphs: dict[str, object]
    if config.source_subgraphs_path:
        source_subgraphs = _path_record(config.source_subgraphs_path)
        source_subgraphs["mode"] = str(config.synthetic_generator)
        if config.synthetic_generator == "source_extracted":
            source_subgraphs.update(
                {
                    "fidelity": "verified per record during strict generation",
                    "record_count": sum(
                        bool(line.strip())
                        for line in config.source_subgraphs_path.read_bytes().splitlines()
                    ),
                }
            )
        else:
            source_subgraphs["fidelity"] = "development-only normalized source patterns"
        extraction_manifest = config.source_subgraphs_path.with_suffix(".manifest.json")
        if extraction_manifest.exists():
            source_subgraphs["extraction_manifest"] = _path_record(extraction_manifest)
    else:
        source_module = Path(__file__).with_name("source_grounded.py")
        source_subgraphs = {
            "mode": "built_in_pattern_catalog",
            "fidelity": "pattern_only",
            "implementation_sha256": _file_hash(source_module),
        }
    return {
        "pat": [_path_record(path) for path in pat_files],
        "hoh": _path_record(config.hoh_input_path),
        "source_subgraphs": source_subgraphs,
    }


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    return _tree_hash(root, sorted(root.rglob("*.py")))


def _environment_hash() -> str:
    project_root = Path(__file__).resolve().parents[3]
    paths = [
        path
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.exists()
    ]
    return _tree_hash(project_root, paths)


def _tree_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_hash(path),
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
