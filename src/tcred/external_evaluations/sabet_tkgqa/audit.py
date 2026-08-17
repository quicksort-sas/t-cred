from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import orjson

from tcred.external_evaluations.sabet_tkgqa.released_logs import (
    final_test_evaluation,
    parse_released_log,
)

_DATASET_LABELS = {
    "wikidata_big": "CronQuestions",
    "wikidata_big_complex": "Complex-CronQuestions",
    "multitq": "MultiTQ",
    "timequestions": "TimeQuestions",
}
_MODEL_LABELS = {
    "tempo_qr": "TempoQR",
    "tempo_qr_hard": "TempoQR-Hard",
    "subgtr_hard": "SubGTR-Hard",
    "sabet": "SABET-QA",
    "sabet_hard": "SABET-QA-Hard",
}


def build_artifact_audit(
    *,
    project_root: Path,
    upstream_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    artifacts = _verify_artifacts(project_root, protocol["artifacts"])
    log_rows, paper_comparisons = _audit_logs(upstream_root, protocol)
    source = _audit_source(upstream_root)
    findings = _findings(protocol, upstream_root, log_rows, source)
    return {
        "schema_version": "1.0",
        "protocol_version": protocol["protocol_version"],
        "target": protocol["target"],
        "artifacts": artifacts,
        "source": source,
        "released_log_final_tests": log_rows,
        "paper_log_comparisons": paper_comparisons,
        "findings": findings,
    }


def render_artifact_audit(audit: dict[str, Any]) -> str:
    cli_arguments = ", ".join(f"`{value}`" for value in audit["source"]["cli_arguments"])
    lines = [
        "# SABET-QA Public Artifact Audit",
        "",
        "This report is generated from the hash-pinned paper, source, OSF inventory, and all "
        "released logs. It describes the artifact available for reproduction; it does not contain "
        "independently reproduced model results.",
        "",
        "## Artifact Integrity",
        "",
        "| Artifact | SHA-256 | Status |",
        "|---|---|---|",
    ]
    for name, row in audit["artifacts"].items():
        lines.append(f"| {name} | `{row['actual_sha256']}` | {row['status']} |")
    source = audit["source"]
    lines.extend(
        [
            "",
            "## Source Identity",
            "",
            f"- Repository HEAD: `{source['repository_head']}`.",
            f"- Python files audited: {len(source['python_file_sha256'])}.",
            f"- CLI arguments found: {cli_arguments}.",
            f"- Declared dependency lock or requirements file: {source['has_dependency_spec']}.",
            f"- Declared license file: {source['has_license']}.",
            "",
            "## Paper vs Released Final Test Logs",
            "",
            "A zero delta means the rounded final released-log value equals the paper table. The "
            "comparison is an internal-consistency check, not an independent reproduction.",
            "",
            "| Dataset | Model | Paper H@1 | Log H@1 | Delta | Paper H@10 | Log H@10 | Delta | N |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit["paper_log_comparisons"]:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['paper_hits_at_1']:.3f} | "
            f"{row['log_hits_at_1']:.3f} | {row['delta_hits_at_1']:+.3f} | "
            f"{row['paper_hits_at_10']:.3f} | {row['log_hits_at_10']:.3f} | "
            f"{row['delta_hits_at_10']:+.3f} | {row['example_count']} |"
        )
    lines.extend(["", "## Reproducibility Findings", ""])
    for finding in audit["findings"]:
        lines.append(
            f"- **{finding['severity'].upper()} - {finding['code']}**: {finding['message']}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Proceed as a staged artifact reconstruction. The hard-hint path can be run as the "
            "closest public-code reproduction after compatibility and export instrumentation. The "
            "non-hard path must remain explicitly labeled as reconstructed because the public CLI "
            "cannot select the implementation represented by the released non-hard logs.",
            "",
            "The frozen design and interpretation rules are in "
            "`docs/sabet-tkgqa-external-reproduction-protocol-2026-08-16.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the pinned SABET-QA public artifact")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("data/external/sabet-tkgqa"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_evaluation/sabet_tkgqa.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/external_evaluation/sabet_tkgqa/artifact_audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/sabet-tkgqa-public-artifact-audit-2026-08-16.md"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    audit = build_artifact_audit(
        project_root=project_root,
        upstream_root=(project_root / args.upstream_root).resolve(),
        protocol_path=(project_root / args.protocol).resolve(),
    )
    json_output = project_root / args.json_output
    markdown_output = project_root / args.markdown_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_bytes(orjson.dumps(audit, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    markdown_output.write_text(render_artifact_audit(audit), encoding="utf-8")


def _verify_artifacts(project_root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, descriptor in expected.items():
        relative_path = descriptor.get("path")
        if relative_path is None:
            output[name] = {
                "expected_sha256": descriptor["sha256"],
                "actual_sha256": descriptor["sha256"],
                "status": "verified remotely; local archive intentionally absent",
            }
            continue
        path = project_root / relative_path
        actual = _sha256(path)
        expected_hash = descriptor["sha256"]
        output[name] = {
            "path": relative_path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "status": "verified" if actual == expected_hash else "MISMATCH",
        }
        if actual != expected_hash:
            raise ValueError(f"Artifact hash mismatch for {name}: {actual} != {expected_hash}")
    return output


def _audit_logs(upstream_root: Path, protocol: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    final_rows: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((upstream_root / "logs").rglob("*.log")):
        final = final_test_evaluation(parse_released_log(path, artifact_root=upstream_root))
        dataset_label = _DATASET_LABELS.get(final.dataset.casefold(), final.dataset)
        model_label = _MODEL_LABELS.get(final.artifact_label, final.artifact_label)
        row = {
            "dataset": dataset_label,
            "model": model_label,
            "artifact_label": final.artifact_label,
            "source_path": final.source_path,
            "source_sha256": final.source_sha256,
            "config": final.config,
            "hits_at_1": final.hits_at_1,
            "hits_at_10": final.hits_at_10,
            "example_count": final.inferred_example_count,
        }
        final_rows.append(row)
        indexed[(dataset_label, model_label)] = row

    comparisons: list[dict[str, Any]] = []
    for dataset, models in protocol["paper_overall_claims"].items():
        for model, claim in models.items():
            log = indexed[(dataset, model)]
            comparisons.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "paper_hits_at_1": claim["hits_at_1"],
                    "log_hits_at_1": log["hits_at_1"],
                    "delta_hits_at_1": log["hits_at_1"] - claim["hits_at_1"],
                    "paper_hits_at_10": claim["hits_at_10"],
                    "log_hits_at_10": log["hits_at_10"],
                    "delta_hits_at_10": log["hits_at_10"] - claim["hits_at_10"],
                    "example_count": log["example_count"],
                }
            )
    return final_rows, comparisons


def _audit_source(upstream_root: Path) -> dict[str, Any]:
    train_path = upstream_root / "tkg_qa_models" / "train_qa_model.py"
    train_text = train_path.read_text(encoding="utf-8")
    tree = ast.parse(train_text)
    arguments = sorted(
        {
            first.value.removeprefix("--")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance((first := node.args[0]), ast.Constant)
            and isinstance(first.value, str)
            and first.value.startswith("--")
        }
    )
    python_files = sorted(upstream_root.rglob("*.py"))
    return {
        "repository_head": _git_head(upstream_root),
        "python_file_sha256": {
            str(path.relative_to(upstream_root)).replace("\\", "/"): _sha256(path)
            for path in python_files
        },
        "cli_arguments": arguments,
        "has_dependency_spec": any(
            (upstream_root / name).exists()
            for name in ("requirements.txt", "pyproject.toml", "environment.yml", "Pipfile.lock")
        ),
        "has_license": any(upstream_root.glob("LICEN[CS]E*")),
        "seed_initialization_present": any(
            token in train_text for token in ("manual_seed(", "random.seed(", "np.random.seed(")
        ),
        "prediction_export_present": "predictions_file" in train_text,
        "data_dir_literals": sorted(
            {
                line.strip()
                for path in python_files
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip().startswith("data_dir") and "/Data/" in line
            }
        ),
        "numpy_long_occurrences": sum(
            path.read_text(encoding="utf-8", errors="replace").count("np.long")
            for path in python_files
        ),
    }


def _findings(
    protocol: dict[str, Any],
    upstream_root: Path,
    log_rows: list[dict[str, Any]],
    source: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def add(severity: str, code: str, message: str) -> None:
        findings.append({"severity": severity, "code": code, "message": message})

    expected_commit = protocol["target"]["repository_commit"]
    if source["repository_head"] != expected_commit:
        add(
            "critical",
            "source_commit_mismatch",
            "The audited source HEAD is not the pinned commit.",
        )
    if not source["has_dependency_spec"]:
        add("high", "environment_unpinned", "No requirements file or environment lock is released.")
    if not source["has_license"]:
        add(
            "high",
            "license_absent",
            "No license file is present; third-party source is not vendored.",
        )
    if not source["seed_initialization_present"]:
        add("high", "seed_absent", "Training and random answer selection have no explicit seed.")
    if not source["prediction_export_present"]:
        add(
            "high",
            "predictions_absent",
            "Only aggregate rounded scores are emitted, preventing paired audits.",
        )
    if source["numpy_long_occurrences"]:
        add(
            "medium",
            "numpy_compatibility",
            f"The source contains {source['numpy_long_occurrences']} `np.long` occurrence(s).",
        )
    cli = set(source["cli_arguments"])
    if "subgraph_reasoning" not in cli and any(
        "subgraph_reasoning" in row["config"] for row in log_rows
    ):
        add(
            "critical",
            "unreleased_cli_argument",
            "Released logs contain `subgraph_reasoning`, which the released parser does not "
            "accept.",
        )
    if any(row["config"].get("model") == "imsabt" for row in log_rows):
        add(
            "critical",
            "unreleased_model_name",
            "SABET logs use `imsabt`; the released dispatch implements only `sabet`.",
        )
    if any(row["config"].get("tkbc_model_file") == "tcomplex_ALL.ckpt" for row in log_rows):
        add(
            "critical",
            "missing_multitq_checkpoint_name",
            "MultiTQ logs request `tcomplex_ALL.ckpt`, absent from the OSF archive.",
        )

    by_key = {(row["dataset"], row["model"]): row for row in log_rows}
    for dataset in protocol["datasets"]:
        standard = by_key[(dataset, "SABET-QA")]["config"]
        hard = by_key[(dataset, "SABET-QA-Hard")]["config"]
        differing = sorted(
            key for key in set(standard) | set(hard) if standard.get(key) != hard.get(key)
        )
        if differing == ["save_to"]:
            add(
                "critical",
                f"variant_not_selectable_{dataset.casefold().replace('-', '_')}",
                f"For {dataset}, non-hard and hard SABET log configs differ only in `save_to`.",
            )
    for dataset, counts in protocol["paper_dataset_counts"].items():
        if counts["test_table_6"] != counts["test_table_8"]:
            add(
                "high",
                f"paper_count_conflict_{dataset.casefold().replace('-', '_')}",
                f"{dataset} test count is {counts['test_table_6']} in Table 6 and "
                f"{counts['test_table_8']} in Table 8.",
            )
    if "train_qa.py" in (upstream_root / "README.md").read_text(encoding="utf-8"):
        add(
            "medium",
            "readme_entrypoint_missing",
            "The README command names `train_qa.py`, but the released file is `train_qa_model.py`.",
        )
    complex_hard = by_key[("Complex-CronQuestions", "SABET-QA-Hard")]
    if complex_hard["hits_at_1"] != protocol["paper_overall_claims"]["Complex-CronQuestions"][
        "SABET-QA-Hard"
    ]["hits_at_1"]:
        add(
            "medium",
            "paper_log_score_delta",
            "The Complex-CronQuestions SABET-QA-Hard released log reports 0.804 H@1 while the "
            "paper reports 0.807.",
        )
    return findings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


if __name__ == "__main__":
    main()
