from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson

from tcred.dataset.source_disjoint_validation import (
    PROTOCOL_ID,
    SOURCE_SPLIT,
    verify_implementation_lock,
)
from tcred.metrics.diagnostic_models import (
    DiagnosticCase,
    DiagnosticPair,
    DiagnosticSuite,
    diagnostic_inference_cluster_ids,
)
from tcred.qa.corpus import dataset_content_hash

DEFAULT_STUDY_ROOT = Path("data/validation/tcred_v1_4_source_disjoint")
DEFAULT_PROTOCOL_PATH = Path("docs/protocols/tcred-v1.4-source-disjoint-validation-v1.json")


def load_prepared_suite(
    *,
    repository_root: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
) -> tuple[DiagnosticSuite, dict[str, Any]]:
    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    verify_implementation_lock(
        repository_root=repository_root,
        lock_path=study_root / "implementation_lock.json",
    )
    return _load_frozen_suite_contents(
        repository_root=repository_root,
        study_root=study_root,
    )


def load_frozen_suite_for_retrospective_scoring(
    *,
    repository_root: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
) -> tuple[DiagnosticSuite, dict[str, Any], dict[str, Any]]:
    """Load immutable challenge artifacts while reporting later source-code drift.

    Construction and preflight continue to use :func:`load_prepared_suite`, whose
    implementation lock is strict. Retrospective scorers do not execute the historical
    generator, so later generator changes must not make an otherwise byte-identical frozen
    challenge unusable. This loader therefore fails on any frozen protocol/data drift and
    reports, but never hides, drift in construction-only source files.
    """

    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    integrity = audit_frozen_suite_integrity(
        repository_root=repository_root,
        study_root=study_root,
    )
    suite, protocol = _load_frozen_suite_contents(
        repository_root=repository_root,
        study_root=study_root,
    )
    return suite, protocol, integrity


def audit_frozen_suite_integrity(
    *,
    repository_root: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
) -> dict[str, Any]:
    """Verify every locked non-code artifact and inventory construction-code drift."""

    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    lock_path = study_root / "implementation_lock.json"
    lock = read_json(lock_path)
    if lock.get("status") != "score_blind_preflight_passed_and_locked":
        raise ValueError("Validation implementation lock is not complete")

    frozen_checked: list[dict[str, Any]] = []
    implementation_checked: list[dict[str, Any]] = []
    frozen_errors: list[dict[str, Any]] = []
    implementation_drift: list[dict[str, Any]] = []
    records = lock.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Validation implementation lock has no file inventory")
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("Validation implementation lock contains an invalid record")
        declared_path = str(raw_record.get("path", ""))
        path = Path(declared_path)
        if not path.is_absolute():
            path = repository_root / path
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = sha256(path) if exists else None
        matches = (
            exists
            and actual_bytes == raw_record.get("bytes")
            and actual_sha256 == raw_record.get("sha256")
        )
        result = {
            "path": declared_path.replace("\\", "/"),
            "status": "match" if matches else "drift",
            "expected_bytes": raw_record.get("bytes"),
            "actual_bytes": actual_bytes,
            "expected_sha256": raw_record.get("sha256"),
            "actual_sha256": actual_sha256,
        }
        if result["path"].startswith("src/"):
            implementation_checked.append(result)
            if not matches:
                implementation_drift.append(result)
        else:
            frozen_checked.append(result)
            if not matches:
                frozen_errors.append(result)

    if frozen_errors:
        paths = [str(record["path"]) for record in frozen_errors]
        raise ValueError(f"Frozen source-disjoint artifacts changed after lock: {paths}")
    return {
        "schema_version": "tcred-frozen-suite-integrity-v1",
        "status": (
            "passed_with_reported_implementation_drift" if implementation_drift else "passed"
        ),
        "implementation_lock_sha256": sha256(lock_path),
        "frozen_artifacts_checked": len(frozen_checked),
        "frozen_artifact_drift": frozen_errors,
        "implementation_files_checked": len(implementation_checked),
        "implementation_drift": implementation_drift,
        "interpretation": (
            "All immutable protocol, source-sample, dataset-manifest, challenge, and preflight "
            "artifacts match the historical lock. Reported source-code drift is permitted only "
            "for retrospective scoring because no construction code is executed."
        ),
    }


def _load_frozen_suite_contents(
    *,
    repository_root: Path,
    study_root: Path,
) -> tuple[DiagnosticSuite, dict[str, Any]]:
    preflight = read_json(study_root / "preflight_audit.json")
    if preflight.get("status") != "pass" or preflight.get("score_blind") is not True:
        raise ValueError("Source-disjoint validation preflight is not a score-blind pass")
    protocol = load_protocol(repository_root=repository_root)
    challenge = mapping(protocol, "challenge_generation")
    cases = [
        DiagnosticCase.model_validate(row)
        for row in read_jsonl(study_root / "challenge" / "diagnostic_cases.jsonl")
    ]
    pairs = [
        DiagnosticPair.model_validate(row)
        for row in read_jsonl(study_root / "challenge" / "diagnostic_pairs.jsonl")
    ]
    dataset_root = study_root / "dataset"
    hashes = {
        path.name: dataset_content_hash(path)
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
    }
    diagnostic_audit = preflight.get("diagnostic_suite")
    if not isinstance(diagnostic_audit, dict):
        raise ValueError("Preflight has no diagnostic-suite audit")
    suite = DiagnosticSuite(
        seed=int(challenge["diagnostic_seed"]),
        source_split=SOURCE_SPLIT,
        pair_cap_per_phenomenon=int(challenge["pair_cap_per_phenomenon"]),
        dataset_content_hashes=hashes,
        cases=cases,
        pairs=pairs,
        audit={
            "case_count": len(cases),
            "pair_count": len(pairs),
            "question_clusters": len(
                {(case.metric_input.dataset_family, case.metric_input.qid) for case in cases}
            ),
            "source_scenarios": len(
                {
                    (case.metric_input.dataset_family, case.metric_input.scenario_id)
                    for case in cases
                }
            ),
            "inference_clusters": len(set(diagnostic_inference_cluster_ids(cases, pairs).values())),
            "pair_counts_by_test_type": _counter(pairs, "test_type"),
            "pair_counts_by_construct": _counter(pairs, "target_construct"),
            "pair_counts_by_phenomenon": _counter(pairs, "phenomenon"),
            "pair_counts_by_dataset": _counter(pairs, "dataset_family"),
            "selection_policy": (
                "Checksum-locked source-disjoint challenge after score-blind semantic "
                "pair-isolation filtering; no metric score informed construction."
            ),
            "score_blind_pair_filter": diagnostic_audit.get("score_blind_pair_filter", {}),
        },
    )
    if len(cases) != int(diagnostic_audit.get("cases", -1)):
        raise ValueError("Challenge case count no longer matches the preflight")
    if len(pairs) != int(diagnostic_audit.get("pairs", -1)):
        raise ValueError("Challenge pair count no longer matches the preflight")
    return suite, protocol


def load_protocol(*, repository_root: Path) -> dict[str, Any]:
    protocol = read_json(repository_root / DEFAULT_PROTOCOL_PATH)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected source-disjoint validation protocol")
    return protocol


def challenge_hashes(study_root: Path) -> dict[str, str]:
    root = study_root / "challenge"
    return {path.name: sha256(path) for path in sorted(root.iterdir()) if path.is_file()}


def read_json(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Protocol field {key!r} must be an object")
    return result


def _counter(rows: list[object], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
