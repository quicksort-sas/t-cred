from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.display_labels import (
    ResolvedAnswerText,
    resolve_answer_text,
)
from tcred.external_evaluations.sabet_tkgqa.evaluation import score_prediction
from tcred.external_evaluations.sabet_tkgqa.label_bundle import (
    TimeQuestionsLabelResolver,
)
from tcred.external_evaluations.sabet_tkgqa.metric_inputs import (
    build_pairwise_metric_inputs,
    score_answer_only_tcred,
)
from tcred.external_evaluations.sabet_tkgqa.schema import SabetPredictionRecord
from tcred.metrics import neural_worker as neural_worker_module
from tcred.metrics.config import (
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_REVISION,
    PEDANTS_REVISION,
    SAS_MODEL,
    SAS_REVISION,
)
from tcred.metrics.models import MetricInput
from tcred.metrics.neural_worker import (
    _NEURAL_INPUT_VERSION,
    _NEURAL_OUTPUT_VERSION,
    _NEURAL_RUNTIME_VERSION,
    _PEDANTS_ASSETS,
    _SAS_RUNTIME_FILES,
    _SAS_TOKENIZER_BACKEND,
    _TRANSFORMERS_RUNTIME_FILES,
    neural_input_hash,
)

_CANONICAL_PAIR_VERSION = "sabet-answer-pair-v1"
_EXPECTED_NEURAL_SCORES = {
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "pedants_match",
}
_NEURAL_SCORE_BOUND_TOLERANCE = 1e-5
_NEURAL_SCORE_COLUMNS = (
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "pedants_match",
)


@dataclass
class _ScoreBoundNormalization:
    corrected_component_count: int = 0
    corrected_metric_ids: set[str] = field(default_factory=set)
    lower_bound_count: int = 0
    upper_bound_count: int = 0
    by_score: Counter[str] = field(default_factory=Counter)
    maximum_absolute_adjustment: float = 0.0

    def record(
        self,
        *,
        metric_id: str,
        score_name: str,
        raw_score: float,
        bounded_score: float,
    ) -> None:
        adjustment = abs(raw_score - bounded_score)
        if adjustment == 0.0:
            return
        self.corrected_component_count += 1
        self.corrected_metric_ids.add(metric_id)
        self.by_score[score_name] += 1
        self.maximum_absolute_adjustment = max(
            self.maximum_absolute_adjustment,
            adjustment,
        )
        if bounded_score > raw_score:
            self.lower_bound_count += 1
        else:
            self.upper_bound_count += 1

    def merge(self, other: _ScoreBoundNormalization) -> None:
        self.corrected_component_count += other.corrected_component_count
        self.corrected_metric_ids.update(other.corrected_metric_ids)
        self.lower_bound_count += other.lower_bound_count
        self.upper_bound_count += other.upper_bound_count
        self.by_score.update(other.by_score)
        self.maximum_absolute_adjustment = max(
            self.maximum_absolute_adjustment,
            other.maximum_absolute_adjustment,
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "corrected_component_count": self.corrected_component_count,
            "corrected_pair_count": len(self.corrected_metric_ids),
            "lower_bound_count": self.lower_bound_count,
            "upper_bound_count": self.upper_bound_count,
            "by_score": dict(sorted(self.by_score.items())),
            "maximum_absolute_adjustment": self.maximum_absolute_adjustment,
        }


class _NeuralScoreStore:
    """Transient disk-backed lookup for large immutable neural score shards."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if self.path.exists():
            raise FileExistsError(f"Neural score index already exists: {self.path}")
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.connection.execute("PRAGMA cache_size=-262144")
        self.connection.execute(
            """
            CREATE TABLE neural_scores (
                metric_id TEXT PRIMARY KEY,
                input_sha256 TEXT NOT NULL,
                bertscore_precision REAL,
                bertscore_recall REAL,
                bertscore_f1 REAL,
                sas_cross_encoder REAL,
                pedants_probability REAL,
                pedants_match REAL,
                used INTEGER NOT NULL DEFAULT 0 CHECK (used IN (0, 1))
            ) WITHOUT ROWID
            """
        )
        self.row_count = 0
        self._closed = False

    def insert(self, rows: list[tuple[object, ...]]) -> None:
        if not rows:
            return
        try:
            self.connection.executemany(
                """
                INSERT INTO neural_scores (
                    metric_id,
                    input_sha256,
                    bertscore_precision,
                    bertscore_recall,
                    bertscore_f1,
                    sas_cross_encoder,
                    pedants_probability,
                    pedants_match
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate metric ID across neural score artifacts") from error
        self.row_count += len(rows)

    def commit(self) -> None:
        self.connection.commit()

    def get_and_mark_used(self, metric_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            """
            SELECT
                input_sha256,
                bertscore_precision,
                bertscore_recall,
                bertscore_f1,
                sas_cross_encoder,
                pedants_probability,
                pedants_match,
                used
            FROM neural_scores
            WHERE metric_id = ?
            """,
            (metric_id,),
        ).fetchone()
        if row is None:
            return None
        if row[-1] == 0:
            self.connection.execute(
                "UPDATE neural_scores SET used = 1 WHERE metric_id = ?",
                (metric_id,),
            )
        return {
            "input_sha256": row[0],
            "scores": dict(zip(_NEURAL_SCORE_COLUMNS, row[1:-1], strict=True)),
        }

    def unused_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM neural_scores WHERE used = 0"
        ).fetchone()
        assert row is not None
        return int(row[0])

    def close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True

    def __len__(self) -> int:
        return self.row_count

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def prepare_metric_inputs(
    *,
    prediction_paths: list[Path],
    output_dir: Path,
    timequestions_label_bundle_path: Path | None = None,
) -> dict[str, Path]:
    """Create de-duplicated neural inputs and deterministic unit scores."""

    if not prediction_paths:
        raise ValueError("At least one prediction export is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "metric_inputs.unique.jsonl"
    deterministic_path = output_dir / "unit_scores.deterministic.jsonl"
    seen_pairs: dict[str, tuple[str, str, str]] = {}
    seen_units: set[tuple[str, int]] = set()
    run_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    missing_reference_count = 0
    unreadable_reference_count = 0
    unreadable_candidate_count = 0
    unreadable_reference_value_count = 0
    expanded_pair_count = 0
    input_files = []
    label_sources: Counter[str] = Counter()
    resolver = _load_label_resolver(timequestions_label_bundle_path)

    with inputs_path.open("w", encoding="utf-8", newline="\n") as input_stream, (
        deterministic_path.open("w", encoding="utf-8", newline="\n")
    ) as deterministic_stream:
        for path in sorted(item.resolve() for item in prediction_paths):
            input_files.append(_file_identity(path))
            for record in _read_predictions(path):
                resolved_text = resolve_answer_text(record, resolver=resolver)
                _count_label_sources(label_sources, resolved_text)
                unreadable_reference_value_count += resolved_text.unreadable_reference_count
                unit_key = (record.run_id, record.source_index)
                if unit_key in seen_units:
                    raise ValueError(f"Duplicate prediction unit: {unit_key}")
                seen_units.add(unit_key)
                run_counts[record.run_id] += 1
                dataset_counts[record.dataset] += 1
                deterministic_stream.write(
                    score_prediction(record, resolved_text=resolved_text).model_dump_json()
                    + "\n"
                )
                pair_rows = build_pairwise_metric_inputs(
                    record,
                    resolved_text=resolved_text,
                )
                if not pair_rows:
                    missing_reference_count += int(not record.gold_answer_labels)
                    unreadable_reference_count += int(
                        bool(record.gold_answer_labels) and not resolved_text.references
                    )
                    unreadable_candidate_count += int(
                        resolved_text.candidate is None
                    )
                    continue
                expanded_pair_count += len(pair_rows)
                for row in pair_rows:
                    canonical = canonical_metric_input(row)
                    content = (
                        canonical.question,
                        canonical.reference_answer,
                        canonical.candidate_answer,
                    )
                    previous = seen_pairs.get(canonical.metric_id)
                    if previous is not None:
                        if previous != content:
                            raise ValueError(
                                f"Canonical pair hash collision: {canonical.metric_id}"
                            )
                        continue
                    seen_pairs[canonical.metric_id] = content
                    input_stream.write(canonical.model_dump_json() + "\n")

    manifest_path = output_dir / "metric_input_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "canonical_pair_version": _CANONICAL_PAIR_VERSION,
        "prediction_files": input_files,
        "unit_count": len(seen_units),
        "run_counts": dict(sorted(run_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "missing_reference_unit_count": missing_reference_count,
        "unreadable_reference_unit_count": unreadable_reference_count,
        "unreadable_reference_value_count": unreadable_reference_value_count,
        "unreadable_candidate_unit_count": unreadable_candidate_count,
        "display_label_source_counts": dict(sorted(label_sources.items())),
        "timequestions_label_bundle": resolver.identity if resolver is not None else None,
        "expanded_reference_pair_count": expanded_pair_count,
        "unique_reference_pair_count": len(seen_pairs),
        "deduplication_fraction": (
            1.0 - len(seen_pairs) / expanded_pair_count if expanded_pair_count else 0.0
        ),
        "artifacts": {
            "metric_inputs": _file_identity(inputs_path),
            "deterministic_unit_scores": _file_identity(deterministic_path),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "inputs": inputs_path,
        "deterministic_scores": deterministic_path,
    }


def finalize_metric_scores(
    *,
    prediction_paths: list[Path],
    neural_scores_path: Path | Sequence[Path],
    neural_manifest_path: Path | Sequence[Path],
    output_dir: Path,
    timequestions_label_bundle_path: Path | None = None,
) -> dict[str, Path]:
    """Reduce pairwise neural and T-CRED scores to one record per QA output."""

    if not prediction_paths:
        raise ValueError("At least one prediction export is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    neural_score_paths = _normalize_artifact_paths(
        neural_scores_path, name="neural score"
    )
    neural_manifest_paths = _normalize_artifact_paths(
        neural_manifest_path, name="neural manifest"
    )
    neural_index_path = output_dir / ".neural-score-index.sqlite3"
    (
        neural,
        neural_manifests,
        score_bound_normalization,
        metadata_identities,
    ) = _load_neural_artifacts(
        neural_score_paths,
        neural_manifest_paths,
        database_path=neural_index_path,
    )
    neural_manifest = neural_manifests[0]
    runtime_contract_sha256 = str(neural_manifest["runtime_contract_sha256"])
    scores_path = output_dir / "unit_metric_scores.jsonl"
    context_path = output_dir / "unit_metric_context.jsonl"
    unit_count = 0
    missing_reference_count = 0
    unreadable_reference_count = 0
    unreadable_candidate_count = 0
    unreadable_reference_value_count = 0
    run_counts: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    resolver = _load_label_resolver(timequestions_label_bundle_path)
    with scores_path.open("w", encoding="utf-8", newline="\n") as stream, (
        context_path.open("w", encoding="utf-8", newline="\n")
    ) as context_stream:
        for path in sorted(item.resolve() for item in prediction_paths):
            for record in _read_predictions(path):
                resolved_text = resolve_answer_text(record, resolver=resolver)
                _count_label_sources(label_sources, resolved_text)
                unreadable_reference_value_count += resolved_text.unreadable_reference_count
                context_stream.write(
                    json.dumps(
                        {
                            "run_id": record.run_id,
                            "dataset": record.dataset,
                            "model": record.model,
                            "variant": record.variant,
                            "seed": record.seed,
                            "source_index": record.source_index,
                            "qid": record.qid,
                            "question": record.question,
                            "question_type": record.question_type,
                            "answer_type": record.answer_type,
                            "gold_answer_ids": record.gold_answer_ids,
                            "gold_answer_labels": record.gold_answer_labels,
                            "resolved_gold_answer_labels": [
                                {
                                    "answer_id": item.answer_id,
                                    "label": item.text,
                                    "source": item.source,
                                    "original_reference_index": (
                                        item.original_reference_index
                                    ),
                                    "wikidata_lastrevid": item.wikidata_lastrevid,
                                    "wikidata_canonical_qid": (
                                        item.wikidata_canonical_qid
                                    ),
                                }
                                for item in resolved_text.references
                            ],
                            "predicted_answer_id": record.predicted_answer_ids[0],
                            "predicted_answer_label": record.predicted_answer_labels[0],
                            "resolved_predicted_answer": (
                                {
                                    "label": resolved_text.candidate.text,
                                    "source": resolved_text.candidate.source,
                                    "wikidata_lastrevid": (
                                        resolved_text.candidate.wikidata_lastrevid
                                    ),
                                    "wikidata_canonical_qid": (
                                        resolved_text.candidate.wikidata_canonical_qid
                                    ),
                                }
                                if resolved_text.candidate is not None
                                else None
                            ),
                            "source_record_sha256": record.source_record_sha256,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                unit = score_prediction(record, resolved_text=resolved_text)
                pair_rows = build_pairwise_metric_inputs(
                    record,
                    resolved_text=resolved_text,
                )
                if not pair_rows:
                    missing_reference_count += int(not record.gold_answer_labels)
                    unreadable_reference_count += int(
                        bool(record.gold_answer_labels) and not resolved_text.references
                    )
                    unreadable_candidate_count += int(
                        resolved_text.candidate is None
                    )
                    stream.write(unit.model_dump_json() + "\n")
                    unit_count += 1
                    run_counts[record.run_id] += 1
                    continue

                best_scores: dict[str, float | None] = {}
                best_references: dict[str, int] = {}
                for row in pair_rows:
                    reference_index = _original_reference_index(row)
                    canonical = canonical_metric_input(row)
                    neural_row = neural.get_and_mark_used(canonical.metric_id)
                    if neural_row is None:
                        raise ValueError(f"Missing neural score: {canonical.metric_id}")
                    expected_input_sha256 = neural_input_hash(canonical.model_dump())
                    if neural_row.get("input_sha256") != expected_input_sha256:
                        raise ValueError(
                            f"Neural input hash mismatch: {canonical.metric_id}"
                        )
                    pair_scores = _validated_score_dict(neural_row, canonical.metric_id)
                    tcred = score_answer_only_tcred(
                        canonical,
                        baseline_scores={
                            "pedants_probability": pair_scores.get(
                                "pedants_probability"
                            )
                        },
                    )
                    _update_per_metric_max(
                        best_scores,
                        best_references,
                        {**pair_scores, **tcred.scores},
                        reference_index=reference_index,
                    )
                unit = unit.model_copy(
                    update={
                        "scores": {**unit.scores, **best_scores},
                        "winning_reference_by_metric": {
                            **unit.winning_reference_by_metric,
                            **best_references,
                        },
                    }
                )
                stream.write(unit.model_dump_json() + "\n")
                unit_count += 1
                run_counts[record.run_id] += 1

    unused_count = neural.unused_count()
    if unused_count:
        raise ValueError(f"Neural score file has {unused_count} unused rows")
    unique_neural_pair_count = len(neural)
    neural.close()
    neural_index_path.unlink()
    manifest_path = output_dir / "metric_score_manifest.json"
    manifest = {
        "schema_version": "1.2",
        "generated_utc": datetime.now(UTC).isoformat(),
        "finalizer_implementation_sha256": _file_identity(
            Path(__file__).resolve()
        )["sha256"],
        "canonical_pair_version": _CANONICAL_PAIR_VERSION,
        "unit_count": unit_count,
        "run_counts": dict(sorted(run_counts.items())),
        "missing_reference_unit_count": missing_reference_count,
        "unreadable_reference_unit_count": unreadable_reference_count,
        "unreadable_reference_value_count": unreadable_reference_value_count,
        "unreadable_candidate_unit_count": unreadable_candidate_count,
        "display_label_source_counts": dict(sorted(label_sources.items())),
        "timequestions_label_bundle": resolver.identity if resolver is not None else None,
        "unique_neural_pair_count": unique_neural_pair_count,
        "neural_shard_count": len(neural_score_paths),
        "unique_tcred_pair_count": unique_neural_pair_count,
        "multi_reference_reduction": "per-metric maximum",
        "neural_score_lookup": {
            "backend": "transient SQLite primary-key index",
            "source_jsonl_loading": "streaming",
            "deleted_after_exact_coverage_validation": True,
        },
        "neural_inputs": [_file_identity(path) for path in neural_score_paths],
        "neural_runtime_manifests": [
            _file_identity(path) for path in neural_manifest_paths
        ],
        "neural_runtime_contract_sha256": runtime_contract_sha256,
        "neural_score_bound_normalization": score_bound_normalization,
        "neural_metadata_variants": [
            json.loads(value) for value in metadata_identities
        ],
        "unit_scores": _file_identity(scores_path),
        "unit_context": _file_identity(context_path),
        "applicability": {
            "native_hits": "all released rows",
            "reference_text_metrics": (
                "rows with at least one readable gold label and a readable top-1 candidate; "
                "supplemental TimeQuestions labels, when configured, are provenance-bound"
            ),
            "tcred": "answer-only applicable components; unsupported dimensions remain null",
        },
    }
    _write_json(manifest_path, manifest)
    return {"manifest": manifest_path, "scores": scores_path, "context": context_path}


def canonical_metric_input(row: MetricInput) -> MetricInput:
    payload = {
        "version": _CANONICAL_PAIR_VERSION,
        "question": row.question,
        "reference_answer": row.reference_answer,
        "candidate_answer": row.candidate_answer,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return row.model_copy(
        update={
            "metric_id": f"sabet-pair:{digest}",
            "system_name": None,
            "unit_id": None,
            "gold_labels": {},
            "gold_provenance": {},
        }
    )


def _update_per_metric_max(
    scores: dict[str, float | None],
    winners: dict[str, int],
    candidate: dict[str, float | None],
    *,
    reference_index: int,
) -> None:
    for name, value in candidate.items():
        if value is None:
            scores.setdefault(name, None)
            continue
        previous = scores.get(name)
        if previous is None or value > previous:
            scores[name] = value
            winners[name] = reference_index


def _original_reference_index(row: MetricInput) -> int:
    oracle = row.gold_provenance.get("external_identity_oracle")
    if not isinstance(oracle, dict):
        raise ValueError(f"Missing external identity provenance for {row.metric_id}")
    indices = oracle.get("gold_answer_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        )
    ):
        raise ValueError(f"Invalid original reference indices for {row.metric_id}")
    return min(indices)


def _load_neural_manifest(
    path: Path,
    *,
    neural_scores_path: Path,
    expected_row_count: int | None = None,
) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read neural runtime manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Neural runtime manifest must be a JSON object")
    expected_top_level = {
        "schema_version": "1.0",
        "worker_contract": _NEURAL_OUTPUT_VERSION,
        "requested_metrics": ["bertscore", "pedants", "sas"],
    }
    for name, expected in expected_top_level.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"Neural runtime manifest mismatch for {name}: "
                f"expected {expected!r}, observed {manifest.get(name)!r}"
            )
    declared_row_count = manifest.get("row_count")
    if (
        isinstance(declared_row_count, bool)
        or not isinstance(declared_row_count, int)
        or declared_row_count < 0
    ):
        raise ValueError("Neural runtime manifest has an invalid row count")
    if expected_row_count is not None and declared_row_count != expected_row_count:
        raise ValueError(
            "Neural runtime manifest mismatch for row_count: "
            f"expected {expected_row_count!r}, observed {declared_row_count!r}"
        )

    output_identity = manifest.get("output")
    if not isinstance(output_identity, dict):
        raise ValueError("Neural runtime manifest has no output identity")
    actual_output = _file_identity(neural_scores_path.resolve())
    for name in ("size_bytes", "sha256", "line_count"):
        if output_identity.get(name) != actual_output[name]:
            raise ValueError(f"Neural score artifact differs from manifest: {name}")
    if output_identity.get("line_count") != declared_row_count:
        raise ValueError("Neural manifest row count differs from its output identity")

    worker_path = Path(str(neural_worker_module.__file__)).resolve()
    expected_worker_sha256 = _file_identity(worker_path)["sha256"]
    if manifest.get("worker_implementation_sha256") != expected_worker_sha256:
        raise ValueError("Neural worker implementation differs from the scored runtime")

    runtime = manifest.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("Neural runtime contract is missing")
    runtime_sha256 = _canonical_sha256(runtime)
    if manifest.get("runtime_contract_sha256") != runtime_sha256:
        raise ValueError("Neural runtime contract hash mismatch")
    _validate_runtime_contract(runtime)

    expected_configuration = {
        "bertscore": {
            "model": BERTSCORE_MODEL,
            "revision": BERTSCORE_REVISION,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
            "allowed_snapshot_files": list(_TRANSFORMERS_RUNTIME_FILES),
        },
        "pedants": {
            "artifact_revision": PEDANTS_REVISION,
            "artifact_sha256": dict(_PEDANTS_ASSETS),
        },
        "sas": {
            "model": SAS_MODEL,
            "revision": SAS_REVISION,
            "activation": "sigmoid",
            "tokenizer_backend": _SAS_TOKENIZER_BACKEND,
            "allowed_snapshot_files": list(_SAS_RUNTIME_FILES),
        },
    }
    if manifest.get("metric_configuration") != expected_configuration:
        raise ValueError("Neural metric configuration differs from the frozen external protocol")
    return manifest


def _normalize_artifact_paths(
    value: Path | Sequence[Path], *, name: str
) -> list[Path]:
    paths = [value] if isinstance(value, Path) else list(value)
    if not paths:
        raise ValueError(f"At least one {name} artifact is required")
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError(f"Every {name} artifact must be a Path")
    return [path.resolve() for path in paths]


def _load_neural_artifacts(
    score_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    *,
    database_path: Path,
) -> tuple[
    _NeuralScoreStore,
    list[dict[str, object]],
    dict[str, object],
    list[str],
]:
    if len(score_paths) != len(manifest_paths):
        raise ValueError(
            "Neural score and manifest artifact counts differ: "
            f"{len(score_paths)} != {len(manifest_paths)}"
        )
    store = _NeuralScoreStore(database_path)
    manifests: list[dict[str, object]] = []
    aggregate_normalization = _ScoreBoundNormalization()
    shard_normalizations: list[dict[str, object]] = []
    metadata_identities: set[str] = set()
    runtime_contract_sha256: str | None = None
    try:
        for score_path, manifest_path in zip(
            score_paths,
            manifest_paths,
            strict=True,
        ):
            manifest = _load_neural_manifest(
                manifest_path,
                neural_scores_path=score_path,
            )
            current_runtime = str(manifest["runtime_contract_sha256"])
            if runtime_contract_sha256 is None:
                runtime_contract_sha256 = current_runtime
            elif current_runtime != runtime_contract_sha256:
                raise ValueError(
                    "Neural shards were produced under different runtime contracts"
                )
            normalization = _ScoreBoundNormalization()
            row_count = _ingest_neural_scores(
                score_path,
                store=store,
                manifest=manifest,
                normalization=normalization,
                metadata_identities=metadata_identities,
            )
            if row_count != manifest["row_count"]:
                raise ValueError(
                    "Neural score row count differs from its validated manifest: "
                    f"{row_count} != {manifest['row_count']}"
                )
            store.commit()
            manifests.append(manifest)
            aggregate_normalization.merge(normalization)
            output_identity = manifest["output"]
            assert isinstance(output_identity, dict)
            shard_normalizations.append(
                {
                    "neural_score_sha256": output_identity["sha256"],
                    **normalization.manifest_payload(),
                }
            )
        expected_metadata_identity_count = 1 if len(store) else 0
        if len(metadata_identities) != expected_metadata_identity_count:
            raise ValueError(
                "Neural scores must have one consistent metadata identity when present; "
                f"found {len(metadata_identities)}"
            )
    except Exception:
        store.close()
        database_path.unlink(missing_ok=True)
        raise
    return (
        store,
        manifests,
        {
            "operation": (
                "clip finite floating-point boundary excursions to each metric's "
                "theoretical score domain after raw artifact hash validation"
            ),
            "tolerance": _NEURAL_SCORE_BOUND_TOLERANCE,
            "source_artifacts_modified": False,
            "aggregate": aggregate_normalization.manifest_payload(),
            "shards": shard_normalizations,
        },
        sorted(metadata_identities),
    )


def _validate_runtime_contract(runtime: dict[str, object]) -> None:
    expected = {
        "contract_version": _NEURAL_RUNTIME_VERSION,
        "input_contract_version": _NEURAL_INPUT_VERSION,
        "requested_metrics": ["bertscore", "pedants", "sas"],
        "resolved_device": "cuda",
    }
    for name, value in expected.items():
        if runtime.get(name) != value:
            raise ValueError(f"Neural runtime contract mismatch for {name}")
    packages = runtime.get("distributions")
    required_packages = {
        "bert-score",
        "huggingface-hub",
        "joblib",
        "numpy",
        "pydantic",
        "pydantic-core",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    }
    if not isinstance(packages, dict) or any(
        not isinstance(packages.get(name), str) or not packages[name]
        for name in required_packages
    ):
        raise ValueError("Neural runtime contract has missing package versions")
    module_origins = runtime.get("module_origins")
    required_modules = {
        "bert_score",
        "huggingface_hub",
        "joblib",
        "numpy",
        "pydantic",
        "pydantic_core",
        "scipy",
        "sklearn",
        "torch",
        "transformers",
    }
    if not isinstance(module_origins, dict) or any(
        not isinstance(module_origins.get(name), str) or not module_origins[name]
        for name in required_modules
    ):
        raise ValueError("Neural runtime contract has missing module origins")
    cuda = runtime.get("cuda")
    if (
        not isinstance(cuda, dict)
        or cuda.get("available") is not True
        or not isinstance(cuda.get("selected_device"), dict)
    ):
        raise ValueError("CUDA runtime identity is missing from the neural manifest")


def _validate_neural_metadata(
    row: dict[str, object],
    *,
    runtime_contract_sha256: str,
    runtime_contract: object,
) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing neural metadata for {row.get('metric_id')}")
    expected_keys = {"runtime_contract_sha256", "bertscore", "sas", "pedants"}
    if set(metadata) != expected_keys:
        raise ValueError(f"Unexpected neural metadata fields for {row.get('metric_id')}")
    if metadata.get("runtime_contract_sha256") != runtime_contract_sha256:
        raise ValueError(f"Neural runtime identity mismatch for {row.get('metric_id')}")
    if not isinstance(runtime_contract, dict):
        raise ValueError("Malformed neural runtime contract")
    runtime_device = runtime_contract.get("resolved_device")
    expected_bertscore = {
        "model": BERTSCORE_MODEL,
        "revision": BERTSCORE_REVISION,
        "num_layers": BERTSCORE_NUM_LAYERS,
        "idf": False,
        "rescale_with_baseline": False,
        "runtime_device": runtime_device,
    }
    expected_sas = {
        "model": SAS_MODEL,
        "revision": SAS_REVISION,
        "activation": "sigmoid",
        "tokenizer_backend": _SAS_TOKENIZER_BACKEND,
        "allowed_snapshot_files": list(_SAS_RUNTIME_FILES),
        "runtime_device": runtime_device,
    }
    if metadata.get("bertscore") != expected_bertscore:
        raise ValueError(f"BERTScore metadata mismatch for {row.get('metric_id')}")
    if metadata.get("sas") != expected_sas:
        raise ValueError(f"SAS metadata mismatch for {row.get('metric_id')}")
    pedants = metadata.get("pedants")
    packages = runtime_contract.get("distributions")
    if not isinstance(pedants, dict) or not isinstance(packages, dict):
        raise ValueError(f"PEDANTS metadata is missing for {row.get('metric_id')}")
    expected_pedants = {
        "artifact_revision": PEDANTS_REVISION,
        "artifact_sha256": dict(_PEDANTS_ASSETS),
        "scikit_learn_runtime": packages.get("scikit-learn"),
    }
    for name, expected in expected_pedants.items():
        if pedants.get(name) != expected:
            raise ValueError(f"PEDANTS metadata mismatch for {row.get('metric_id')}: {name}")
    boolean_fields = {
        "legacy_loss_unpickle_shim_required",
        "legacy_loss_unpickle_shim_installed_during_run",
    }
    if set(pedants) != set(expected_pedants) | boolean_fields or any(
        not isinstance(pedants.get(name), bool) for name in boolean_fields
    ):
        raise ValueError(f"Unexpected PEDANTS metadata fields for {row.get('metric_id')}")


def _ingest_neural_scores(
    path: Path,
    *,
    store: _NeuralScoreStore,
    manifest: dict[str, object],
    normalization: _ScoreBoundNormalization,
    metadata_identities: set[str],
) -> int:
    runtime_contract_sha256 = str(manifest["runtime_contract_sha256"])
    runtime_contract = manifest["runtime_contract"]
    pending: list[tuple[object, ...]] = []
    row_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid neural score JSON at {path}:{line_number}"
                ) from error
            metric_id = row.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id:
                raise ValueError(f"Neural score line {line_number} has no metric_id")
            if row.get("schema_version") != _NEURAL_OUTPUT_VERSION:
                raise ValueError(f"Neural score schema mismatch for {metric_id}")
            input_sha256 = row.get("input_sha256")
            if not (
                isinstance(input_sha256, str)
                and len(input_sha256) == 64
                and all(character in "0123456789abcdef" for character in input_sha256)
            ):
                raise ValueError(f"Invalid neural input hash for {metric_id}")
            scores = _validated_score_dict(
                row,
                metric_id,
                normalization=normalization,
            )
            _validate_neural_metadata(
                row,
                runtime_contract_sha256=runtime_contract_sha256,
                runtime_contract=runtime_contract,
            )
            metadata_identities.add(
                json.dumps(
                    row["metadata"],
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            pending.append(
                (
                    metric_id,
                    input_sha256,
                    *(scores[name] for name in _NEURAL_SCORE_COLUMNS),
                )
            )
            row_count += 1
            if len(pending) == 4096:
                store.insert(pending)
                pending = []
    store.insert(pending)
    return row_count


def _validated_score_dict(
    row: dict[str, object],
    metric_id: str,
    *,
    normalization: _ScoreBoundNormalization | None = None,
) -> dict[str, float | None]:
    values = row.get("scores")
    if not isinstance(values, dict):
        raise ValueError(f"Malformed neural scores for {metric_id}")
    output: dict[str, float | None] = {}
    for name, value in values.items():
        if not isinstance(name, str):
            raise ValueError(f"Non-string metric name for {metric_id}")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"Non-numeric score for {metric_id}: {name}")
        if value is None:
            output[name] = None
            continue
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for {metric_id}: {name}")
        lower_bound = -1.0 if name.startswith("bertscore_") else 0.0
        if (
            score < lower_bound - _NEURAL_SCORE_BOUND_TOLERANCE
            or score > 1.0 + _NEURAL_SCORE_BOUND_TOLERANCE
        ):
            raise ValueError(f"Out-of-range score for {metric_id}: {name}={score}")
        bounded_score = min(max(score, lower_bound), 1.0)
        if normalization is not None:
            normalization.record(
                metric_id=metric_id,
                score_name=name,
                raw_score=score,
                bounded_score=bounded_score,
            )
        output[name] = bounded_score
    if set(output) != _EXPECTED_NEURAL_SCORES:
        missing = sorted(_EXPECTED_NEURAL_SCORES - set(output))
        unexpected = sorted(set(output) - _EXPECTED_NEURAL_SCORES)
        raise ValueError(
            f"Neural metric schema mismatch for {metric_id}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return output


def _load_label_resolver(
    path: Path | None,
) -> TimeQuestionsLabelResolver | None:
    return TimeQuestionsLabelResolver(path) if path is not None else None


def _count_label_sources(
    counts: Counter[str],
    resolved: ResolvedAnswerText,
) -> None:
    if resolved.candidate is None:
        counts["candidate:unresolved"] += 1
    else:
        counts[f"candidate:{resolved.candidate.source}"] += 1
    if resolved.unreadable_reference_count:
        counts["reference:unresolved"] += resolved.unreadable_reference_count
    for reference in resolved.references:
        counts[f"reference:{reference.source}"] += 1


def _read_predictions(path: Path) -> Iterator[SabetPredictionRecord]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield SabetPredictionRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid prediction at {path}:{line_number}: {error}") from error


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            line_count += block.count(b"\n")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "line_count": line_count,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _paths(values: Iterable[str]) -> list[Path]:
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SABET answer-compatible metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--predictions", nargs="+", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--timequestions-label-bundle", type=Path)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--predictions", nargs="+", required=True)
    finalize.add_argument("--neural-scores", type=Path, nargs="+", required=True)
    finalize.add_argument("--neural-manifest", type=Path, nargs="+", required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--timequestions-label-bundle", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        outputs = prepare_metric_inputs(
            prediction_paths=_paths(args.predictions),
            output_dir=args.output_dir,
            timequestions_label_bundle_path=args.timequestions_label_bundle,
        )
    else:
        outputs = finalize_metric_scores(
            prediction_paths=_paths(args.predictions),
            neural_scores_path=args.neural_scores,
            neural_manifest_path=args.neural_manifest,
            output_dir=args.output_dir,
            timequestions_label_bundle_path=args.timequestions_label_bundle,
        )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
