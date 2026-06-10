from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import orjson
from pydantic import BaseModel

from tcred.dataset.models import DATASET_SCHEMA_VERSION, DatasetBundle


class DatasetWriter:
    """Write one complete dataset release atomically.

    The private research artifacts remain at the dataset root. ``runtime/`` is
    a deliberately smaller projection that QA systems can read without loading
    gold answers, symbolic programs, fact roles, or controlled answer variants.
    """

    _PRIVATE_FILES = {
        "scenarios": "scenarios.jsonl",
        "entities": "entities.jsonl",
        "facts": "facts.jsonl",
        "snapshots": "snapshots.jsonl",
        "questions": "questions.jsonl",
        "graph_paths": "graph_paths.jsonl",
        "context_packs": "context_packs.jsonl",
        "answer_variants": "answer_variants.jsonl",
    }

    def __init__(self, output_dir: Path, *, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite

    def write_bundle(self, bundle: DatasetBundle) -> dict[str, Path]:
        self._guard_output()
        parent = self.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{self.output_dir.name}.staging-{uuid4().hex}"
        staging.mkdir()
        try:
            self._write_private(staging, bundle)
            self._write_runtime(staging / "runtime", bundle)
            self._write_manifest(staging, bundle)
            self._commit(staging)
        except Exception:
            if staging.exists():
                rmtree(staging)
            raise

        artifacts = {
            name: self.output_dir / relative for name, relative in self._PRIVATE_FILES.items()
        }
        artifacts["splits"] = self.output_dir / "splits.json"
        artifacts["dataset_manifest"] = self.output_dir / "dataset_manifest.json"
        artifacts.update(
            {
                "runtime_entities": self.output_dir / "runtime" / "entities.jsonl",
                "runtime_facts": self.output_dir / "runtime" / "facts.jsonl",
                "runtime_questions": self.output_dir / "runtime" / "questions.jsonl",
            }
        )
        return artifacts

    def _write_private(self, root: Path, bundle: DatasetBundle) -> None:
        for name, relative in self._PRIVATE_FILES.items():
            self._write_models(root / relative, getattr(bundle, name))
        (root / "splits.json").write_bytes(
            orjson.dumps(bundle.splits, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    def _write_runtime(self, root: Path, bundle: DatasetBundle) -> None:
        root.mkdir()
        self._write_rows(
            root / "entities.jsonl",
            [
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "aliases": entity.aliases,
                }
                for entity in bundle.entities
            ],
        )
        self._write_rows(
            root / "facts.jsonl",
            [
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "fact_id": fact.fact_id,
                    "subject_id": fact.subject_id,
                    "relation": fact.relation,
                    "object_id": fact.object_id,
                    "context_id": fact.context_id,
                    "answer_entity_id": fact.answer_entity_id,
                    "graph_source_id": fact.graph_source_id,
                    "graph_target_id": fact.graph_target_id,
                    "source_relation_id": fact.source_relation_id,
                    "source_relation_label": fact.source_relation_label,
                    "relation_direction": fact.relation_direction,
                    "source_record_id": fact.source_record_id,
                    "source_revision": fact.source_revision,
                    "valid_time": fact.valid_time.model_dump(mode="json"),
                    "publication_time": fact.publication_time,
                    "transaction_time": fact.transaction_time,
                    "snapshot_visible_from": fact.snapshot_visible_from,
                    "source_type": "source_record",
                    "canonical_evidence": fact.canonical_evidence,
                    "paraphrased_evidence": fact.paraphrased_evidence,
                }
                for fact in bundle.facts
            ],
        )
        scenario_splits: dict[str, list[str]] = {}
        for split, scenario_ids in bundle.splits.items():
            for scenario_id in scenario_ids:
                scenario_splits.setdefault(scenario_id, []).append(split)
        self._write_rows(
            root / "questions.jsonl",
            [
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "qid": question.qid,
                    "scenario_id": question.scenario_id,
                    "dataset_family": question.dataset_family,
                    "question": question.question,
                    "snapshot_id": question.program.snapshot_id,
                    "temporal_basis": question.program.temporal_basis,
                    "splits": sorted(scenario_splits.get(question.scenario_id, [])),
                }
                for question in bundle.questions
            ],
        )

    def _write_manifest(self, root: Path, bundle: DatasetBundle) -> None:
        private_paths = [root / relative for relative in self._PRIVATE_FILES.values()]
        private_paths.append(root / "splits.json")
        runtime_paths = sorted((root / "runtime").glob("*.jsonl"))
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_counts": {
                "scenarios": len(bundle.scenarios),
                "entities": len(bundle.entities),
                "facts": len(bundle.facts),
                "snapshots": len(bundle.snapshots),
                "questions": len(bundle.questions),
                "graph_paths": len(bundle.graph_paths),
                "context_packs": len(bundle.context_packs),
                "answer_variants": len(bundle.answer_variants),
            },
            "private_artifacts": self._artifact_hashes(root, private_paths),
            "runtime_artifacts": self._artifact_hashes(root, runtime_paths),
            "private_payload_sha256": _combined_hash(root, private_paths),
            "runtime_payload_sha256": _combined_hash(root, runtime_paths),
            "runtime_boundary": (
                "The runtime projection excludes gold answers, question programs except the "
                "requested snapshot, fact roles, gold paths, context packs, and answer variants."
            ),
        }
        (root / "dataset_manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    @staticmethod
    def _artifact_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
        return {path.relative_to(root).as_posix(): _file_hash(path) for path in sorted(paths)}

    @staticmethod
    def _write_models(path: Path, rows: Iterable[BaseModel]) -> None:
        DatasetWriter._write_rows(
            path,
            (row.model_dump(mode="json") for row in rows),
        )

    @staticmethod
    def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
        with path.open("wb") as handle:
            for row in rows:
                handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))

    def _guard_output(self) -> None:
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(f"Dataset output is not a directory: {self.output_dir}")
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not self.overwrite:
            raise FileExistsError(
                f"{self.output_dir} already contains data; pass --overwrite to replace it"
            )

    def _commit(self, staging: Path) -> None:
        backup = self.output_dir.parent / f".{self.output_dir.name}.backup-{uuid4().hex}"
        had_existing = self.output_dir.exists()
        if had_existing:
            self.output_dir.replace(backup)
        try:
            staging.replace(self.output_dir)
        except Exception:
            if had_existing and backup.exists():
                backup.replace(self.output_dir)
            raise
        if backup.exists():
            rmtree(backup)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
