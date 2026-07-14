from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict

from tcred.dataset.models import Question
from tcred.qa.corpus import RuntimeQuestion, runtime_snapshot_id
from tcred.qa.generation import PROMPT_VERSION
from tcred.qa.models import QARunConfig, QASystemName

_BEHAVIOR_MODULES = (
    "corpus.py",
    "embeddings.py",
    "generation.py",
    "graph_retrieval.py",
    "lexical.py",
    "models.py",
    "pipeline.py",
    "retrieval.py",
    "temporal.py",
)


class CheckpointMetadata(BaseModel):
    """Compatibility and integrity record for one family/system output shard."""

    model_config = ConfigDict(use_enum_values=True)

    schema_version: str = "qa_checkpoint_v1"
    dataset_family: str
    system_name: QASystemName
    dataset_sha256: str
    question_set_sha256: str
    implementation_sha256: str
    configuration_sha256: str
    prompt_version: str
    record_count: int = 0
    output_sha256: str = ""

    def compatibility_payload(self) -> dict[str, object]:
        return self.model_dump(exclude={"record_count", "output_sha256"}, mode="json")


def build_checkpoint_metadata(
    *,
    config: QARunConfig,
    family: str,
    system_name: QASystemName,
    dataset_sha256: str,
    questions: list[Question | RuntimeQuestion],
) -> CheckpointMetadata:
    return CheckpointMetadata(
        dataset_family=family,
        system_name=system_name,
        dataset_sha256=dataset_sha256,
        question_set_sha256=_json_sha256(
            [
                {
                    "qid": question.qid,
                    "scenario_id": question.scenario_id,
                    "question": question.question,
                    "snapshot_id": runtime_snapshot_id(question),
                }
                for question in questions
            ]
        ),
        implementation_sha256=qa_implementation_sha256(),
        configuration_sha256=_json_sha256(_behavioral_config(config)),
        prompt_version=PROMPT_VERSION,
    )


def checkpoint_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".meta.json")


def read_checkpoint_metadata(output_path: Path) -> CheckpointMetadata | None:
    path = checkpoint_metadata_path(output_path)
    if not path.exists():
        return None
    return CheckpointMetadata.model_validate(orjson.loads(path.read_bytes()))


def write_checkpoint_metadata(
    output_path: Path,
    metadata: CheckpointMetadata,
    *,
    record_count: int,
) -> CheckpointMetadata:
    completed = metadata.model_copy(
        update={
            "record_count": record_count,
            "output_sha256": file_sha256(output_path) if output_path.exists() else "",
        }
    )
    path = checkpoint_metadata_path(output_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(completed.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    temporary.replace(path)
    return completed


def assert_checkpoint_compatible(
    actual: CheckpointMetadata,
    expected: CheckpointMetadata,
    *,
    output_path: Path,
) -> None:
    if actual.compatibility_payload() != expected.compatibility_payload():
        raise ValueError(
            f"Checkpoint provenance does not match the current run: {output_path}. "
            "Use --overwrite to regenerate this shard."
        )


def checkpoint_integrity_matches(
    metadata: CheckpointMetadata,
    output_path: Path,
    *,
    record_count: int,
) -> bool:
    return (
        output_path.exists()
        and metadata.record_count == record_count
        and bool(metadata.output_sha256)
        and metadata.output_sha256 == file_sha256(output_path)
    )


def qa_implementation_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in _BEHAVIOR_MODULES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _behavioral_config(config: QARunConfig) -> dict[str, object]:
    return {
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "generator_provider": config.generator_provider,
        "generator_model": config.generator_model,
        "reasoning_effort": config.reasoning_effort,
        "top_k": config.top_k,
        "candidate_k": config.candidate_k,
        "graph_seed_k": config.graph_seed_k,
        "graph_max_hops": config.graph_max_hops,
        "graph_path_limit": config.graph_path_limit,
        "splits": config.splits,
        "seed": config.seed,
    }


def _json_sha256(value: object) -> str:
    payload = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()
