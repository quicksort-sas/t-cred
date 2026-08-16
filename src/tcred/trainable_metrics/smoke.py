from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.config import TrainingConfig
from tcred.trainable_metrics.preprocessing import pretokenize_corpus
from tcred.trainable_metrics.schema import (
    CurriculumStage,
    EvidencePassage,
    SemanticRecord,
    SemanticTarget,
    SemanticTask,
    stable_unit_id,
)
from tcred.trainable_metrics.trainer import train_semantic_metric


def run_cpu_pipeline_smoke(
    *,
    workspace: Path,
    backbone_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    corpus_dir = workspace / "corpus"
    tokenized_dir = workspace / "tokenized"
    run_dir = workspace / "run"
    if workspace.exists() and any(workspace.iterdir()) and not overwrite:
        raise FileExistsError(f"Smoke workspace is not empty: {workspace}")
    records_dir = corpus_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "development", "calibration"):
        path = records_dir / f"{partition}.stage_b.jsonl"
        with path.open("wb") as handle:
            for record in _smoke_records(partition):
                handle.write(
                    orjson.dumps(
                        record.model_dump(mode="json"),
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                )
    pretokenize_corpus(
        corpus_dir=corpus_dir,
        backbone_dir=backbone_dir,
        output_dir=tokenized_dir,
        max_length=64,
        batch_size=32,
    )
    config = TrainingConfig(
        experiment_name="tcred-sl-cpu-smoke",
        backbone_revision="44acabbec0ef496f6dbc93adadea57f376b7c0ec",
        seed=42,
        final_seeds=[42],
        max_length=64,
        effective_batch_size=2,
        micro_batch_size=2,
        stage_a_epochs=0,
        stage_b_epochs=1,
        learning_rate=1e-5,
        precision="fp32",
        num_workers=0,
        checkpoint_every_steps=100,
        evaluate_every_steps=100,
        monitor_rows_per_task=100,
        forecast_after_steps=10,
    )
    return train_semantic_metric(
        config=config,
        tokenized_dir=tokenized_dir,
        backbone_dir=backbone_dir,
        output_dir=run_dir,
    )


def _smoke_records(partition: str) -> list[SemanticRecord]:
    rows: list[SemanticRecord] = []
    for index in range(2):
        positive = index == 0
        common = {
            "source_dataset": "smoke",
            "source_version": "smoke-v1",
            "source_native_split": partition,
            "source_group_id": f"{partition}:group:{index}",
            "curriculum_stage": CurriculumStage.TASK_MATCHED,
            "question": f"Who held the role in {partition} example {index}?",
            "query_time_or_interval": "January 1, 2020",
            "temporal_operator": "as_of",
            "reference_answers": ["Alice"],
            "candidate_or_claim": "Alice held the role." if positive else "Bob held the role.",
            "evidence_passages": [
                EvidencePassage(
                    evidence_id=f"{partition}:{index}:evidence",
                    text="Alice held the role on January 1, 2020.",
                )
            ],
            "label_provenance": "deterministic smoke fixture",
            "license_record": "fixture",
        }
        targets = {
            SemanticTask.ANSWER: SemanticTarget(
                answer_u1=float(positive),
                answer_u2=float(positive),
                equivalence=float(positive),
            ),
            SemanticTask.SUPPORT: SemanticTarget(
                class_distribution={"entailment" if positive else "contradiction": 1.0}
            ),
            SemanticTask.RELEVANCE: SemanticTarget(relevance=float(positive)),
            SemanticTask.TEMPORAL: SemanticTarget(
                class_distribution={"support" if positive else "contradiction": 1.0}
            ),
            SemanticTask.ANSWERABILITY: SemanticTarget(answerable=float(positive)),
            SemanticTask.CITATION: SemanticTarget(
                class_distribution={"appropriate" if positive else "inappropriate": 1.0}
            ),
        }
        for task, target in targets.items():
            native_id = f"{partition}:{task}:{index}"
            values = dict(common)
            values.update(
                {
                    "unit_id": stable_unit_id("smoke", native_id),
                    "source_native_id": native_id,
                    "task": task,
                    "target": target,
                    "transformation_family": f"smoke:{task}",
                    "citations": (
                        [f"{partition}:{index}:evidence"]
                        if task == SemanticTask.CITATION and positive
                        else []
                    ),
                }
            )
            rows.append(SemanticRecord.create(**values))
    return rows
