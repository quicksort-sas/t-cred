from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import orjson

from tcred.dataset.io import load_bundle, read_jsonl
from tcred.dataset.models import DatasetBundle
from tcred.human_eval.augmented import system_public_unit
from tcred.human_eval.presentation import ANNOTATION_TEXT_REPAIR_VERSION
from tcred.metrics.inputs import load_metric_inputs
from tcred.metrics.models import MetricInput
from tcred.metrics.task_judge_models import JudgeSplit, TaskJudgeInput
from tcred.qa.models import SystemOutput

DEFAULT_SPLIT_SEED = 20260814
DEFAULT_CALIBRATION_FRACTION = 0.40


def load_task_judge_inputs(
    *,
    gold_dir: Path,
    dataset_root: Path,
    system_output_root: Path,
) -> list[TaskJudgeInput]:
    """Load the same blinded response cards used by humans and align them to metric IDs."""

    metric_rows = load_metric_inputs(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
    )
    base_by_id = {row.metric_id: row for row in metric_rows}
    task_rows: list[TaskJudgeInput] = []

    for raw in read_jsonl(gold_dir / "gold_units.jsonl"):
        metric_id = f"gold:{raw['unit_id']}"
        task_rows.append(
            _task_input(
                metric_id=metric_id,
                public_unit=_mapping(raw.get("unit"), name="gold unit"),
                base=base_by_id[metric_id],
            )
        )

    run_manifest = orjson.loads((system_output_root / "run_manifest.json").read_bytes())
    bundles: dict[str, DatasetBundle] = {}
    for summary in run_manifest.get("summaries", []):
        family = str(summary["family"])
        if family not in bundles:
            bundles[family] = load_bundle(dataset_root / family)
        bundle = bundles[family]
        output_path = Path(str(summary["output_path"]))
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        for raw in read_jsonl(output_path):
            output = SystemOutput.model_validate(raw)
            if output.status != "success":
                continue
            metric_id = f"full:{output.dataset_family}:{output.system_name}:{output.qid}"
            base = base_by_id[metric_id]
            if output.answer_text != base.candidate_answer:
                raise ValueError(f"QA-output answer drift for {metric_id}")
            task_rows.append(
                _task_input(
                    metric_id=metric_id,
                    public_unit=system_public_unit(bundle, output),
                    base=base,
                )
            )

    expected = set(base_by_id)
    actual = {row.metric_id for row in task_rows}
    if len(actual) != len(task_rows):
        raise ValueError("Task-judge inputs contain duplicate metric IDs")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Task-judge input mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return sorted(task_rows, key=lambda row: row.metric_id)


def write_task_judge_inputs(rows: list[TaskJudgeInput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def load_or_create_split(
    rows: list[TaskJudgeInput],
    path: Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
) -> JudgeSplit:
    human = [row for row in rows if row.population == "human_gold"]
    gold_hash = task_input_hash(human)
    if path.is_file():
        split = JudgeSplit.model_validate_json(path.read_bytes())
        _validate_split(split, human=human, gold_hash=gold_hash)
        return split
    split = _build_split(
        human,
        seed=seed,
        calibration_fraction=calibration_fraction,
        gold_hash=gold_hash,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        orjson.dumps(
            split.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
    )
    temporary.replace(path)
    return split


def task_input_hash(rows: list[TaskJudgeInput]) -> str:
    payload = [row.model_dump(mode="json") for row in sorted(rows, key=lambda item: item.metric_id)]
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _task_input(
    *, metric_id: str, public_unit: dict[str, object], base: MetricInput
) -> TaskJudgeInput:
    public_question = str(public_unit.get("question", ""))
    public_candidate = str(public_unit.get("answer_text", ""))
    public_reference = str(public_unit.get("reference_answer", ""))
    if base.population == "human_gold" and public_candidate != base.candidate_answer:
        raise ValueError(f"Candidate answer drift for {metric_id}")
    if base.population == "human_gold" and public_reference != base.reference_answer:
        raise ValueError(f"Reference answer drift for {metric_id}")
    changed_fields = [
        field
        for field, public_value, source_value in (
            ("question", public_question, base.question),
            ("reference_answer", public_reference, base.reference_answer),
            ("candidate_answer", public_candidate, base.candidate_answer),
        )
        if public_value != source_value
    ]
    return TaskJudgeInput(
        metric_id=metric_id,
        population=base.population,
        dataset_family=base.dataset_family,
        source_kind=base.source_kind,
        system_name=base.system_name,
        unit_id=base.unit_id,
        qid=base.qid,
        scenario_id=base.scenario_id,
        question=public_question,
        reference_answer=public_reference,
        candidate_answer=public_candidate,
        cited_evidence_ids=[str(value) for value in public_unit.get("cited_evidence_ids", [])],
        cited_evidence=public_unit.get("cited_evidence", []),
        retrieved_evidence=public_unit.get("retrieved_evidence", []),
        graph_paths=public_unit.get("graph_paths", []),
        context_note=str(public_unit.get("context_note", "")),
        applicable_fields=[str(value) for value in public_unit.get("applicable_fields", [])],
        gold_labels=base.gold_labels,
        gold_provenance=base.gold_provenance,
        source_question_sha256=_text_sha256(base.question),
        source_reference_answer_sha256=_text_sha256(base.reference_answer),
        source_candidate_answer_sha256=_text_sha256(base.candidate_answer),
        presentation_changed_fields=changed_fields,
        presentation_contract=(
            "human_eval_blind_public_payload_v1+annotation_text_repair_"
            f"{ANNOTATION_TEXT_REPAIR_VERSION}"
        ),
    )


def _build_split(
    human: list[TaskJudgeInput],
    *,
    seed: int,
    calibration_fraction: float,
    gold_hash: str,
) -> JudgeSplit:
    if not 0.2 <= calibration_fraction <= 0.8:
        raise ValueError("calibration_fraction must be between 0.2 and 0.8")
    groups: defaultdict[str, list[TaskJudgeInput]] = defaultdict(list)
    for row in human:
        groups[f"{row.dataset_family}::{row.qid}"].append(row)
    group_ids = sorted(groups)
    target_rows = round(len(human) * calibration_fraction)
    feature_totals = Counter(feature for row in human for feature in _balance_features(row))
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...]] | None = None
    for _ in range(50_000):
        selected = tuple(group_id for group_id in group_ids if rng.random() < calibration_fraction)
        if not selected or len(selected) == len(group_ids):
            continue
        calibration = [row for group_id in selected for row in groups[group_id]]
        score = ((len(calibration) - target_rows) / max(target_rows, 1)) ** 2 * 20
        selected_features = Counter(
            feature for row in calibration for feature in _balance_features(row)
        )
        for feature, total in feature_totals.items():
            expected = total * calibration_fraction
            score += ((selected_features[feature] - expected) / max(expected, 1.0)) ** 2
            if total >= 2 and selected_features[feature] in {0, total}:
                score += 25
        candidate = (score, selected)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("Unable to construct grouped judge split")
    calibration_groups = set(best[1])
    calibration = sorted(
        row.metric_id for group_id in calibration_groups for row in groups[group_id]
    )
    held_out = sorted(row.metric_id for row in human if row.metric_id not in set(calibration))
    split = JudgeSplit(
        seed=seed,
        target_calibration_fraction=calibration_fraction,
        gold_input_sha256=gold_hash,
        calibration_metric_ids=calibration,
        held_out_metric_ids=held_out,
        balance_summary={
            "calibration": _split_summary(
                [row for row in human if row.metric_id in set(calibration)]
            ),
            "held_out": _split_summary([row for row in human if row.metric_id in set(held_out)]),
            "optimization_score": best[0],
            "grouping_unit": "dataset_family + qid",
        },
    )
    _validate_split(split, human=human, gold_hash=gold_hash)
    return split


def _balance_features(row: TaskJudgeInput) -> tuple[str, ...]:
    features = [
        f"dataset={row.dataset_family}",
        f"source={row.source_kind}",
        f"system={row.system_name or 'controlled'}",
    ]
    for field, label in sorted(row.gold_labels.items()):
        features.extend([f"field={field}", f"label={field}:{label}"])
    return tuple(features)


def _split_summary(rows: list[TaskJudgeInput]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "question_clusters": len({(row.dataset_family, row.qid) for row in rows}),
        "datasets": dict(sorted(Counter(row.dataset_family for row in rows).items())),
        "source_kinds": dict(sorted(Counter(row.source_kind for row in rows).items())),
        "systems": dict(sorted(Counter(row.system_name or "controlled" for row in rows).items())),
        "field_labels": {
            field: dict(
                sorted(
                    Counter(
                        row.gold_labels[field] for row in rows if field in row.gold_labels
                    ).items()
                )
            )
            for field in sorted({field for row in rows for field in row.gold_labels})
        },
    }


def _validate_split(split: JudgeSplit, *, human: list[TaskJudgeInput], gold_hash: str) -> None:
    if split.gold_input_sha256 != gold_hash:
        raise ValueError("Task-judge split belongs to different human-gold inputs")
    expected = {row.metric_id for row in human}
    calibration = set(split.calibration_metric_ids)
    held_out = set(split.held_out_metric_ids)
    if calibration & held_out:
        raise ValueError("Task-judge calibration and held-out IDs overlap")
    if calibration | held_out != expected:
        raise ValueError("Task-judge split does not cover the human-gold inputs exactly")
    cluster_side: dict[tuple[str, str], str] = {}
    for row in human:
        side = "calibration" if row.metric_id in calibration else "held_out"
        cluster = (row.dataset_family, row.qid)
        previous = cluster_side.setdefault(cluster, side)
        if previous != side:
            raise ValueError(f"Question cluster crosses judge split: {cluster}")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
