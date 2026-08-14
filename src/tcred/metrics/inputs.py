from __future__ import annotations

import hashlib
from pathlib import Path

import orjson

from tcred.dataset.io import load_bundle, read_jsonl
from tcred.metrics.deterministic import ranked_retrieval_scores, set_precision_recall
from tcred.metrics.models import EvidenceText, MetricInput
from tcred.qa.corpus import RuntimeCorpus, dataset_content_hash
from tcred.qa.models import SystemOutput


def load_metric_inputs(
    *,
    gold_dir: Path,
    dataset_root: Path,
    system_output_root: Path,
) -> list[MetricInput]:
    rows = load_human_gold_inputs(gold_dir)
    rows.extend(load_full_system_inputs(dataset_root, system_output_root))
    identities = [row.metric_id for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Metric inputs contain duplicate metric IDs")
    return rows


def load_human_gold_inputs(gold_dir: Path) -> list[MetricInput]:
    path = gold_dir / "gold_units.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing human-gold units: {path}")
    manifest_path = gold_dir / "gold_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing human-gold manifest: {manifest_path}")
    manifest = orjson.loads(manifest_path.read_bytes())
    described = {str(item["path"]): str(item["sha256"]) for item in manifest.get("files", [])}
    if described.get(path.name) != _sha256(path):
        raise ValueError("Human-gold units do not match gold_manifest.json")
    rows: list[MetricInput] = []
    for raw in read_jsonl(path):
        unit = _mapping(raw.get("unit"), name="unit")
        metadata = _mapping(raw.get("metadata"), name="metadata")
        rows.append(
            MetricInput(
                metric_id=f"gold:{raw['unit_id']}",
                population="human_gold",
                dataset_family=str(metadata["dataset_family"]),
                source_kind=str(metadata["source_kind"]),
                system_name=_optional_text(metadata.get("system_name")),
                unit_id=str(raw["unit_id"]),
                qid=str(metadata["qid"]),
                scenario_id=str(metadata["scenario_id"]),
                question=str(unit["question"]),
                reference_answer=str(unit["reference_answer"]),
                candidate_answer=str(unit["answer_text"]),
                retrieved_evidence=_visible_evidence(unit.get("retrieved_evidence")),
                cited_evidence=_visible_evidence(unit.get("cited_evidence")),
                gold_labels={str(key): str(value) for key, value in raw["gold_labels"].items()},
                gold_provenance={
                    str(field): _mapping(value, name=f"field_provenance.{field}")
                    for field, value in _mapping(
                        raw.get("field_provenance"), name="field_provenance"
                    ).items()
                },
            )
        )
    return rows


def load_full_system_inputs(dataset_root: Path, system_output_root: Path) -> list[MetricInput]:
    manifest_path = system_output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing QA run manifest: {manifest_path}")
    manifest = orjson.loads(manifest_path.read_bytes())
    for family, expected_hash in manifest.get("dataset_hashes", {}).items():
        actual_hash = dataset_content_hash(dataset_root / family)
        if actual_hash != expected_hash:
            raise ValueError(f"Dataset runtime hash changed since the QA run: {family}")
    rows: list[MetricInput] = []
    family_state: dict[str, tuple[object, RuntimeCorpus, dict[str, object]]] = {}

    for summary in manifest.get("summaries", []):
        family = str(summary["family"])
        if family not in family_state:
            bundle = load_bundle(dataset_root / family)
            family_state[family] = (
                bundle,
                RuntimeCorpus(bundle),
                {question.qid: question for question in bundle.questions},
            )
        bundle, corpus, question_by_id = family_state[family]
        del bundle
        output_path = Path(str(summary["output_path"]))
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        expected_output_hash = str(summary.get("output_sha256") or "")
        if expected_output_hash and _sha256(output_path) != expected_output_hash:
            raise ValueError(f"QA output hash does not match the run manifest: {output_path}")
        for raw in read_jsonl(output_path):
            output = SystemOutput.model_validate(raw)
            if output.status != "success":
                continue
            question = question_by_id.get(output.qid)
            if question is None:
                raise ValueError(f"QA output references an unknown question: {output.qid}")
            rows.append(_full_system_input(output, question=question, corpus=corpus))
    return rows


def write_metric_inputs(rows: list[MetricInput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")


def _full_system_input(
    output: SystemOutput, *, question: object, corpus: RuntimeCorpus
) -> MetricInput:
    required_keys = {
        corpus.semantic_fact_key(fact_id) for fact_id in question.required_valid_evidence_ids
    }
    relevance: list[bool] = []
    credited: set[tuple[str, ...]] = set()
    retrieved_evidence: list[EvidenceText] = []
    for hit in output.retrieval.hits:
        key = corpus.semantic_fact_key(hit.fact_id)
        relevant = key in required_keys and key not in credited
        relevance.append(relevant)
        if relevant:
            credited.add(key)
        retrieved_evidence.append(
            EvidenceText(
                evidence_id=hit.fact_id,
                text=corpus.prompt_view(hit.fact_id).evidence_text,
            )
        )

    cited_ids = list(dict.fromkeys(output.resolved_cited_evidence_ids))
    cited_keys = {corpus.semantic_fact_key(fact_id) for fact_id in cited_ids}
    retrieval_metrics = ranked_retrieval_scores(
        relevance=relevance,
        required_count=len(required_keys),
        k=10,
    )
    citation_metrics = set_precision_recall(
        predicted=cited_keys,
        required=required_keys,
        prefix="required_citation",
    )
    citation_metrics["citation_resolution_rate"] = (
        len(cited_ids) / len(output.cited_evidence_ids) if output.cited_evidence_ids else None
    )
    return MetricInput(
        metric_id=f"full:{output.dataset_family}:{output.system_name}:{output.qid}",
        population="system_full",
        dataset_family=output.dataset_family,
        source_kind="system_output",
        system_name=str(output.system_name),
        qid=output.qid,
        scenario_id=output.scenario_id,
        question=question.question,
        reference_answer=_reference_answer(question),
        candidate_answer=output.answer_text,
        retrieved_evidence=retrieved_evidence,
        cited_evidence=[
            EvidenceText(
                evidence_id=fact_id,
                text=corpus.prompt_view(fact_id).evidence_text,
            )
            for fact_id in cited_ids
        ],
        retrieval_metrics=retrieval_metrics,
        citation_metrics=citation_metrics,
        unresolved_citation_count=len(output.unresolved_citation_ids),
    )


def _reference_answer(question: object) -> str:
    if question.should_abstain:
        return "No answer is supported by the available evidence at the requested time."
    return ", ".join(question.gold_answer_text)


def _visible_evidence(value: object) -> list[EvidenceText]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Visible evidence must be a list")
    rows: list[EvidenceText] = []
    for item in value:
        mapping = _mapping(item, name="evidence")
        rows.append(
            EvidenceText(evidence_id=str(mapping["evidence_id"]), text=str(mapping["text"]))
        )
    return rows


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
