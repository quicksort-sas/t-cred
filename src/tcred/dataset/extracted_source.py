from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcred.dataset.generator import SyntheticDatasetGenerator, stable_opaque_id
from tcred.dataset.intervals import human_interval, point
from tcred.dataset.models import (
    AnswerType,
    DatasetBundle,
    DatasetFamily,
    Entity,
    EntityType,
    EvalDifficulty,
    Fact,
    FactRole,
    Question,
    QuestionProgram,
    Relation,
    Scenario,
    Snapshot,
    SourceProvenance,
    SystemDifficulty,
    TemporalInterval,
    TemporalOperator,
)
from tcred.dataset.solver import GoldSolver
from tcred.dataset.splits import stable_group_splits
from tcred.dataset.text import normalize_visible_text

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_USER_AGENT = "T-CRED-research/3.0 (temporal benchmark source extraction)"
SOURCE_SAMPLING_SALT = "tcred-temporal-series-v1"
# One salted 1/16 hash bucket is an efficient, reproducible global sample. The
# source is frozen immediately, so later endpoint ordering cannot change a release.
SOURCE_HASH_BUCKETS = ("0",)
ENTITY_CACHE_CHECKPOINT_IDS = 500
GREGORIAN_CALENDAR_ID = "Q1985727"
LOGGER = logging.getLogger(__name__)
WIKIDATA_API_MAXLAG_SECONDS = 600


class _RetryableSourceResponse(RuntimeError):
    """A successful HTTP response whose source API payload asks us to retry."""


@dataclass(frozen=True)
class WikidataPropertySpec:
    property_id: str
    domain: str
    relation: Relation
    relation_label: str
    answer_noun: str
    point_template: str
    evidence_template: str


WIKIDATA_PROPERTY_SPECS: tuple[WikidataPropertySpec, ...] = (
    WikidataPropertySpec(
        property_id="P6",
        domain="government_leadership",
        relation=Relation.HELD_ROLE,
        relation_label="head of government",
        answer_noun="head of government",
        point_template="Who was the head of government of {context} on {date}?",
        evidence_template=("{answer} served as the head of government of {context} {interval}."),
    ),
    WikidataPropertySpec(
        property_id="P35",
        domain="heads_of_state",
        relation=Relation.HELD_ROLE,
        relation_label="head of state",
        answer_noun="head of state",
        point_template="Who was the head of state of {context} on {date}?",
        evidence_template="{answer} served as the head of state of {context} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P39",
        domain="positions_held",
        relation=Relation.HELD_ROLE,
        relation_label="position held",
        answer_noun="position",
        point_template="Which position did {context} hold on {date}?",
        evidence_template="{context} held the position {answer} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P54",
        domain="sports_memberships",
        relation=Relation.MEMBER_OF,
        relation_label="member of sports team",
        answer_noun="sports team",
        point_template="Which sports team was {context} a member of on {date}?",
        evidence_template="{context} was a member of {answer} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P169",
        domain="executive_leadership",
        relation=Relation.HELD_ROLE,
        relation_label="chief executive officer",
        answer_noun="chief executive officer",
        point_template="Who was the chief executive officer of {context} on {date}?",
        evidence_template=(
            "{answer} served as the chief executive officer of {context} {interval}."
        ),
    ),
    WikidataPropertySpec(
        property_id="P286",
        domain="sports_coaching",
        relation=Relation.HELD_ROLE,
        relation_label="head coach",
        answer_noun="head coach",
        point_template="Who was the head coach of {context} on {date}?",
        evidence_template="{answer} served as the head coach of {context} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P488",
        domain="organizational_leadership",
        relation=Relation.HELD_ROLE,
        relation_label="chairperson",
        answer_noun="chairperson",
        point_template="Who was the chairperson of {context} on {date}?",
        evidence_template="{answer} served as the chairperson of {context} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P108",
        domain="employment_history",
        relation=Relation.EMPLOYED_BY,
        relation_label="employed by",
        answer_noun="employer",
        point_template="Which organization employed {context} on {date}?",
        evidence_template="{context} was employed by {answer} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P102",
        domain="political_affiliations",
        relation=Relation.POLITICAL_AFFILIATION,
        relation_label="member of political party",
        answer_noun="political party",
        point_template="Which political party was {context} affiliated with on {date}?",
        evidence_template="{context} was affiliated with {answer} {interval}.",
    ),
    WikidataPropertySpec(
        property_id="P463",
        domain="organization_memberships",
        relation=Relation.MEMBER_OF,
        relation_label="member of organization",
        answer_noun="organization",
        point_template="Which organization was {context} a member of on {date}?",
        evidence_template="{context} was a member of {answer} {interval}.",
    ),
)

_SPEC_BY_PROPERTY = {spec.property_id: spec for spec in WIKIDATA_PROPERTY_SPECS}


class ExtractedTemporalClaim(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    statement_id: str
    answer_source_id: str
    answer_label: str
    start: date
    end: date | None = None
    start_precision: Literal["day", "month", "year"]
    end_precision: Literal["day", "month", "year"] | None = None

    @model_validator(mode="after")
    def valid_interval(self) -> ExtractedTemporalClaim:
        if self.end is not None and self.end < self.start:
            raise ValueError("Extracted claim ends before it starts")
        return self

    def interval(self) -> TemporalInterval:
        return TemporalInterval(
            type="open_interval" if self.end is None else "interval",
            start=self.start,
            end=self.end,
            granularity=self.start_precision,
        )


class ExtractedTemporalSource(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    source_id: str
    source_family: Literal["wikidata"] = "wikidata"
    source_fidelity: Literal["source_extracted"] = "source_extracted"
    source_revision: str
    extraction_time: datetime
    property_id: str
    property_label: str
    domain: str
    relation: Relation
    answer_noun: str
    context_source_id: str
    context_label: str
    context_revision_id: int
    claims: list[ExtractedTemporalClaim] = Field(min_length=3)
    topology_signature: str = "temporal_property_series"

    @model_validator(mode="after")
    def visible_answer_labels_are_unambiguous(self) -> ExtractedTemporalSource:
        if not _labels_are_unambiguous(self.claims):
            raise ValueError("Distinct answer entities share the same visible English label")
        if not _has_answer_contrast(self.claims):
            raise ValueError(
                "Temporal source must contain at least two distinct answer entities"
            )
        return self

    @property
    def source_record_ids(self) -> list[str]:
        return [claim.statement_id for claim in self.claims]


def load_extracted_sources(path: Path) -> list[ExtractedTemporalSource]:
    rows = [
        ExtractedTemporalSource.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Extracted source file is empty: {path}")
    source_ids = [row.source_id for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"Extracted source file contains duplicate source IDs: {path}")
    return rows


def extract_wikidata_temporal_sources(
    *,
    output_path: Path,
    target: int = 700,
    candidate_rows_per_property: int = 5000,
    cache_dir: Path | None = None,
    sampling_salt: str = SOURCE_SAMPLING_SALT,
    hash_buckets: tuple[str, ...] = SOURCE_HASH_BUCKETS,
    excluded_source_ids: set[str] | None = None,
    excluded_entity_ids: set[str] | None = None,
) -> list[ExtractedTemporalSource]:
    """Create a frozen, provenance-complete temporal source catalog."""
    if target < 1:
        raise ValueError("target must be positive")
    _validate_sampling_contract(sampling_salt=sampling_salt, hash_buckets=hash_buckets)
    forbidden_sources = excluded_source_ids or set()
    forbidden_entities = excluded_entity_ids or set()
    exclusion_counts: Counter[str] = Counter()
    cache_root = cache_dir or output_path.parent / ".extraction_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    extraction_time = datetime.now(UTC)
    with httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"User-Agent": WIKIDATA_USER_AGENT},
        follow_redirects=True,
    ) as client:
        candidate_cache_path = cache_root / "wikidata_temporal_candidates.json"
        candidate_cache_key = {
            "properties": [spec.property_id for spec in WIKIDATA_PROPERTY_SPECS],
            "row_limit": candidate_rows_per_property,
            "salt": sampling_salt,
            "buckets": list(hash_buckets),
        }
        cached_candidates = _load_candidate_cache(
            candidate_cache_path,
            expected_key=candidate_cache_key,
        )
        candidates_by_property = cached_candidates or {}
        for spec in WIKIDATA_PROPERTY_SPECS:
            if spec.property_id in candidates_by_property:
                LOGGER.info("Reusing Wikidata candidates for %s", spec.property_id)
                continue
            LOGGER.info("Sampling Wikidata candidates for %s", spec.property_id)
            candidates_by_property[spec.property_id] = _candidate_context_ids(
                client,
                property_id=spec.property_id,
                row_limit=candidate_rows_per_property,
                sampling_salt=sampling_salt,
                hash_buckets=hash_buckets,
            )
            _write_json_atomic(
                candidate_cache_path,
                {"cache_key": candidate_cache_key, "candidates": candidates_by_property},
            )
        if cached_candidates is not None and len(cached_candidates) == len(WIKIDATA_PROPERTY_SPECS):
            LOGGER.info("Reusing frozen Wikidata candidate cache %s", candidate_cache_path)
        candidates_by_property = {
            property_id: [
                context_id
                for context_id in context_ids
                if not _candidate_is_forbidden(
                    context_id,
                    property_id=property_id,
                    forbidden_sources=forbidden_sources,
                    forbidden_entities=forbidden_entities,
                    exclusion_counts=exclusion_counts,
                )
            ]
            for property_id, context_ids in candidates_by_property.items()
        }
        entity_ids = sorted({item for values in candidates_by_property.values() for item in values})
        entities = _fetch_entities(
            client,
            entity_ids,
            props="labels|claims|info",
            batch_size=50,
            cache_path=cache_root / "wikidata_temporal_context_entities.json",
        )
        LOGGER.info("Loaded %d context entities", len(entities))
        parsed: list[tuple[WikidataPropertySpec, str, dict[str, Any], list[dict[str, Any]]]] = []
        answer_ids: set[str] = set()
        for spec in WIKIDATA_PROPERTY_SPECS:
            for context_id in candidates_by_property[spec.property_id]:
                entity = entities.get(context_id)
                if not entity or "en" not in entity.get("labels", {}):
                    continue
                claims = _parse_entity_claims(entity, spec.property_id)
                if len(claims) < 3:
                    continue
                parsed.append((spec, context_id, entity, claims))
                answer_ids.update(str(claim["answer_source_id"]) for claim in claims)

        labels = _fetch_entities(
            client,
            sorted(answer_ids),
            props="labels",
            batch_size=50,
            cache_path=cache_root / "wikidata_temporal_answer_entities.json",
        )
        LOGGER.info("Loaded %d answer-label entities", len(labels))

    by_property: defaultdict[str, list[ExtractedTemporalSource]] = defaultdict(list)
    for spec, context_id, entity, claim_rows in parsed:
        context_label = _english_label(entity)
        claims: list[ExtractedTemporalClaim] = []
        for row in claim_rows:
            answer_id = str(row["answer_source_id"])
            answer_entity = labels.get(answer_id)
            if not answer_entity or "en" not in answer_entity.get("labels", {}):
                continue
            claims.append(
                ExtractedTemporalClaim(
                    **row,
                    answer_label=_english_label(answer_entity),
                )
            )
        claims = _deduplicate_claims(claims)
        if (
            len(claims) < 3
            or not _has_unique_probe(claims)
            or not _labels_are_unambiguous(claims)
            or not _has_answer_contrast(claims)
        ):
            continue
        revision = int(entity.get("lastrevid") or 0)
        if revision <= 0:
            continue
        source = ExtractedTemporalSource(
            source_id=f"wikidata:{context_id}:{spec.property_id}",
            source_revision=f"{context_id}@{revision}",
            extraction_time=extraction_time,
            property_id=spec.property_id,
            property_label=spec.relation_label,
            domain=spec.domain,
            relation=spec.relation,
            answer_noun=spec.answer_noun,
            context_source_id=context_id,
            context_label=context_label,
            context_revision_id=revision,
            claims=claims,
        )
        if source.source_id in forbidden_sources:
            exclusion_counts["source_id_overlap"] += 1
            continue
        source_entities = {
            source.context_source_id,
            *(claim.answer_source_id for claim in source.claims),
        }
        if source_entities & forbidden_entities:
            exclusion_counts["entity_id_overlap"] += 1
            continue
        by_property[spec.property_id].append(source)

    selected = _balanced_source_selection(by_property, target=target)
    if len(selected) < target:
        raise RuntimeError(
            f"Wikidata extraction produced only {len(selected)} usable temporal series; "
            f"requested {target}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(
        b"".join(
            orjson.dumps(row.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE)
            for row in selected
        )
    )
    temporary.replace(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_bytes(
        orjson.dumps(
            {
                "schema_version": "wikidata_temporal_source_manifest_v1",
                "source_file": output_path.name,
                "source_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "extraction_time": extraction_time,
                "sparql_endpoint": WIKIDATA_SPARQL_ENDPOINT,
                "entity_api_endpoint": WIKIDATA_API_ENDPOINT,
                "candidate_sampling": {
                    "method": "salted_sha256_prefix_buckets",
                    "salt": sampling_salt,
                    "buckets": list(hash_buckets),
                    "candidate_rows_per_property": candidate_rows_per_property,
                },
                "candidate_context_counts": {
                    key: len(value) for key, value in candidates_by_property.items()
                },
                "usable_source_counts": {key: len(value) for key, value in by_property.items()},
                "selected_source_counts": dict(
                    sorted(
                        {
                            key: sum(row.property_id == key for row in selected)
                            for key in _SPEC_BY_PROPERTY
                        }.items()
                    )
                ),
                "selected_sources": len(selected),
                "disjointness_exclusions": dict(sorted(exclusion_counts.items())),
                "forbidden_source_ids": len(forbidden_sources),
                "forbidden_entity_ids": len(forbidden_entities),
            },
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
    )
    manifest_temporary.replace(manifest_path)
    return selected


def reconstruct_wikidata_temporal_sources_from_cache(
    *,
    candidate_cache_path: Path,
    context_cache_path: Path,
    answer_cache_path: Path,
    extraction_time: datetime,
    candidate_rows_per_property: int = 5000,
    sampling_salt: str = SOURCE_SAMPLING_SALT,
    hash_buckets: tuple[str, ...] = SOURCE_HASH_BUCKETS,
) -> list[ExtractedTemporalSource]:
    """Rebuild the complete usable source population without network access.

    The parser and eligibility rules are identical to live extraction, but no
    balanced release sample is drawn. Supplying the original snapshot time
    prevents reconstruction from inventing new source provenance.
    """

    if extraction_time.tzinfo is None or extraction_time.utcoffset() is None:
        raise ValueError("extraction_time must be timezone-aware")
    _validate_sampling_contract(sampling_salt=sampling_salt, hash_buckets=hash_buckets)
    candidate_cache_key = {
        "properties": [spec.property_id for spec in WIKIDATA_PROPERTY_SPECS],
        "row_limit": candidate_rows_per_property,
        "salt": sampling_salt,
        "buckets": list(hash_buckets),
    }
    candidates_by_property = _load_candidate_cache(
        candidate_cache_path,
        expected_key=candidate_cache_key,
    )
    if candidates_by_property is None:
        raise ValueError(
            "Frozen Wikidata candidate cache is missing or has a different sampling contract"
        )
    entities = _load_entity_cache(context_cache_path, props="labels|claims|info")
    labels = _load_entity_cache(answer_cache_path, props="labels")
    if not entities:
        raise ValueError("Frozen Wikidata context cache is empty or has the wrong props contract")
    if not labels:
        raise ValueError("Frozen Wikidata answer cache is empty or has the wrong props contract")

    missing_contexts = sorted(
        {
            context_id
            for context_ids in candidates_by_property.values()
            for context_id in context_ids
            if context_id not in entities
        }
    )
    if missing_contexts:
        preview = ", ".join(missing_contexts[:10])
        raise ValueError(
            f"Frozen Wikidata context cache is incomplete ({len(missing_contexts)} missing): "
            f"{preview}"
        )

    reconstructed: list[ExtractedTemporalSource] = []
    for spec in WIKIDATA_PROPERTY_SPECS:
        for context_id in candidates_by_property.get(spec.property_id, []):
            entity = entities[context_id]
            if "en" not in entity.get("labels", {}):
                continue
            claim_rows = _parse_entity_claims(entity, spec.property_id)
            if len(claim_rows) < 3:
                continue
            claims: list[ExtractedTemporalClaim] = []
            for row in claim_rows:
                answer_id = str(row["answer_source_id"])
                answer_entity = labels.get(answer_id)
                if not answer_entity or "en" not in answer_entity.get("labels", {}):
                    continue
                claims.append(
                    ExtractedTemporalClaim(
                        **row,
                        answer_label=_english_label(answer_entity),
                    )
                )
            claims = _deduplicate_claims(claims)
            if (
                len(claims) < 3
                or not _has_unique_probe(claims)
                or not _labels_are_unambiguous(claims)
                or not _has_answer_contrast(claims)
            ):
                continue
            revision = int(entity.get("lastrevid") or 0)
            if revision <= 0:
                continue
            reconstructed.append(
                ExtractedTemporalSource(
                    source_id=f"wikidata:{context_id}:{spec.property_id}",
                    source_revision=f"{context_id}@{revision}",
                    extraction_time=extraction_time,
                    property_id=spec.property_id,
                    property_label=spec.relation_label,
                    domain=spec.domain,
                    relation=spec.relation,
                    answer_noun=spec.answer_noun,
                    context_source_id=context_id,
                    context_label=_english_label(entity),
                    context_revision_id=revision,
                    claims=claims,
                )
            )

    reconstructed.sort(key=lambda row: (row.property_id, _stable_hash(row.source_id)))
    source_ids = [row.source_id for row in reconstructed]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Reconstructed Wikidata source population contains duplicate source IDs")
    if not reconstructed:
        raise ValueError("Frozen Wikidata caches produced no usable temporal source series")
    return reconstructed


class ExtractedSourceDatasetGenerator(SyntheticDatasetGenerator):
    """Build deterministic benchmark worlds from frozen source statement series."""

    def __init__(
        self,
        *,
        sources: list[ExtractedTemporalSource],
        seed: int = 7,
        scenario_prefix: str = "xs",
    ) -> None:
        if not sources:
            raise ValueError("Certified generation requires at least one extracted source")
        if (
            not scenario_prefix
            or not scenario_prefix.isascii()
            or not scenario_prefix.isalnum()
            or not scenario_prefix[0].isalpha()
        ):
            raise ValueError("scenario_prefix must be a non-empty ASCII alphanumeric identifier")
        super().__init__(seed=seed)
        self.sources = sources
        self.scenario_prefix = scenario_prefix.lower()
        self.solver = GoldSolver()

    def generate(
        self,
        *,
        scenario_count: int,
        questions_per_scenario: int = 4,
    ) -> DatasetBundle:
        if questions_per_scenario != 4:
            raise ValueError("Extracted generation currently requires four questions per scenario")
        selected = _balanced_source_selection(
            _group_sources(self.sources),
            target=scenario_count,
        )
        if len(selected) < scenario_count:
            raise ValueError(f"Need {scenario_count} distinct source series, found {len(selected)}")

        scenarios: list[Scenario] = []
        entities: list[Entity] = []
        facts: list[Fact] = []
        snapshots: list[Snapshot] = []
        questions: list[Question] = []
        graph_paths = []
        context_packs = []
        answer_variants = []

        for index, source in enumerate(selected):
            peer = _peer_source(selected, index=index, source=source)
            built = self._build_scenario(index=index, source=source, peer=peer)
            scenario_entities, scenario_facts, scenario_snapshots, scenario_questions = built
            scenario_paths = self._generate_graph_paths(
                questions=scenario_questions,
                facts=scenario_facts,
                scenario_id=self._scenario_id(index),
            )
            scenario_contexts = self._generate_context_packs(
                questions=scenario_questions,
                facts=scenario_facts,
            )
            entity_by_id = {entity.entity_id: entity for entity in scenario_entities}
            scenario_answers = self._generate_answer_variants(
                questions=scenario_questions,
                facts=scenario_facts,
                entities=entity_by_id,
                paths=scenario_paths,
            )
            update_behavior = _update_behavior(index)
            scenario = Scenario(
                scenario_id=self._scenario_id(index),
                split_group_id=source.source_id,
                domain=source.domain,
                blueprint=f"wikidata_{source.property_id.lower()}_temporal_series",
                entities=scenario_entities,
                facts=scenario_facts,
                snapshots=scenario_snapshots,
                question_ids=[question.qid for question in scenario_questions],
                update_behavior=update_behavior,
                source_provenance=SourceProvenance(
                    source_id=source.source_id,
                    source_family="wikidata",
                    fidelity="source_extracted",
                    source_record_ids=sorted(
                        {fact.source_record_id for fact in scenario_facts if fact.source_record_id}
                    ),
                    source_revision=source.source_revision,
                    source_relation=f"{source.property_id}-{source.property_label}",
                    topology_signature=source.topology_signature,
                ),
                notes=(
                    "Generated from frozen Wikidata statements without changing entity names, "
                    "relation direction, or valid-time qualifiers. Snapshot interventions alter "
                    "evidence visibility only."
                ),
            )
            scenarios.append(scenario)
            entities.extend(scenario_entities)
            facts.extend(scenario_facts)
            snapshots.extend(scenario_snapshots)
            questions.extend(scenario_questions)
            graph_paths.extend(scenario_paths)
            context_packs.extend(scenario_contexts)
            answer_variants.extend(scenario_answers)

        splits = stable_group_splits(
            {
                scenario.scenario_id: scenario.split_group_id or scenario.scenario_id
                for scenario in scenarios
            },
            namespace=f"source-extracted:{self.seed}",
        )
        return DatasetBundle(
            scenarios=scenarios,
            entities=entities,
            facts=facts,
            snapshots=snapshots,
            questions=questions,
            graph_paths=graph_paths,
            context_packs=context_packs,
            answer_variants=answer_variants,
            splits=splits,
        )

    def _build_scenario(
        self,
        *,
        index: int,
        source: ExtractedTemporalSource,
        peer: ExtractedTemporalSource,
    ) -> tuple[list[Entity], list[Fact], list[Snapshot], list[Question]]:
        scenario_id = self._scenario_id(index)
        context = Entity(
            entity_id=f"e_{scenario_id}_context",
            name=source.context_label,
            entity_type=EntityType.WIKIDATA_ENTITY,
            aliases=[source.context_source_id],
            domain=source.domain,
        )
        qualifier = Entity(
            entity_id=f"e_{scenario_id}_target",
            name=source.answer_noun,
            entity_type=EntityType.ROLE,
            domain=source.domain,
        )
        answer_entities: dict[str, Entity] = {}
        for claim in source.claims:
            answer_entities.setdefault(
                claim.answer_source_id,
                Entity(
                    entity_id=stable_opaque_id("e", scenario_id, "answer", claim.answer_source_id),
                    name=claim.answer_label,
                    entity_type=EntityType.WIKIDATA_ENTITY,
                    aliases=[claim.answer_source_id],
                    domain=source.domain,
                ),
            )

        probe_claim, probe_time = _unique_probe(source.claims)
        behavior = _update_behavior(index)
        paired = behavior != "not_applicable"
        scenario_facts: list[Fact] = []
        for claim in source.claims:
            answer = answer_entities[claim.answer_source_id]
            visible_from = (
                "S1"
                if behavior == "answer_should_change"
                and claim.statement_id == probe_claim.statement_id
                else "S0"
            )
            scenario_facts.append(
                _source_fact(
                    scenario_id=scenario_id,
                    context=context,
                    qualifier=qualifier,
                    answer=answer,
                    source=source,
                    claim=claim,
                    visible_from=visible_from,
                    role=FactRole.SOURCE_ASSERTION,
                )
            )

        peer_context, peer_answer, peer_fact = _peer_fact(
            scenario_id=scenario_id,
            source=peer,
            qualifier=qualifier,
            query_time=probe_time,
            visible_from="S1" if behavior == "answer_should_stay" else "S0",
        )
        scenario_facts.append(peer_fact)
        scenario_entities = [
            context,
            qualifier,
            *answer_entities.values(),
            peer_context,
            peer_answer,
        ]

        snapshot_date = source.extraction_time.date()
        s0_ids = [fact.fact_id for fact in scenario_facts if fact.snapshot_visible_from == "S0"]
        s1_ids = [fact.fact_id for fact in scenario_facts]
        scenario_snapshots = [
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S0",
                snapshot_time=snapshot_date - timedelta(days=1),
                visible_fact_ids=s0_ids,
                description="Controlled evidence snapshot before one declared update.",
            ),
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S1",
                snapshot_time=snapshot_date,
                visible_fact_ids=s1_ids,
                description="Controlled evidence snapshot after one declared update.",
            ),
        ]
        programs = _question_programs(
            index=index,
            source=source,
            context_id=context.entity_id,
            object_id=qualifier.entity_id,
            probe_time=probe_time,
            paired=behavior != "not_applicable",
        )
        question_rows: list[Question] = []
        entity_by_id = {entity.entity_id: entity for entity in scenario_entities}
        for q_index, program in enumerate(programs):
            answer_ids, evidence_ids = self.solver.solve(scenario_facts, program)
            should_abstain = not answer_ids
            operator = program.operator
            text = _render_question(
                source=source,
                program=program,
                snapshot_time=next(
                    snapshot.snapshot_time
                    for snapshot in scenario_snapshots
                    if snapshot.snapshot_id == program.snapshot_id
                ),
                show_snapshot=paired and q_index < 2,
                answer_count=len(answer_ids),
            )
            qid = f"q_{scenario_id}_{q_index:02d}"
            question_rows.append(
                Question(
                    qid=qid,
                    scenario_id=scenario_id,
                    dataset_family=DatasetFamily.SYNTH,
                    canonical_question=text,
                    question=text,
                    program=program,
                    temporal_operator=operator,
                    answer_type=(
                        AnswerType.REFUSAL
                        if should_abstain
                        else AnswerType.LIST
                        if len(answer_ids) > 1
                        else AnswerType.ENTITY
                    ),
                    gold_answer_entity_ids=answer_ids,
                    gold_answer_text=[entity_by_id[answer_id].name for answer_id in answer_ids],
                    required_valid_evidence_ids=evidence_ids,
                    should_abstain=should_abstain,
                    system_difficulty=_provisional_system_difficulty(operator, len(evidence_ids)),
                    eval_difficulty=_provisional_eval_difficulty(operator, len(answer_ids)),
                    difficulty_provenance="source_heuristic",
                    human_pool_candidate=q_index in {1, 2},
                    semantic_series_id=_semantic_series_id(
                        source=source,
                        program=program,
                        paired_update=paired and q_index < 2,
                        probe_time=probe_time,
                    ),
                    template_family_id=f"{source.property_id}:{operator}",
                    certification_status="certified",
                )
            )
        return scenario_entities, scenario_facts, scenario_snapshots, question_rows

    def _scenario_id(self, index: int) -> str:
        return f"{self.scenario_prefix}_{index:04d}"


def _candidate_context_ids(
    client: httpx.Client,
    *,
    property_id: str,
    row_limit: int,
    sampling_salt: str = SOURCE_SAMPLING_SALT,
    hash_buckets: tuple[str, ...] = SOURCE_HASH_BUCKETS,
) -> list[str]:
    counts: defaultdict[str, int] = defaultdict(int)
    _validate_sampling_contract(sampling_salt=sampling_salt, hash_buckets=hash_buckets)
    rows_per_bucket = max(1, (row_limit + len(hash_buckets) - 1) // len(hash_buckets))
    for bucket in hash_buckets:
        query = f"""
SELECT ?context ?statement WHERE {{
  ?context p:{property_id} ?statement .
  ?statement ps:{property_id} ?value ; pq:P580 ?start .
  FILTER(
    SUBSTR(
      SHA256(CONCAT(STR(?context), "{sampling_salt}:{property_id}")),
      1,
      1
    ) = "{bucket}"
  )
}}
LIMIT {rows_per_bucket}
""".strip()
        payload = _get_json_with_retry(
            client,
            WIKIDATA_SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
        )
        for binding in payload["results"]["bindings"]:
            context_id = str(binding["context"]["value"]).rsplit("/", 1)[-1]
            counts[context_id] += 1
    return sorted(
        (context_id for context_id, count in counts.items() if count >= 3),
        key=lambda value: _stable_hash(f"{property_id}:{value}"),
    )


def _validate_sampling_contract(
    *,
    sampling_salt: str,
    hash_buckets: tuple[str, ...],
) -> None:
    if not sampling_salt.strip():
        raise ValueError("sampling_salt must be non-empty")
    if not hash_buckets or len(hash_buckets) != len(set(hash_buckets)):
        raise ValueError("hash_buckets must be non-empty and unique")
    invalid = [
        bucket
        for bucket in hash_buckets
        if len(bucket) != 1 or bucket.casefold() not in "0123456789abcdef"
    ]
    if invalid:
        raise ValueError(f"hash_buckets must contain hexadecimal prefixes: {invalid}")


def _candidate_is_forbidden(
    context_id: str,
    *,
    property_id: str,
    forbidden_sources: set[str],
    forbidden_entities: set[str],
    exclusion_counts: Counter[str],
) -> bool:
    source_id = f"wikidata:{context_id}:{property_id}"
    if source_id in forbidden_sources:
        exclusion_counts["candidate_source_id_overlap"] += 1
        return True
    if context_id in forbidden_entities:
        exclusion_counts["candidate_context_entity_overlap"] += 1
        return True
    return False


def _fetch_entities(
    client: httpx.Client,
    entity_ids: list[str],
    *,
    props: str,
    batch_size: int,
    cache_path: Path | None = None,
    max_workers: int = 2,
) -> dict[str, dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    result = _load_entity_cache(cache_path, props=props) if cache_path else {}
    missing_ids = [entity_id for entity_id in entity_ids if entity_id not in result]
    checkpoint_batches = max(1, ENTITY_CACHE_CHECKPOINT_IDS // batch_size)
    batches = [
        missing_ids[start : start + batch_size]
        for start in range(0, len(missing_ids), batch_size)
    ]

    def fetch_batch(batch: list[str]) -> tuple[list[str], dict[str, Any]]:
        payload = _get_json_with_retry(
            client,
            WIKIDATA_API_ENDPOINT,
            params={
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": props,
                "languages": "en",
                "format": "json",
                # This offline extraction is resumable. A generous bound retains
                # maxlag protection without turning normal Query Service lag into
                # hundreds of empty entity batches.
                "maxlag": WIKIDATA_API_MAXLAG_SECONDS,
            },
        )
        response_entities = payload.get("entities")
        if not isinstance(response_entities, dict):
            raise RuntimeError("Wikidata entity response did not contain an entities object")
        omitted = sorted(set(batch) - set(response_entities))
        if omitted:
            raise RuntimeError(
                "Wikidata entity response omitted requested IDs: " + ", ".join(omitted)
            )
        return batch, response_entities

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wikidata-entity") as pool:
        responses = pool.map(fetch_batch, batches)
        for batch_number, (_, response_entities) in enumerate(responses, start=1):
            result.update(response_entities)
            LOGGER.info(
                "Fetched Wikidata entity batch %d/%d for %s",
                min(batch_number * batch_size, len(missing_ids)),
                len(missing_ids),
                props,
            )
            final_batch = batch_number == len(batches)
            if cache_path and (batch_number % checkpoint_batches == 0 or final_batch):
                _write_json_atomic(cache_path, {"props": props, "entities": result})
    return result


def _load_candidate_cache(
    path: Path,
    *,
    expected_key: dict[str, object],
) -> dict[str, list[str]] | None:
    if not path.exists():
        return None
    payload = orjson.loads(path.read_bytes())
    if payload.get("cache_key") != expected_key:
        return None
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError(f"Invalid source candidate cache: {path}")
    return {
        str(property_id): [str(entity_id) for entity_id in entity_ids]
        for property_id, entity_ids in candidates.items()
    }


def _load_entity_cache(path: Path | None, *, props: str) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = orjson.loads(path.read_bytes())
    if payload.get("props") != props:
        return {}
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError(f"Invalid Wikidata entity cache: {path}")
    return {str(entity_id): entity for entity_id, entity in entities.items()}


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _parse_entity_claims(entity: dict[str, Any], property_id: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for statement in entity.get("claims", {}).get(property_id, []):
        if statement.get("rank") == "deprecated":
            continue
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        answer_id = value.get("id") if isinstance(value, dict) else None
        qualifiers = statement.get("qualifiers", {})
        if len(qualifiers.get("P580", [])) != 1 or len(qualifiers.get("P582", [])) > 1:
            continue
        start_payload = _qualifier_time(qualifiers.get("P580", []))
        end_payload = _qualifier_time(qualifiers.get("P582", []))
        if not answer_id or start_payload is None:
            continue
        start, start_precision = _parse_wikidata_time(start_payload, end_boundary=False)
        if start is None:
            continue
        end: date | None = None
        end_precision: str | None = None
        if end_payload is not None:
            end, end_precision = _parse_wikidata_time(end_payload, end_boundary=True)
            if end is None or end < start:
                continue
        statement_id = str(statement.get("id") or "")
        if not statement_id:
            continue
        parsed.append(
            {
                "statement_id": statement_id,
                "answer_source_id": str(answer_id),
                "start": start,
                "end": end,
                "start_precision": start_precision,
                "end_precision": end_precision,
            }
        )
    return sorted(
        parsed,
        key=lambda row: (row["start"], row["end"] or date.max, row["statement_id"]),
    )


def _qualifier_time(snaks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for snak in snaks:
        value = snak.get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("time"):
            return value
    return None


def _parse_wikidata_time(
    payload: dict[str, Any],
    *,
    end_boundary: bool,
) -> tuple[date | None, Literal["day", "month", "year"] | None]:
    raw = str(payload.get("time") or "")
    calendar_id = str(payload.get("calendarmodel") or "").rsplit("/", 1)[-1]
    if calendar_id and calendar_id != GREGORIAN_CALENDAR_ID:
        return None, None
    if not raw.startswith("+"):
        return None, None
    try:
        year = int(raw[1:5])
        month = int(raw[6:8])
        day = int(raw[9:11])
    except (TypeError, ValueError):
        return None, None
    precision = int(payload.get("precision") or 0)
    try:
        if precision >= 11:
            return date(year, month, day), "day"
        if precision == 10:
            if end_boundary:
                next_month = date(
                    year + (month == 12),
                    1 if month == 12 else month + 1,
                    1,
                )
                return next_month - timedelta(days=1), "month"
            return date(year, month, 1), "month"
        if precision == 9:
            return (date(year, 12, 31) if end_boundary else date(year, 1, 1)), "year"
    except ValueError:
        return None, None
    return None, None


def _english_label(entity: dict[str, Any]) -> str:
    return str(entity["labels"]["en"]["value"]).strip()


def _deduplicate_claims(
    claims: list[ExtractedTemporalClaim],
) -> list[ExtractedTemporalClaim]:
    unique: dict[tuple[str, date, date | None], ExtractedTemporalClaim] = {}
    for claim in claims:
        unique.setdefault(
            (claim.answer_source_id, claim.start, claim.end),
            claim,
        )
    return sorted(
        unique.values(),
        key=lambda claim: (claim.start, claim.end or date.max, claim.answer_source_id),
    )


def _labels_are_unambiguous(claims: list[ExtractedTemporalClaim]) -> bool:
    entity_ids_by_label: defaultdict[str, set[str]] = defaultdict(set)
    for claim in claims:
        entity_ids_by_label[normalize_visible_text(claim.answer_label)].add(claim.answer_source_id)
    return all(label and len(entity_ids) == 1 for label, entity_ids in entity_ids_by_label.items())


def _has_answer_contrast(claims: list[ExtractedTemporalClaim]) -> bool:
    """Require a source-local answer that can serve as a valid counterfactual."""

    return len({claim.answer_source_id for claim in claims}) >= 2


def _has_unique_probe(claims: list[Any]) -> bool:
    try:
        _unique_probe(claims)
    except ValueError:
        return False
    return True


def _unique_probe(
    claims: list[Any],
) -> tuple[Any, date]:
    ordered = sorted(claims, key=lambda claim: (claim.start, claim.end or date.max), reverse=True)
    for claim in ordered:
        candidates = [claim.start]
        if claim.end is not None:
            candidates.append(claim.start + (claim.end - claim.start) // 2)
        for candidate in candidates:
            active = [
                item
                for item in claims
                if item.start <= candidate and (item.end is None or item.end >= candidate)
            ]
            if len(active) == 1 and active[0].statement_id == claim.statement_id:
                return claim, candidate
    raise ValueError("Temporal series has no uniquely answerable point")


def _balanced_source_selection(
    by_property: dict[str, list[ExtractedTemporalSource]],
    *,
    target: int,
) -> list[ExtractedTemporalSource]:
    queues = {
        key: sorted(values, key=lambda value: _stable_hash(value.source_id))
        for key, values in by_property.items()
    }
    selected: list[ExtractedTemporalSource] = []
    deferred: list[ExtractedTemporalSource] = []
    used_context_ids: set[str] = set()
    used_context_labels: set[str] = set()
    keys = sorted(queues)
    positions = dict.fromkeys(keys, 0)
    while len(selected) < target:
        progressed = False
        for key in keys:
            position = positions[key]
            if position >= len(queues[key]) or len(selected) >= target:
                continue
            candidate = queues[key][position]
            positions[key] += 1
            progressed = True
            normalized_label = normalize_visible_text(candidate.context_label)
            if (
                candidate.context_source_id in used_context_ids
                or normalized_label in used_context_labels
            ):
                deferred.append(candidate)
                continue
            selected.append(candidate)
            used_context_ids.add(candidate.context_source_id)
            used_context_labels.add(normalized_label)
        if not progressed:
            break
    if len(selected) < target:
        selected.extend(
            sorted(deferred, key=lambda value: _stable_hash(value.source_id))[
                : target - len(selected)
            ]
        )
    return selected


def _group_sources(
    sources: list[ExtractedTemporalSource],
) -> dict[str, list[ExtractedTemporalSource]]:
    grouped: defaultdict[str, list[ExtractedTemporalSource]] = defaultdict(list)
    for source in sources:
        grouped[source.property_id].append(source)
    return grouped


def _peer_source(
    sources: list[ExtractedTemporalSource],
    *,
    index: int,
    source: ExtractedTemporalSource,
) -> ExtractedTemporalSource:
    fallback: ExtractedTemporalSource | None = None
    for offset in range(1, len(sources)):
        candidate = sources[(index + offset) % len(sources)]
        if not _answer_sets_are_disjoint(source, candidate):
            continue
        if candidate.property_id == source.property_id:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError(f"No answer-disjoint peer source available for {source.source_id}")


def _answer_sets_are_disjoint(
    source: ExtractedTemporalSource,
    candidate: ExtractedTemporalSource,
) -> bool:
    source_ids = {claim.answer_source_id for claim in source.claims}
    candidate_ids = {claim.answer_source_id for claim in candidate.claims}
    if not source_ids.isdisjoint(candidate_ids):
        return False
    source_labels = {normalize_visible_text(claim.answer_label) for claim in source.claims}
    candidate_labels = {
        normalize_visible_text(claim.answer_label) for claim in candidate.claims
    }
    return source_labels.isdisjoint(candidate_labels)


def _source_fact(
    *,
    scenario_id: str,
    context: Entity,
    qualifier: Entity,
    answer: Entity,
    source: ExtractedTemporalSource,
    claim: ExtractedTemporalClaim,
    visible_from: str,
    role: FactRole,
) -> Fact:
    interval = claim.interval()
    spec = _SPEC_BY_PROPERTY[source.property_id]
    statement = spec.evidence_template.format(
        context=context.name,
        answer=answer.name,
        interval=_source_interval_text(claim),
    )
    return Fact(
        fact_id=stable_opaque_id("f", scenario_id, claim.statement_id),
        scenario_id=scenario_id,
        subject_id=answer.entity_id,
        relation=source.relation,
        object_id=qualifier.entity_id,
        context_id=context.entity_id,
        answer_entity_id=answer.entity_id,
        graph_source_id=context.entity_id,
        graph_target_id=answer.entity_id,
        source_relation_id=source.property_id,
        source_relation_label=source.property_label,
        relation_direction="directed",
        source_record_id=claim.statement_id,
        source_revision=source.source_revision,
        valid_time=interval,
        publication_time=None,
        transaction_time=None,
        snapshot_visible_from=visible_from,
        source_type=f"wikidata:{source.property_id}",
        provenance_reliability="high",
        fact_role=role,
        canonical_evidence=f"According to Wikidata, {statement}",
    )


def _peer_fact(
    *,
    scenario_id: str,
    source: ExtractedTemporalSource,
    qualifier: Entity,
    query_time: date,
    visible_from: str,
) -> tuple[Entity, Entity, Fact]:
    active = [
        claim
        for claim in source.claims
        if claim.start <= query_time and (claim.end is None or claim.end >= query_time)
    ]
    claim = active[0] if active else source.claims[0]
    context = Entity(
        entity_id=f"e_{scenario_id}_peer_context",
        name=source.context_label,
        entity_type=EntityType.WIKIDATA_ENTITY,
        aliases=[source.context_source_id],
        domain=source.domain,
    )
    answer = Entity(
        entity_id=f"e_{scenario_id}_peer_answer",
        name=claim.answer_label,
        entity_type=EntityType.WIKIDATA_ENTITY,
        aliases=[claim.answer_source_id],
        domain=source.domain,
    )
    fact = _source_fact(
        scenario_id=scenario_id,
        context=context,
        qualifier=qualifier,
        answer=answer,
        source=source,
        claim=claim,
        visible_from=visible_from,
        role=FactRole.HARD_NEGATIVE,
    )
    return context, answer, fact


def _update_behavior(index: int) -> str:
    remainder = index % 4
    if remainder == 0:
        return "answer_should_change"
    if remainder == 1:
        return "answer_should_stay"
    return "not_applicable"


def _question_programs(
    *,
    index: int,
    source: ExtractedTemporalSource,
    context_id: str,
    object_id: str,
    probe_time: date,
    paired: bool,
) -> list[QuestionProgram]:
    programs: list[QuestionProgram] = []
    if paired:
        for snapshot_id in ("S0", "S1"):
            programs.append(
                _program(
                    operator=TemporalOperator.AS_OF,
                    query_time=point(probe_time.year, probe_time.month, probe_time.day),
                    source=source,
                    context_id=context_id,
                    object_id=object_id,
                    snapshot_id=snapshot_id,
                )
            )
    candidates = _candidate_programs(
        source=source,
        context_id=context_id,
        object_id=object_id,
        probe_time=probe_time,
    )
    start = index % len(candidates)
    for offset in range(len(candidates)):
        if len(programs) == 4:
            break
        candidate = candidates[(start + offset) % len(candidates)]
        if any(
            program.operator == candidate.operator and program.query_time == candidate.query_time
            for program in programs
        ):
            continue
        programs.append(candidate)
    if len(programs) != 4:
        raise ValueError(f"Could not create four distinct programs for {source.source_id}")
    return programs


def _candidate_programs(
    *,
    source: ExtractedTemporalSource,
    context_id: str,
    object_id: str,
    probe_time: date,
) -> list[QuestionProgram]:
    claims = sorted(source.claims, key=lambda claim: (claim.start, claim.end or date.max))
    probe_claim, unique_probe = _unique_probe(claims)
    candidates = [
        _program(
            operator=TemporalOperator.CURRENT,
            query_time=point(unique_probe.year, unique_probe.month, unique_probe.day),
            source=source,
            context_id=context_id,
            object_id=object_id,
        ),
        _program(
            operator=TemporalOperator.EFFECTIVE,
            query_time=point(unique_probe.year, unique_probe.month, unique_probe.day),
            source=source,
            context_id=context_id,
            object_id=object_id,
        ),
        _program(
            operator=TemporalOperator.FIRST,
            query_time=point(probe_time.year, probe_time.month, probe_time.day),
            source=source,
            context_id=context_id,
            object_id=object_id,
        ),
        _program(
            operator=TemporalOperator.LATEST,
            query_time=point(probe_time.year, probe_time.month, probe_time.day),
            source=source,
            context_id=context_id,
            object_id=object_id,
        ),
        _program(
            operator=TemporalOperator.LAST,
            query_time=point(probe_time.year, probe_time.month, probe_time.day),
            source=source,
            context_id=context_id,
            object_id=object_id,
        ),
    ]
    ended = [claim for claim in claims if claim.end is not None]
    if ended:
        cutoff = max(claim.end for claim in ended) + timedelta(days=1)
        candidates.append(
            _program(
                operator=TemporalOperator.BEFORE,
                query_time=point(cutoff.year, cutoff.month, cutoff.day),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
        candidates.append(
            _program(
                operator=TemporalOperator.EXPIRED,
                query_time=point(cutoff.year, cutoff.month, cutoff.day),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
    if len(claims) >= 2:
        cutoff = claims[1].start - timedelta(days=1)
        candidates.append(
            _program(
                operator=TemporalOperator.AFTER,
                query_time=point(cutoff.year, cutoff.month, cutoff.day),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
        range_start = claims[0].start
        range_end = claims[min(2, len(claims) - 1)].start
        candidates.append(
            _program(
                operator=TemporalOperator.BETWEEN,
                query_time=TemporalInterval(
                    type="interval",
                    start=range_start,
                    end=range_end,
                    granularity="day",
                ),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
    during_end = (
        min(probe_claim.end, unique_probe + timedelta(days=30))
        if probe_claim.end is not None
        else unique_probe + timedelta(days=30)
    )
    if during_end > unique_probe:
        candidates.append(
            _program(
                operator=TemporalOperator.DURING,
                query_time=TemporalInterval(
                    type="interval",
                    start=unique_probe,
                    end=during_end,
                    granularity="day",
                ),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
    previous_probe = _ordinal_probe(claims, direction="previous")
    if previous_probe is not None:
        candidates.append(
            _program(
                operator=TemporalOperator.PREVIOUS,
                query_time=point(
                    previous_probe.year,
                    previous_probe.month,
                    previous_probe.day,
                ),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
    next_probe = _ordinal_probe(claims, direction="next")
    if next_probe is not None:
        candidates.append(
            _program(
                operator=TemporalOperator.NEXT,
                query_time=point(next_probe.year, next_probe.month, next_probe.day),
                source=source,
                context_id=context_id,
                object_id=object_id,
            )
        )
    return candidates


def _program(
    *,
    operator: TemporalOperator,
    query_time: TemporalInterval,
    source: ExtractedTemporalSource,
    context_id: str,
    object_id: str,
    snapshot_id: str = "S1",
) -> QuestionProgram:
    return QuestionProgram(
        operator=operator,
        target=source.answer_noun,
        query_time=query_time,
        relation=source.relation,
        context_id=context_id,
        object_id=object_id,
        snapshot_id=snapshot_id,
        answer_function=f"{operator}_{source.property_id.lower()}_validity",
        required_path_semantics="single_source_statement",
        temporal_basis="world_valid_time",
    )


def _render_question(
    *,
    source: ExtractedTemporalSource,
    program: QuestionProgram,
    snapshot_time: date,
    show_snapshot: bool,
    answer_count: int,
) -> str:
    spec = _SPEC_BY_PROPERTY[source.property_id]
    date_text = _date_text(program.query_time.start)
    prefix = (
        f"Using the evidence snapshot dated {_date_text(snapshot_time)}, " if show_snapshot else ""
    )
    relation_word = "relationships" if answer_count > 1 else "relationship"
    if program.operator == TemporalOperator.AS_OF:
        if answer_count > 1:
            body = (
                f"Which {source.answer_noun} relationships for {source.context_label} were "
                f"in effect on {date_text}?"
            )
        else:
            body = spec.point_template.format(context=source.context_label, date=date_text)
    elif program.operator == TemporalOperator.CURRENT:
        verb = "were" if answer_count > 1 else "was"
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} {verb} current "
            f"on {date_text}?"
        )
    elif program.operator == TemporalOperator.EFFECTIVE:
        verb = "were" if answer_count > 1 else "was"
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} {verb} "
            f"in effect on {date_text}?"
        )
    elif program.operator == TemporalOperator.FIRST:
        if answer_count > 1:
            body = (
                f"Which {source.answer_noun} relationships for {source.context_label} began "
                "at the earliest recorded time?"
            )
        else:
            body = f"What was the first recorded {source.answer_noun} for {source.context_label}?"
    elif program.operator == TemporalOperator.LATEST:
        if answer_count > 1:
            body = (
                f"Which {source.answer_noun} relationships for {source.context_label} began "
                f"at the latest recorded time by {date_text}?"
            )
        else:
            body = (
                f"What was the latest recorded {source.answer_noun} for {source.context_label} "
                f"as of {date_text}?"
            )
    elif program.operator == TemporalOperator.LAST:
        if answer_count > 1:
            body = (
                f"Which were the last recorded {source.answer_noun} relationships to begin "
                f"for {source.context_label} by {date_text}?"
            )
        else:
            body = (
                f"Which was the last recorded {source.answer_noun} relationship to begin for "
                f"{source.context_label} by {date_text}?"
            )
    elif program.operator == TemporalOperator.BEFORE:
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} ended most "
            f"recently before {date_text}?"
        )
    elif program.operator == TemporalOperator.EXPIRED:
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} had most "
            f"recently expired by {date_text}?"
        )
    elif program.operator == TemporalOperator.AFTER:
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} began first "
            f"after {date_text}?"
        )
    elif program.operator == TemporalOperator.BETWEEN:
        body = (
            f"Which {source.answer_noun} relationships applied to {source.context_label} at any "
            f"time from {_date_text(program.query_time.start)} through "
            f"{_date_text(program.query_time.end)}?"
        )
    elif program.operator == TemporalOperator.DURING:
        body = (
            f"Which {source.answer_noun} relationships for {source.context_label} held "
            f"throughout the period from {_date_text(program.query_time.start)} through "
            f"{_date_text(program.query_time.end)}?"
        )
    elif program.operator == TemporalOperator.PREVIOUS:
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} ended most "
            f"recently before the relationship in effect on {date_text} began?"
        )
    elif program.operator == TemporalOperator.NEXT:
        body = (
            f"Which {source.answer_noun} {relation_word} for {source.context_label} began first "
            f"after the relationship in effect on {date_text} ended?"
        )
    else:
        body = spec.point_template.format(context=source.context_label, date=date_text)
    if not prefix:
        return body
    return prefix + body[0].lower() + body[1:]


def _source_interval_text(claim: ExtractedTemporalClaim) -> str:
    interval = claim.interval()
    if claim.end is not None and claim.start_precision == claim.end_precision == "day":
        return human_interval(interval)

    start = _precision_date_text(claim.start, claim.start_precision)
    if claim.end is None:
        source_text = (
            f"starting {_precision_preposition(claim.start_precision)} {start}, "
            "with no end date recorded"
        )
        precision_text = f"source start precision: {claim.start_precision}"
    else:
        end_precision = claim.end_precision or claim.start_precision
        source_text = f"from {start} through {_precision_date_text(claim.end, end_precision)}"
        precision_text = f"source precision: start={claim.start_precision}, end={end_precision}"
    return (
        f"{source_text} ({precision_text}; benchmark-normalized interval: "
        f"{human_interval(interval)})"
    )


def _precision_date_text(value: date, precision: str) -> str:
    if precision == "year":
        return str(value.year)
    if precision == "month":
        return value.strftime("%B %Y")
    return _date_text(value)


def _precision_preposition(precision: str) -> str:
    return "on" if precision == "day" else "in"


def _semantic_series_id(
    *,
    source: ExtractedTemporalSource,
    program: QuestionProgram,
    paired_update: bool,
    probe_time: date,
) -> str:
    if paired_update:
        return f"{source.source_id}:update:{probe_time.isoformat()}"
    operator = TemporalOperator(program.operator)
    equivalence = {
        TemporalOperator.AS_OF: "point_validity",
        TemporalOperator.CURRENT: "point_validity",
        TemporalOperator.EFFECTIVE: "point_validity",
        TemporalOperator.LATEST: "latest_start",
        TemporalOperator.LAST: "latest_start",
        TemporalOperator.BEFORE: "latest_end",
        TemporalOperator.EXPIRED: "latest_end",
    }.get(operator, str(operator))
    if operator == TemporalOperator.FIRST:
        return f"{source.source_id}:{equivalence}"
    return f"{source.source_id}:{equivalence}:{program.query_time.start}:{program.query_time.end}"


def _provisional_system_difficulty(
    operator: TemporalOperator,
    evidence_count: int,
) -> SystemDifficulty:
    if evidence_count > 1 or operator in {TemporalOperator.BETWEEN, TemporalOperator.PREVIOUS}:
        return SystemDifficulty.HARD
    if operator in {TemporalOperator.AS_OF, TemporalOperator.CURRENT, TemporalOperator.FIRST}:
        return SystemDifficulty.EASY
    return SystemDifficulty.MEDIUM


def _provisional_eval_difficulty(
    operator: TemporalOperator,
    answer_count: int,
) -> EvalDifficulty:
    if answer_count > 1:
        return EvalDifficulty.HARD
    if operator in {
        TemporalOperator.BEFORE,
        TemporalOperator.EXPIRED,
        TemporalOperator.LATEST,
        TemporalOperator.LAST,
        TemporalOperator.PREVIOUS,
        TemporalOperator.NEXT,
    }:
        return EvalDifficulty.MEDIUM
    return EvalDifficulty.EASY


def _date_text(value: date | None) -> str:
    if value is None:
        return "the stated date"
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _get_json_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, object],
    attempts: int = 8,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        response: httpx.Response | None = None
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object from {url}")
            api_error = payload.get("error")
            if isinstance(api_error, dict):
                code = str(api_error.get("code") or "unknown")
                info = str(api_error.get("info") or "source API error")
                if code in {"maxlag", "ratelimited", "readonly", "internal_api_error"}:
                    raise _RetryableSourceResponse(f"{code}: {info}")
                raise RuntimeError(f"Source API rejected the request ({code}): {info}")
            return payload
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status != 429 and status < 500:
                raise
            last_error = exc
        except (httpx.TransportError, ValueError, _RetryableSourceResponse) as exc:
            last_error = exc
        if attempt + 1 == attempts:
            break
        retry_after = response.headers.get("retry-after") if response is not None else None
        try:
            requested_delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            requested_delay = 0.0
        time.sleep(max(requested_delay, min(2 ** (attempt + 1), 60)))
    raise RuntimeError(f"Source request failed after {attempts} attempts: {url}") from last_error


def _ordinal_probe(
    claims: list[ExtractedTemporalClaim],
    *,
    direction: Literal["previous", "next"],
) -> date | None:
    ordered = sorted(claims, key=lambda claim: (claim.start, claim.end or date.max))
    for claim in ordered:
        probe = _unique_point_for_claim(claim, ordered)
        if probe is None:
            continue
        if direction == "previous":
            previous = [
                candidate
                for candidate in ordered
                if candidate.end is not None and candidate.end < claim.start
            ]
            if previous:
                latest_end = max(candidate.end for candidate in previous if candidate.end)
                selected = [candidate for candidate in previous if candidate.end == latest_end]
                if any(
                    candidate.answer_source_id != claim.answer_source_id for candidate in selected
                ):
                    return probe
        if direction == "next" and claim.end is not None:
            following = [candidate for candidate in ordered if candidate.start > claim.end]
            if following:
                first_start = min(candidate.start for candidate in following)
                selected = [candidate for candidate in following if candidate.start == first_start]
                if any(
                    candidate.answer_source_id != claim.answer_source_id for candidate in selected
                ):
                    return probe
    return None


def _unique_point_for_claim(
    claim: ExtractedTemporalClaim,
    claims: list[ExtractedTemporalClaim],
) -> date | None:
    candidates = [claim.start]
    if claim.end is not None:
        candidates.append(claim.start + (claim.end - claim.start) // 2)
    for candidate in candidates:
        active = [
            item
            for item in claims
            if item.start <= candidate and (item.end is None or item.end >= candidate)
        ]
        if len(active) == 1 and active[0].statement_id == claim.statement_id:
            return candidate
    return None
