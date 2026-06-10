from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.dataset.domains import DOMAIN_SPECS, DomainSpec
from tcred.dataset.generator import (
    FIRST_NAMES,
    LAST_NAMES,
    LOCATIONS,
    SyntheticDatasetGenerator,
    _snapshot_visible_from_for_timeline,
    _timeline_year_profile,
    _update_behavior_for_index,
    stable_opaque_id,
)
from tcred.dataset.intervals import make_interval, unknown_interval
from tcred.dataset.models import (
    Entity,
    EntityType,
    Fact,
    FactRole,
    Relation,
    Snapshot,
    TemporalInterval,
)
from tcred.dataset.solver import fact_visible
from tcred.dataset.verbalize import entity_lookup, fact_sentence


class SourceTemporalClaim(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    label: str
    start_year: int
    end_year: int | None = None
    role: Literal["stale", "valid", "future"] = "stale"

    def interval(self) -> TemporalInterval:
        return make_interval(self.start_year, self.end_year)


class SourceGroundedSubgraph(BaseModel):
    """Normalized source subgraph used before conversion into T-CRED artifacts.

    The schema is intentionally source-agnostic: extraction from Wikidata/EventKG should happen
    upstream, then write this compact representation. Generation then remains deterministic and
    does not depend on live network state or changing SPARQL endpoints.
    """

    model_config = ConfigDict(use_enum_values=True)

    source_id: str
    source_family: Literal["wikidata", "eventkg", "domain_registry"]
    source_fidelity: Literal["pattern_only", "source_extracted"] = "pattern_only"
    source_record_ids: list[str] = Field(default_factory=list)
    source_revision: str | None = None
    domain: str
    relation: Relation
    path_relation: Relation = Relation.AFFILIATED_WITH
    source_relation: str
    source_path_relation: str
    context_label: str
    object_label: str
    hard_negative_context_label: str
    claims: list[SourceTemporalClaim] = Field(min_length=5)
    degree_hint: int = 10
    topology_signature: str
    notes: str

    def score(self) -> float:
        temporal_roles = {claim.role for claim in self.claims}
        temporal_score = len(temporal_roles) / 3
        interval_score = sum(claim.end_year is not None for claim in self.claims) / len(self.claims)
        hub_penalty = min(self.degree_hint / 200, 0.7)
        return round(temporal_score + interval_score - hub_penalty, 4)


class SourceGroundedDatasetGenerator(SyntheticDatasetGenerator):
    """Generate T-CRED scenarios from normalized real-world temporal subgraph patterns."""

    def __init__(
        self,
        seed: int = 7,
        *,
        source_subgraphs: list[SourceGroundedSubgraph] | None = None,
        pseudonymize: bool = True,
    ) -> None:
        super().__init__(seed=seed)
        self.source_subgraphs = source_subgraphs or built_in_source_subgraphs()
        self.pseudonymize = pseudonymize
        self._by_domain = _index_by_domain(self.source_subgraphs)
        self._uses_by_source_id: defaultdict[str, int] = defaultdict(int)

    def _generate_scenario(self, index: int, spec: DomainSpec) -> dict[str, object]:
        source = self._select_source_subgraph(index=index, spec=spec)
        claims = _scenario_claims_for_benchmark(source=source, index=index)
        scenario_id = f"sg_{index:04d}"
        update_behavior = _update_behavior_for_index(index)
        context = self._source_entity(
            scenario_id=scenario_id,
            suffix=f"ctx_{index}",
            label=self._context_name_for_source(index=index, spec=spec, source=source),
            entity_type=spec.context_type,
            domain=spec.domain,
        )
        obj = self._source_entity(
            scenario_id=scenario_id,
            suffix=f"obj_{index}",
            label=self._object_name_for_source(index=index, spec=spec, source=source),
            entity_type=spec.object_type,
            domain=spec.domain,
        )
        hard_negative_context = self._source_entity(
            scenario_id=scenario_id,
            suffix=f"negctx_{index}",
            label=self._alternate_context_name(index=index, spec=spec, source=source),
            entity_type=spec.context_type,
            domain=spec.domain,
        )
        answer_entities = [
            self._source_entity(
                scenario_id=scenario_id,
                suffix=f"ans_{claim_index}",
                label=self._answer_name_for_source(
                    index=index,
                    claim_index=claim_index,
                    spec=spec,
                    source=source,
                    claim=claim,
                ),
                entity_type=spec.answer_type,
                domain=spec.domain,
            )
            for claim_index, claim in enumerate(claims)
        ]
        entities = [context, obj, hard_negative_context, *answer_entities]
        facts = self._source_facts(
            scenario_id=scenario_id,
            spec=spec,
            source=source,
            claims=claims,
            context=context,
            obj=obj,
            hard_negative_context=hard_negative_context,
            answer_entities=answer_entities,
            update_behavior=update_behavior,
        )
        lookup = entity_lookup(entities)
        facts = [
            fact.model_copy(update={"canonical_evidence": fact_sentence(fact, lookup)})
            for fact in facts
        ]
        snapshots = _snapshots_for_source_scenario(scenario_id=scenario_id, facts=facts)
        return {
            "scenario_id": scenario_id,
            "entities": entities,
            "facts": facts,
            "snapshots": snapshots,
            "update_behavior": update_behavior,
            "source_subgraph_id": source.source_id,
            "source_family": source.source_family,
            "source_fidelity": source.source_fidelity,
            "source_record_ids": source.source_record_ids,
            "source_revision": source.source_revision,
            "source_relation": source.source_relation,
            "source_path_relation": source.source_path_relation,
            "source_topology": source.topology_signature,
            "source_notes": source.notes,
        }

    def _select_source_subgraph(self, *, index: int, spec: DomainSpec) -> SourceGroundedSubgraph:
        candidates = self._by_domain.get(spec.domain)
        if not candidates:
            raise ValueError(f"No source-grounded subgraphs configured for {spec.domain}")
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                self._uses_by_source_id[candidate.source_id],
                -candidate.score(),
                _stable_key(self.seed, index, candidate.source_id),
            ),
        )
        selected = ordered[0]
        self._uses_by_source_id[selected.source_id] += 1
        return selected

    def _source_facts(
        self,
        *,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        claims: list[SourceTemporalClaim],
        context: Entity,
        obj: Entity,
        hard_negative_context: Entity,
        answer_entities: list[Entity],
        update_behavior: str,
    ) -> list[Fact]:
        roles = [
            FactRole.STALE_DISTRACTOR,
            FactRole.STALE_DISTRACTOR,
            FactRole.STALE_DISTRACTOR,
            FactRole.VALID_SUPPORT,
            FactRole.FUTURE_DISTRACTOR,
        ]
        facts: list[Fact] = []
        for claim_index, claim in enumerate(claims):
            interval = claim.interval()
            facts.append(
                Fact(
                    fact_id=stable_opaque_id("f", scenario_id, "timeline", claim_index),
                    scenario_id=scenario_id,
                    subject_id=answer_entities[claim_index].entity_id,
                    relation=spec.relation,
                    object_id=obj.entity_id,
                    context_id=context.entity_id,
                    valid_time=interval,
                    publication_time=(
                        date(2023, 11, 15) if claim_index == 4 else _publication_date(interval)
                    ),
                    transaction_time=(
                        date(2023, 12, 1) if claim_index == 4 else _transaction_date(interval)
                    ),
                    snapshot_visible_from=_snapshot_visible_from_for_timeline(
                        update_behavior=update_behavior,
                        candidate_index=claim_index,
                    ),
                    source_type=f"{source.source_family}:{source.source_relation}",
                    provenance_reliability="high",
                    fact_role=roles[claim_index],
                    canonical_evidence="",
                )
            )
        facts.extend(
            [
                self._conflict_fact(scenario_id, spec, source, context, obj, answer_entities[2]),
                self._unknown_time_fact(
                    scenario_id, spec, source, context, obj, answer_entities[1]
                ),
                self._publication_only_fact(
                    scenario_id, spec, source, context, obj, answer_entities[0]
                ),
                self._hard_negative_fact(
                    scenario_id,
                    spec,
                    source,
                    hard_negative_context,
                    obj,
                    answer_entities[3],
                ),
                self._historical_support_fact(
                    scenario_id,
                    spec,
                    source,
                    context,
                    obj,
                    answer_entities[3],
                ),
                self._bad_path_fact(
                    scenario_id,
                    spec,
                    source,
                    hard_negative_context,
                    obj,
                    answer_entities[2],
                ),
            ]
        )
        return facts

    def _conflict_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "conflict"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=make_interval(2024, None),
            publication_time=date(2024, 5, 1),
            transaction_time=date(2024, 5, 2),
            snapshot_visible_from="S1",
            source_type=f"{source.source_family}:conflicting_source",
            provenance_reliability="low",
            fact_role=FactRole.CONTRADICTORY,
            canonical_evidence="",
        )

    def _unknown_time_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "unknown"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=unknown_interval(),
            publication_time=date(2022, 9, 1),
            transaction_time=date(2022, 9, 1),
            snapshot_visible_from="S0",
            source_type=f"{source.source_family}:undated_statement",
            provenance_reliability="medium",
            fact_role=FactRole.UNKNOWN_TIME,
            canonical_evidence="",
        )

    def _publication_only_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "publication_only"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=unknown_interval(),
            publication_time=date(2024, 4, 20),
            transaction_time=date(2024, 4, 21),
            snapshot_visible_from="S1",
            source_type=f"{source.source_family}:publication_only",
            provenance_reliability="medium",
            fact_role=FactRole.PUBLICATION_ONLY,
            canonical_evidence="",
        )

    def _hard_negative_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "hard_negative"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=make_interval(2024, None),
            publication_time=date(2024, 1, 5),
            transaction_time=date(2024, 1, 5),
            snapshot_visible_from="S1",
            source_type=f"{source.source_family}:neighbor_context",
            provenance_reliability="high",
            fact_role=FactRole.HARD_NEGATIVE,
            canonical_evidence="",
        )

    def _historical_support_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "historical_same_answer"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=make_interval(2019, 2020),
            publication_time=date(2020, 1, 8),
            transaction_time=date(2020, 1, 8),
            snapshot_visible_from="S0",
            source_type=f"{source.source_family}:{source.source_relation}",
            provenance_reliability="high",
            fact_role=FactRole.BACKGROUND,
            canonical_evidence="",
        )

    def _bad_path_fact(
        self,
        scenario_id: str,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        context: Entity,
        obj: Entity,
        answer: Entity,
    ) -> Fact:
        return Fact(
            fact_id=stable_opaque_id("f", scenario_id, "path_bad"),
            scenario_id=scenario_id,
            subject_id=answer.entity_id,
            relation=spec.relation,
            object_id=obj.entity_id,
            context_id=context.entity_id,
            valid_time=make_interval(2019, 2020),
            publication_time=date(2020, 1, 8),
            transaction_time=date(2020, 1, 8),
            snapshot_visible_from="S0",
            source_type=f"{source.source_family}:{source.source_relation}",
            provenance_reliability="high",
            fact_role=FactRole.GRAPH_INCOHERENT,
            canonical_evidence="",
        )

    def _context_name_for_source(
        self,
        *,
        index: int,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
    ) -> str:
        if not self.pseudonymize:
            return source.context_label
        return _natural_context_name(index=index, spec=spec)

    def _object_name_for_source(
        self,
        *,
        index: int,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
    ) -> str:
        if not self.pseudonymize:
            return source.object_label
        return _natural_object_name(index=index, spec=spec, source=source)

    def _alternate_context_name(
        self,
        *,
        index: int,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
    ) -> str:
        if not self.pseudonymize:
            return source.hard_negative_context_label
        base = _natural_context_name(index=index, spec=spec)
        suffix = {
            EntityType.TEAM: "Academy",
            EntityType.PROJECT: "Pilot",
            EntityType.PRODUCT: "Preview",
        }.get(spec.context_type, "Division")
        return f"{base} {suffix}"

    def _answer_name_for_source(
        self,
        *,
        index: int,
        claim_index: int,
        spec: DomainSpec,
        source: SourceGroundedSubgraph,
        claim: SourceTemporalClaim,
    ) -> str:
        if not self.pseudonymize:
            return claim.label
        return _natural_answer_name(index=index, claim_index=claim_index, spec=spec, source=source)

    @staticmethod
    def _source_entity(
        *,
        scenario_id: str,
        suffix: str,
        label: str,
        entity_type: EntityType,
        domain: str,
    ) -> Entity:
        return Entity(
            entity_id=f"e_{scenario_id}_{suffix}",
            name=label,
            entity_type=entity_type,
            aliases=[],
            domain=domain,
        )


def load_source_subgraphs(path: Path) -> list[SourceGroundedSubgraph]:
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    return [SourceGroundedSubgraph.model_validate(row) for row in rows]


def built_in_source_subgraphs() -> list[SourceGroundedSubgraph]:
    return [_source_subgraph(row) for row in _SOURCE_SUBGRAPH_ROWS]


def _index_by_domain(
    source_subgraphs: list[SourceGroundedSubgraph],
) -> dict[str, list[SourceGroundedSubgraph]]:
    by_domain: dict[str, list[SourceGroundedSubgraph]] = defaultdict(list)
    for subgraph in source_subgraphs:
        by_domain[subgraph.domain].append(subgraph)
    return by_domain


def _source_subgraph(row: dict[str, object]) -> SourceGroundedSubgraph:
    return SourceGroundedSubgraph.model_validate(row)


def _ordered_claims_for_benchmark(
    source: SourceGroundedSubgraph,
) -> list[SourceTemporalClaim]:
    """Select five claims in the temporal shape expected by T-CRED questions.

    Upstream extractors may emit source claims in arbitrary order. The dataset generator needs a
    stable benchmark shape: three stale distractors, one current/valid support claim, and one
    future distractor. This preserves the extracted temporal ordering while making the downstream
    symbolic question programs reliable.
    """

    valid_pool = sorted(
        [
            claim
            for claim in source.claims
            if claim.role == "valid" or _claim_active_in_benchmark_present(claim)
        ],
        key=_claim_sort_key,
    )
    future_pool = sorted(
        [claim for claim in source.claims if claim.role == "future" or claim.start_year > 2024],
        key=_claim_sort_key,
    )
    if not valid_pool:
        raise ValueError(
            f"Source subgraph {source.source_id} needs a claim valid at the benchmark present."
        )
    if not future_pool:
        raise ValueError(f"Source subgraph {source.source_id} needs a future distractor claim.")

    valid_claim = valid_pool[0]
    future_claim = future_pool[0]
    reserved_ids = {id(valid_claim), id(future_claim)}
    stale_pool = sorted(
        [
            claim
            for claim in source.claims
            if id(claim) not in reserved_ids
            and (claim.role == "stale" or _claim_ended_before_benchmark_present(claim))
        ],
        key=_claim_sort_key,
    )
    if len(stale_pool) < 3:
        stale_pool.extend(
            claim
            for claim in sorted(source.claims, key=_claim_sort_key)
            if id(claim) not in reserved_ids and claim not in stale_pool
        )
    if len(stale_pool) < 3:
        raise ValueError(
            f"Source subgraph {source.source_id} needs at least three stale/background claims."
        )
    return [*stale_pool[:3], valid_claim, future_claim]


def _scenario_claims_for_benchmark(
    *,
    source: SourceGroundedSubgraph,
    index: int,
) -> list[SourceTemporalClaim]:
    ordered = _ordered_claims_for_benchmark(source)
    roles: tuple[Literal["stale", "valid", "future"], ...] = (
        "stale",
        "stale",
        "stale",
        "valid",
        "future",
    )
    if source.source_fidelity == "source_extracted":
        return [
            claim.model_copy(update={"role": role})
            for claim, role in zip(ordered, roles, strict=True)
        ]
    years = _timeline_year_profile(index)
    return [
        claim.model_copy(
            update={
                "start_year": start_year,
                "end_year": end_year,
                "role": role,
            }
        )
        for claim, (start_year, end_year), role in zip(ordered, years, roles, strict=True)
    ]


def _claim_active_in_benchmark_present(claim: SourceTemporalClaim) -> bool:
    return claim.start_year <= 2024 and (claim.end_year is None or claim.end_year >= 2024)


def _claim_ended_before_benchmark_present(claim: SourceTemporalClaim) -> bool:
    return claim.end_year is not None and claim.end_year < 2024


def _claim_sort_key(claim: SourceTemporalClaim) -> tuple[int, int, str]:
    return (claim.start_year, claim.end_year or 9999, claim.label)


def _stable_key(seed: int, index: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{index}:{value}".encode()).hexdigest()


def _publication_date(interval: TemporalInterval) -> date | None:
    if interval.start is None:
        return None
    return date(interval.start.year, 2, 15)


def _transaction_date(interval: TemporalInterval) -> date | None:
    if interval.start is None:
        return None
    return date(interval.start.year, 3, 1)


def _snapshots_for_source_scenario(*, scenario_id: str, facts: list[Fact]) -> list[Snapshot]:
    s0_visible = [fact.fact_id for fact in facts if fact.snapshot_visible_from == "S0"]
    s1_visible = [fact.fact_id for fact in facts if fact_visible(fact, "S1")]
    return [
        Snapshot(
            scenario_id=scenario_id,
            snapshot_id="S0",
            snapshot_time=date(2023, 12, 31),
            visible_fact_ids=s0_visible,
            description="Source-grounded snapshot before the latest update.",
        ),
        Snapshot(
            scenario_id=scenario_id,
            snapshot_id="S1",
            snapshot_time=date(2024, 6, 1),
            visible_fact_ids=s1_visible,
            description="Source-grounded snapshot after current update visibility.",
        ),
    ]


def _natural_context_name(*, index: int, spec: DomainSpec) -> str:
    domain_ordinal = index // len(DOMAIN_SPECS)
    root_index = domain_ordinal % len(_CONTEXT_ROOTS)
    block = domain_ordinal // len(_CONTEXT_ROOTS)
    root = _CONTEXT_ROOTS[root_index]
    if block:
        root = f"{_CONTEXT_ROOTS[(root_index + block) % len(_CONTEXT_ROOTS)]} {root}"
    noun = spec.context_nouns[domain_ordinal % len(spec.context_nouns)]
    return f"{root} {noun}"


def _natural_object_name(
    *,
    index: int,
    spec: DomainSpec,
    source: SourceGroundedSubgraph,
) -> str:
    if spec.answer_type == EntityType.POLICY:
        return _POLICY_OBJECTS[index % len(_POLICY_OBJECTS)]
    if spec.answer_type == EntityType.CONTRACT:
        return _CONTRACT_OBJECTS[index % len(_CONTRACT_OBJECTS)]
    if spec.answer_type == EntityType.PRODUCT_VERSION:
        return _VERSION_OBJECTS[index % len(_VERSION_OBJECTS)]
    if spec.answer_type == EntityType.EVENT:
        return _EVENT_OBJECTS[index % len(_EVENT_OBJECTS)]
    return source.object_label


def _natural_answer_name(
    *,
    index: int,
    claim_index: int,
    spec: DomainSpec,
    source: SourceGroundedSubgraph,
) -> str:
    offset = index + claim_index
    if spec.answer_type == EntityType.PERSON:
        first = FIRST_NAMES[offset % len(FIRST_NAMES)]
        last = LAST_NAMES[(index * 5 + claim_index) % len(LAST_NAMES)]
        return f"{first} {last}"
    if spec.answer_type == EntityType.POLICY:
        return _POLICY_NAMES[offset % len(_POLICY_NAMES)]
    if spec.answer_type == EntityType.CONTRACT:
        return _CONTRACT_NAMES[offset % len(_CONTRACT_NAMES)]
    if spec.answer_type == EntityType.PRODUCT_VERSION:
        return _VERSION_NAMES[offset % len(_VERSION_NAMES)]
    if spec.answer_type == EntityType.LOCATION:
        return _LOCATION_NAMES[offset % len(_LOCATION_NAMES)]
    if spec.answer_type == EntityType.EVENT:
        return _EVENT_NAMES[offset % len(_EVENT_NAMES)]
    return source.claims[claim_index].label


_POLICY_OBJECTS = (
    "appeals procedure",
    "public access rule",
    "safety standard",
    "records policy",
)
_CONTRACT_OBJECTS = (
    "service agreement",
    "maintenance agreement",
    "licensing agreement",
    "support agreement",
)
_VERSION_OBJECTS = (
    "supported release",
    "stable release",
    "security release",
    "long-term release",
)
_EVENT_OBJECTS = ("milestone", "review", "deployment", "audit")
_POLICY_NAMES = (
    "Open Records Rule",
    "Appeals Standard",
    "Safety Review Policy",
    "Access Notice Rule",
    "Public Records Code",
)
_CONTRACT_NAMES = (
    "Harbor Services Agreement",
    "Northbridge Support Agreement",
    "Aster Licensing Agreement",
    "Summit Maintenance Agreement",
    "Cedar Services Agreement",
)
_VERSION_NAMES = (
    "Aurora Release",
    "Beacon Release",
    "Cascade Release",
    "Dawn Release",
    "Ember Release",
)
_LOCATION_NAMES = tuple(f"{location} Campus" for location in LOCATIONS)
_EVENT_NAMES = (
    "Planning Review",
    "Public Hearing",
    "Launch Meeting",
    "Final Audit",
    "Budget Vote",
)

_CONTEXT_ROOTS = (
    "Alder",
    "Amber",
    "Arden",
    "Ashford",
    "Beacon",
    "Briar",
    "Brookfield",
    "Calder",
    "Cedar",
    "Clearview",
    "Clover",
    "Coral",
    "Crestwood",
    "Dawn",
    "Delta",
    "Easton",
    "Elmwood",
    "Fairview",
    "Fern",
    "Glenwood",
    "Grandview",
    "Greenfield",
    "Harbor",
    "Hawthorne",
    "Hazel",
    "Highfield",
    "Hillcrest",
    "Juniper",
    "Keystone",
    "Lakewood",
    "Linden",
    "Maple",
    "Meadow",
    "Meridian",
    "Millstone",
    "Northfield",
    "Oakridge",
    "Orchard",
    "Parkview",
    "Pinehurst",
    "Redfield",
    "Redwood",
    "Ridgeway",
    "Riverbend",
    "Riverton",
    "Rosewood",
    "Silverton",
    "Solstice",
    "Southpoint",
    "Springfield",
    "Stonebridge",
    "Stonefield",
    "Summit",
    "Sunnyside",
    "Timberline",
    "Trillium",
    "Valewood",
    "Vantage",
    "Westfield",
    "Willow",
    "Windham",
    "Woodbridge",
    "Auburn",
    "Bayview",
    "Birchwood",
    "Brighton",
    "Cypress",
    "Evergreen",
    "Foxglove",
    "Goldcrest",
    "Heather",
    "Ironwood",
    "Kingswell",
    "Longford",
    "Magnolia",
    "Norwood",
    "Palisade",
    "Queensbridge",
    "Rockwell",
    "Sycamore",
)


_SOURCE_SUBGRAPH_ROWS: tuple[dict[str, object], ...] = (
    {
        "source_id": "wikidata_p39_office_terms",
        "source_family": "wikidata",
        "domain": "corporate_roles",
        "relation": "held_role",
        "path_relation": "affiliated_with",
        "source_relation": "P39-position-held",
        "source_path_relation": "P463-member-of",
        "context_label": "municipal council",
        "object_label": "director",
        "hard_negative_context_label": "regional council",
        "claims": [
            {"label": "officeholder one", "start_year": 2015, "end_year": 2017},
            {"label": "officeholder two", "start_year": 2018, "end_year": 2020},
            {"label": "officeholder three", "start_year": 2021, "end_year": 2023},
            {"label": "officeholder four", "start_year": 2024, "end_year": None, "role": "valid"},
            {
                "label": "announced successor",
                "start_year": 2027,
                "end_year": None,
                "role": "future",
            },
        ],
        "degree_hint": 32,
        "topology_signature": "office-term-succession",
        "notes": (
            "P39-style office timeline with start/end qualifiers and a neighboring "
            "organization edge."
        ),
    },
    {
        "source_id": "wikidata_p54_team_membership",
        "source_family": "wikidata",
        "domain": "sports_memberships",
        "relation": "member_of",
        "path_relation": "affiliated_with",
        "source_relation": "P54-member-of-sports-team",
        "source_path_relation": "P17-country",
        "context_label": "football club",
        "object_label": "captain",
        "hard_negative_context_label": "rival club",
        "claims": [
            {"label": "former captain", "start_year": 2015, "end_year": 2017},
            {"label": "loan captain", "start_year": 2018, "end_year": 2020},
            {"label": "interim captain", "start_year": 2021, "end_year": 2023},
            {"label": "current captain", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "future signing", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 55,
        "topology_signature": "team-membership-transfer",
        "notes": "P54-style roster timeline with adjacent club or country membership context.",
    },
    {
        "source_id": "registry_policy_effective_periods",
        "source_family": "domain_registry",
        "domain": "policies",
        "relation": "policy_effective",
        "path_relation": "document_supports",
        "source_relation": "policy-effective-period",
        "source_path_relation": "document-supports",
        "context_label": "public agency",
        "object_label": "records policy",
        "hard_negative_context_label": "appeals office",
        "claims": [
            {"label": "older policy", "start_year": 2015, "end_year": 2017},
            {"label": "interim policy", "start_year": 2018, "end_year": 2020},
            {"label": "review policy", "start_year": 2021, "end_year": 2023},
            {"label": "active policy", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "scheduled policy", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 18,
        "topology_signature": "policy-supersession",
        "notes": (
            "Effective-period policy register pattern used when public KGs lack "
            "explicit policy lifecycle predicates."
        ),
    },
    {
        "source_id": "registry_contract_lifecycle",
        "source_family": "domain_registry",
        "domain": "contracts",
        "relation": "contract_active",
        "path_relation": "affiliated_with",
        "source_relation": "contract-active-period",
        "source_path_relation": "supplier-affiliation",
        "context_label": "public service provider",
        "object_label": "service agreement",
        "hard_negative_context_label": "neighbor service provider",
        "claims": [
            {"label": "legacy agreement", "start_year": 2015, "end_year": 2017},
            {"label": "bridge agreement", "start_year": 2018, "end_year": 2020},
            {"label": "renewal agreement", "start_year": 2021, "end_year": 2023},
            {"label": "current agreement", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "planned agreement", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 20,
        "topology_signature": "contract-renewal",
        "notes": (
            "Procurement-style validity intervals; retained as source-grounded registry pattern."
        ),
    },
    {
        "source_id": "software_release_lifecycle",
        "source_family": "domain_registry",
        "domain": "product_versions",
        "relation": "support_window",
        "path_relation": "document_supports",
        "source_relation": "release-support-window",
        "source_path_relation": "release-note-supports",
        "context_label": "software platform",
        "object_label": "supported release",
        "hard_negative_context_label": "legacy platform",
        "claims": [
            {"label": "alpha release", "start_year": 2015, "end_year": 2017},
            {"label": "beta release", "start_year": 2018, "end_year": 2020},
            {"label": "stable release", "start_year": 2021, "end_year": 2023},
            {"label": "current release", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "preview release", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 14,
        "topology_signature": "version-lifecycle",
        "notes": "Release support windows modelled after software lifecycle tables.",
    },
    {
        "source_id": "eventkg_project_sequence",
        "source_family": "eventkg",
        "domain": "event_timelines",
        "relation": "event_occurs",
        "path_relation": "event_precedes",
        "source_relation": "sem:hasTime",
        "source_path_relation": "eventKG-succeedingEvent",
        "context_label": "public works project",
        "object_label": "milestone",
        "hard_negative_context_label": "separate project",
        "claims": [
            {"label": "planning meeting", "start_year": 2015, "end_year": 2017},
            {"label": "public hearing", "start_year": 2018, "end_year": 2020},
            {"label": "budget vote", "start_year": 2021, "end_year": 2023},
            {"label": "launch meeting", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "final audit", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 44,
        "topology_signature": "event-sequence",
        "notes": (
            "EventKG-style event sequence with event-time and predecessor/successor relations."
        ),
    },
    {
        "source_id": "wikidata_research_project_participants",
        "source_family": "wikidata",
        "domain": "research_projects",
        "relation": "project_participant",
        "path_relation": "affiliated_with",
        "source_relation": "P710-participant",
        "source_path_relation": "P1416-affiliation",
        "context_label": "research consortium",
        "object_label": "principal investigator",
        "hard_negative_context_label": "related study",
        "claims": [
            {"label": "former investigator", "start_year": 2015, "end_year": 2017},
            {"label": "data steward", "start_year": 2018, "end_year": 2020},
            {"label": "coordinator", "start_year": 2021, "end_year": 2023},
            {"label": "lead investigator", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "incoming lead", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 27,
        "topology_signature": "project-participation",
        "notes": (
            "Participant and affiliation subgraph pattern from Wikidata-like "
            "research project records."
        ),
    },
    {
        "source_id": "wikidata_location_headquarters",
        "source_family": "wikidata",
        "domain": "locations",
        "relation": "located_at",
        "path_relation": "affiliated_with",
        "source_relation": "P159-headquarters-location",
        "source_path_relation": "P131-located-in-admin-entity",
        "context_label": "public institute",
        "object_label": "headquarters",
        "hard_negative_context_label": "regional institute",
        "claims": [
            {"label": "old campus", "start_year": 2015, "end_year": 2017},
            {"label": "temporary campus", "start_year": 2018, "end_year": 2020},
            {"label": "archive campus", "start_year": 2021, "end_year": 2023},
            {"label": "current campus", "start_year": 2024, "end_year": None, "role": "valid"},
            {"label": "planned campus", "start_year": 2027, "end_year": None, "role": "future"},
        ],
        "degree_hint": 38,
        "topology_signature": "facility-location-history",
        "notes": (
            "Headquarters/location history pattern with adjacent administrative-location edge."
        ),
    },
)
