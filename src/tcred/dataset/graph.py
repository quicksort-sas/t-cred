from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tcred.dataset.models import Entity, Fact, GraphPathEdge, Relation

Record = Fact | GraphPathEdge | Entity | Mapping[str, Any]
EntityLookup = Mapping[str, Entity | Mapping[str, Any]]


@dataclass(frozen=True)
class RelationSemantics:
    """Graph-facing semantics for one normalized relation.

    ``reverse_storage`` exists only for legacy records. Certified records carry
    explicit graph endpoints and do not depend on this compatibility mapping.
    """

    symmetric: bool = False
    reverse_storage: bool = False


_RELATION_SEMANTICS: dict[str, RelationSemantics] = {
    str(Relation.AFFILIATED_WITH): RelationSemantics(symmetric=True),
    str(Relation.LOCATED_AT): RelationSemantics(reverse_storage=True),
    str(Relation.WIKIDATA_PROPERTY): RelationSemantics(reverse_storage=True),
    str(Relation.DOCUMENT_SUPPORTS): RelationSemantics(reverse_storage=True),
}


def relation_semantics(relation: str | Relation) -> RelationSemantics:
    return _RELATION_SEMANTICS.get(str(relation), RelationSemantics())


def fact_source_id(fact: Fact | Mapping[str, Any]) -> str:
    """Return the semantic source node for a stored fact."""
    explicit = str(_get(fact, "graph_source_id") or "")
    if explicit:
        return explicit
    subject = str(_get(fact, "subject_id") or "")
    context_or_object = str(_get(fact, "context_id") or _get(fact, "object_id") or "")
    semantics = relation_semantics(str(_get(fact, "relation") or ""))
    return context_or_object if semantics.reverse_storage else subject


def fact_target_id(fact: Fact | Mapping[str, Any]) -> str:
    """Return the semantic target node for a stored fact."""
    explicit = str(_get(fact, "graph_target_id") or "")
    if explicit:
        return explicit
    subject = str(_get(fact, "subject_id") or "")
    context_or_object = str(_get(fact, "context_id") or _get(fact, "object_id") or "")
    semantics = relation_semantics(str(_get(fact, "relation") or ""))
    return subject if semantics.reverse_storage else context_or_object


def fact_answer_id(fact: Fact | Mapping[str, Any]) -> str:
    """Return the entity contributed by a fact to the question answer."""
    return str(_get(fact, "answer_entity_id") or _get(fact, "subject_id") or "")


def fact_endpoint_ids(
    fact: Fact | Mapping[str, Any],
    *,
    traversal_direction: str = "forward",
) -> tuple[str, str]:
    source, target = fact_source_id(fact), fact_target_id(fact)
    if traversal_direction == "forward":
        return source, target
    if traversal_direction == "reverse":
        return target, source
    raise ValueError(f"Unsupported traversal direction: {traversal_direction!r}")


def fact_node_ids(fact: Fact | Mapping[str, Any], *, include_object: bool = True) -> list[str]:
    ids = [fact_source_id(fact), fact_target_id(fact)]
    object_id = str(_get(fact, "object_id") or "")
    if include_object and object_id:
        ids.append(object_id)
    return list(dict.fromkeys(item for item in ids if item))


def graph_path_node_ids(
    edges: list[GraphPathEdge | Mapping[str, Any]],
    fact_by_id: Mapping[str, Fact | Mapping[str, Any]],
    *,
    include_object: bool = False,
) -> list[str]:
    """Return path nodes in traversal order.

    Discontinuities are deliberately retained as an extra source/target pair so
    the validator and UI expose an invalid chain instead of silently turning it
    into an apparently valid set of nodes.
    """
    node_ids: list[str] = []
    for edge in edges:
        fact = fact_by_id.get(str(_get(edge, "fact_id") or ""))
        if fact is None:
            continue
        source_id, target_id = fact_endpoint_ids(
            fact,
            traversal_direction=str(_get(edge, "traversal_direction") or "forward"),
        )
        if not node_ids:
            node_ids.extend([source_id, target_id])
        elif node_ids[-1] == source_id:
            node_ids.append(target_id)
        else:
            node_ids.extend([source_id, target_id])
        if include_object:
            object_id = str(_get(fact, "object_id") or "")
            if object_id and object_id not in node_ids:
                node_ids.append(object_id)
    return [item for item in node_ids if item]


def node_payload(entity_id: str, entities: EntityLookup) -> dict[str, str]:
    entity = entities.get(entity_id)
    return {
        "id": entity_id,
        "label": str(_get(entity, "name") or entity_id) if entity else entity_id,
        "type": str(_get(entity, "entity_type") or "") if entity else "",
    }


def fact_edge_payload(
    fact: Fact | Mapping[str, Any],
    entities: EntityLookup,
    *,
    highlight: bool = False,
) -> dict[str, Any]:
    source_id, target_id = fact_endpoint_ids(fact)
    object_id = str(_get(fact, "object_id") or "")
    object_label = str(_get(entities.get(object_id), "name") or "") if object_id else ""
    relation = str(_get(fact, "relation") or "")
    semantics = relation_semantics(relation)
    explicit_direction = str(_get(fact, "relation_direction") or "")
    symmetric = explicit_direction == "symmetric" if explicit_direction else semantics.symmetric
    source_relation_label = str(_get(fact, "source_relation_label") or "")
    return {
        "id": _get(fact, "fact_id"),
        "fact_id": _get(fact, "fact_id"),
        "source": source_id,
        "target": target_id,
        "source_node": node_payload(source_id, entities),
        "target_node": node_payload(target_id, entities),
        "object_id": object_id,
        "object_label": object_label,
        "relation": relation,
        "relation_label": source_relation_label
        or relation_label(relation, object_label=object_label),
        "role": _get(fact, "fact_role") or "",
        "highlight": highlight,
        "valid_time": _get(fact, "valid_time") or {},
        "source_type": _get(fact, "source_type") or "",
        "evidence_text": _get(fact, "paraphrased_evidence")
        or _get(fact, "canonical_evidence")
        or "",
        "directional": not symmetric,
        "symmetric": symmetric,
        "traversal_direction": "forward",
    }


def path_edge_payload(
    edge: GraphPathEdge | Mapping[str, Any],
    fact_by_id: Mapping[str, Fact | Mapping[str, Any]],
    entities: EntityLookup,
) -> dict[str, Any]:
    fact = fact_by_id.get(str(_get(edge, "fact_id") or ""))
    payload = dict(edge) if isinstance(edge, Mapping) else edge.model_dump(mode="json")
    relation = str(_get(edge, "relation") or "")
    traversal_direction = str(_get(edge, "traversal_direction") or "forward")
    semantics = relation_semantics(relation)
    if fact is None:
        return {
            **payload,
            "source": node_payload("", entities),
            "target": node_payload("", entities),
            "relation_label": relation_label(relation),
            "directional": not semantics.symmetric,
            "symmetric": semantics.symmetric,
            "traversal_direction": traversal_direction,
        }

    source_id, target_id = fact_endpoint_ids(
        fact,
        traversal_direction=traversal_direction,
    )
    fact_payload = fact_edge_payload(fact, entities)
    symmetric = bool(fact_payload["symmetric"])
    label = str(fact_payload["relation_label"])
    if traversal_direction == "reverse" and not symmetric:
        label = f"reverse traversal of {label}"
    return {
        **payload,
        "source": node_payload(source_id, entities),
        "target": node_payload(target_id, entities),
        "object_id": fact_payload["object_id"],
        "object_label": fact_payload["object_label"],
        "relation_label": label,
        "evidence_text": fact_payload["evidence_text"],
        "source_type": fact_payload["source_type"],
        "directional": not symmetric,
        "symmetric": symmetric,
        "traversal_direction": traversal_direction,
    }


def relation_label(relation: str | Relation, *, object_label: str = "") -> str:
    base = {
        "held_role": "held role",
        "member_of": "member of",
        "employed_by": "employed by",
        "political_affiliation": "political affiliation",
        "policy_effective": "effective policy",
        "contract_active": "active contract",
        "product_version": "product version",
        "support_window": "support window",
        "located_at": "located at",
        "event_occurs": "event occurs",
        "event_precedes": "precedes",
        "project_participant": "project participant",
        "affiliated_with": "affiliated with",
        "wikidata_property": "has source answer",
        "document_supports": "supports answer",
    }.get(str(relation), str(relation).replace("_", " "))
    return f"{base}: {object_label}" if object_label else base


def _get(record: Record | None, field: str) -> Any:
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(field)
    return getattr(record, field, None)
