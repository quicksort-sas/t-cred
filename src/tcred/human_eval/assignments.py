from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from tcred.human_eval.complexity import (
    annotation_complexity,
    annotation_complexity_breakdown,
    complexity_manifest,
)
from tcred.human_eval.sampling import proportional_quotas

DEFAULT_ASSIGNMENT_SEED = 20260721
MAX_ANNOTATIONS_PER_UNIT = 3
BALANCE_SEARCH_ATTEMPTS_PER_ASSIGNMENT = 2_000
_BALANCE_WEIGHTS = {
    "family": 500,
    "source": 350,
    "system": 250,
    "path": 40,
    "refusal": 40,
    "detail": 10,
}


@dataclass(frozen=True)
class AnnotationAssignmentPlan:
    assignments: dict[str, list[dict[str, object]]]
    second_coded_unit_ids: tuple[str, ...]
    third_coded_unit_ids: tuple[str, ...]
    question_uniqueness_feasible: bool
    question_uniqueness_achieved: bool
    seed: int

    @property
    def overlap_unit_ids(self) -> tuple[str, ...]:
        """Units assigned to at least two distinct annotators."""
        return self.second_coded_unit_ids

    @property
    def annotation_counts(self) -> dict[str, int]:
        counts = Counter(str(row["unit_id"]) for rows in self.assignments.values() for row in rows)
        return dict(sorted(counts.items()))


def assignment_manifest_metadata(
    plan: AnnotationAssignmentPlan,
    *,
    key_rows: list[dict[str, object]],
) -> dict[str, object]:
    key_by_id = {str(row["unit_id"]): row for row in key_rows}
    counts = plan.annotation_counts
    if set(counts) != set(key_by_id):
        raise ValueError("Assignment rows and private keys contain different unit populations")
    overlap_keys = [key_by_id[unit_id] for unit_id in plan.overlap_unit_ids]
    third_keys = [key_by_id[unit_id] for unit_id in plan.third_coded_unit_ids]
    multiplicities = Counter(counts.values())
    third_ids_digest = hashlib.sha256("\n".join(plan.third_coded_unit_ids).encode()).hexdigest()
    balance = assignment_balance_report(plan, key_rows=key_rows)
    return {
        "assignment_seed": plan.seed,
        "annotation_multiplicity_counts": {
            str(count): units for count, units in sorted(multiplicities.items())
        },
        "minimum_annotations_per_unit": min(counts.values()),
        "maximum_annotations_per_unit": max(counts.values()),
        "overlap_units": len(plan.overlap_unit_ids),
        "overlap_by_family": _key_breakdown(overlap_keys, "dataset_family"),
        "overlap_by_source": _key_breakdown(overlap_keys, "source_kind"),
        "triple_coded_units": len(plan.third_coded_unit_ids),
        "triple_coded_by_family": _key_breakdown(third_keys, "dataset_family"),
        "triple_coded_by_source": _key_breakdown(third_keys, "source_kind"),
        "third_coded_selection": {
            "method": "uniform_without_replacement",
            "population_units": len(key_rows),
            "sample_units": len(plan.third_coded_unit_ids),
            "seed": plan.seed,
            "selected_unit_ids_sha256": third_ids_digest,
        },
        "assignment_order": {
            "method": "ascending_annotation_complexity_v2_with_seeded_ties",
            **complexity_manifest(),
            "tie_break": "seeded_sha256_per_annotator",
            "purpose": "gradual familiarization before higher-load cards",
        },
        "assignment_complexity": _assignment_complexity_report(plan.assignments),
        "question_uniqueness": {
            "feasible": plan.question_uniqueness_feasible,
            "achieved": plan.question_uniqueness_achieved,
        },
        "assignment_balance": balance,
    }


def assign_annotation_units(
    *,
    unit_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
    annotators: int,
    assignments_per_annotator: int,
    seed: int = DEFAULT_ASSIGNMENT_SEED,
) -> AnnotationAssignmentPlan:
    if annotators < 1 or assignments_per_annotator < 1:
        raise ValueError("annotators and assignments_per_annotator must be positive")
    if not unit_rows:
        raise ValueError("At least one human-evaluation unit is required")
    public_unit_ids = [str(row["unit_id"]) for row in unit_rows]
    private_unit_ids = [str(row["unit_id"]) for row in key_rows]
    if len(set(public_unit_ids)) != len(public_unit_ids):
        raise ValueError("Public human-evaluation units contain duplicate unit ids")
    if len(set(private_unit_ids)) != len(private_unit_ids):
        raise ValueError("Private human-evaluation keys contain duplicate unit ids")
    capacity = annotators * assignments_per_annotator
    if capacity < len(unit_rows):
        raise ValueError("Assignment capacity must cover every unique human-evaluation unit")
    maximum_capacity = min(MAX_ANNOTATIONS_PER_UNIT, annotators) * len(unit_rows)
    if capacity > maximum_capacity:
        raise ValueError(
            "Assignment capacity exceeds three distinct annotations per unit "
            "or the number of available annotators"
        )
    key_by_id = dict(zip(private_unit_ids, key_rows, strict=True))
    unit_by_id = dict(zip(public_unit_ids, unit_rows, strict=True))
    if set(key_by_id) != set(unit_by_id):
        raise ValueError("Public units and private keys must contain identical unit ids")

    annotator_ids = [f"annotator_{index:02d}" for index in range(1, annotators + 1)]
    bins: dict[str, list[dict[str, object]]] = {annotator_id: [] for annotator_id in annotator_ids}
    membership: dict[str, set[str]] = {annotator_id: set() for annotator_id in annotator_ids}
    qid_membership: dict[str, set[str]] = {annotator_id: set() for annotator_id in annotator_ids}
    stratum_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    primary = sorted(
        unit_rows,
        key=lambda row: (
            _stratum(key_by_id[str(row["unit_id"])]),
            _hash(f"{seed}:primary-order:{row['unit_id']}"),
        ),
    )
    for row in primary:
        unit_id = str(row["unit_id"])
        key = key_by_id[unit_id]
        qid = str(key.get("qid", ""))
        stratum = _stratum(key)
        candidates = [
            annotator_id
            for annotator_id in annotator_ids
            if len(bins[annotator_id]) < assignments_per_annotator
            and (not qid or qid not in qid_membership[annotator_id])
        ]
        if not candidates:
            candidates = [
                annotator_id
                for annotator_id in annotator_ids
                if len(bins[annotator_id]) < assignments_per_annotator
            ]
        if not candidates:
            raise RuntimeError("Primary assignment exhausted annotator capacity")
        annotator_id = min(
            candidates,
            key=lambda candidate: (
                stratum_counts[(candidate, stratum)],
                len(bins[candidate]),
                _hash(f"{seed}:primary:{unit_id}:{candidate}"),
            ),
        )
        _append(
            bins,
            membership,
            qid_membership,
            annotator_id,
            row,
            qid=qid,
        )
        stratum_counts[(annotator_id, stratum)] += 1

    remaining_capacity = capacity - len(unit_rows)
    second_count = min(len(unit_rows), remaining_capacity)
    second_coded_ids = _stratified_overlap_ids(
        key_rows,
        second_count,
        annotators=annotators,
        seed=seed,
    )
    second_placement = _match_additional_copies(
        unit_ids=second_coded_ids,
        annotator_ids=annotator_ids,
        bins=bins,
        membership=membership,
        qid_membership=qid_membership,
        key_by_id=key_by_id,
        assignments_per_annotator=assignments_per_annotator,
        seed=seed,
        layer="second",
    )
    for unit_id, annotator_id in second_placement.items():
        key = key_by_id[unit_id]
        _append(
            bins,
            membership,
            qid_membership,
            annotator_id,
            unit_by_id[unit_id],
            qid=str(key.get("qid", "")),
        )

    third_count = max(0, capacity - 2 * len(unit_rows))
    third_coded_ids = _uniform_random_sample(
        public_unit_ids,
        target=third_count,
        seed=seed,
    )
    third_placement = _match_additional_copies(
        unit_ids=third_coded_ids,
        annotator_ids=annotator_ids,
        bins=bins,
        membership=membership,
        qid_membership=qid_membership,
        key_by_id=key_by_id,
        assignments_per_annotator=assignments_per_annotator,
        seed=seed,
        layer="third",
    )
    for unit_id, annotator_id in third_placement.items():
        key = key_by_id[unit_id]
        _append(
            bins,
            membership,
            qid_membership,
            annotator_id,
            unit_by_id[unit_id],
            qid=str(key.get("qid", "")),
        )

    strict_question_uniqueness = _question_uniqueness_is_feasible(
        bins,
        key_by_id=key_by_id,
        assignments_per_annotator=assignments_per_annotator,
        annotator_count=annotators,
    )
    repaired = _repair_duplicate_questions(
        bins,
        key_by_id=key_by_id,
        seed=seed,
    )
    if strict_question_uniqueness and not repaired:
        raise RuntimeError(
            "Could not produce question-unique annotator workloads despite feasible capacity"
        )
    _rebalance_bins(
        bins,
        key_by_id=key_by_id,
        seed=seed,
    )

    assignments: dict[str, list[dict[str, object]]] = {}
    for annotator_id, rows in bins.items():
        if len(rows) != assignments_per_annotator:
            raise RuntimeError(
                f"{annotator_id} received {len(rows)} rows; expected {assignments_per_annotator}"
            )
        ordered = _progressive_order(rows, seed=f"{seed}:{annotator_id}")
        assignments[annotator_id] = [
            _assignment_row(
                row,
                annotator_id=annotator_id,
                assignment_index=index,
            )
            for index, row in enumerate(ordered, start=1)
        ]
    plan = AnnotationAssignmentPlan(
        assignments=assignments,
        second_coded_unit_ids=tuple(sorted(second_coded_ids)),
        third_coded_unit_ids=tuple(sorted(third_coded_ids)),
        question_uniqueness_feasible=strict_question_uniqueness,
        question_uniqueness_achieved=repaired,
        seed=seed,
    )
    _validate_plan(
        plan,
        expected_unit_ids=set(public_unit_ids),
        key_by_id=key_by_id,
    )
    return plan


def assignment_balance_report(
    plan: AnnotationAssignmentPlan,
    *,
    key_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Describe realized per-annotator balance without exposing it to annotators."""
    key_by_id = {str(row["unit_id"]): row for row in key_rows}
    annotator_count = len(plan.assignments)
    grouped: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for annotator_id, rows in plan.assignments.items():
        for row in rows:
            key = key_by_id[str(row["unit_id"])]
            for kind, value in _core_balance_features(row=row, key=key):
                grouped[kind][f"{annotator_id}\0{value}"] += 1

    categories: dict[str, dict[str, object]] = {}
    all_within_ideal_bounds = True
    for kind in ("family", "source", "system"):
        values = sorted({encoded.split("\0", maxsplit=1)[1] for encoded in grouped[kind]})
        kind_report: dict[str, object] = {}
        for value in values:
            counts = [
                grouped[kind][f"{annotator_id}\0{value}"] for annotator_id in plan.assignments
            ]
            total = sum(counts)
            lower = total // annotator_count
            upper = lower + int(total % annotator_count > 0)
            within_bounds = min(counts) >= lower and max(counts) <= upper
            all_within_ideal_bounds &= within_bounds
            kind_report[value] = {
                "total_placements": total,
                "ideal_per_annotator": [lower, upper],
                "observed_per_annotator": [min(counts), max(counts)],
                "within_ideal_bounds": within_bounds,
            }
        categories[kind] = kind_report
    hard_constraints = [
        "fixed_unit_multiplicity",
        "distinct_annotators_per_unit",
        "fixed_assignments_per_annotator",
    ]
    if plan.question_uniqueness_achieved:
        hard_constraints.append("one_copy_of_each_question_per_annotator")
    return {
        "method": "seeded_global_swap_optimization",
        "objectives": [
            "dataset_family",
            "answer_source",
            "qa_system",
            "graph_path_presence",
            "refusal_field_presence",
            "fine_sampling_stratum",
        ],
        "hard_constraints_preserved": hard_constraints,
        "core_categories": categories,
        "all_core_categories_within_ideal_bounds": all_within_ideal_bounds,
    }


def _rebalance_bins(
    bins: dict[str, list[dict[str, object]]],
    *,
    key_by_id: dict[str, dict[str, object]],
    seed: int,
) -> None:
    """Reduce annotator-stratum confounding with deterministic valid swaps.

    The overlap sample and every unit's multiplicity are already fixed when this runs. A swap can
    therefore improve workload composition without changing which units receive second or third
    ratings. Unit and question membership checks keep the assignment's independence constraints
    intact.
    """
    annotator_ids = sorted(bins)
    if len(annotator_ids) < 2:
        return
    rows_by_unit = {str(row["unit_id"]): row for rows in bins.values() for row in rows}
    features_by_unit = {
        unit_id: _balance_features(row=row, key=key_by_id[unit_id])
        for unit_id, row in rows_by_unit.items()
    }
    total_feature_counts: Counter[tuple[str, str]] = Counter()
    annotator_feature_counts: dict[str, Counter[tuple[str, str]]] = {
        annotator_id: Counter() for annotator_id in annotator_ids
    }
    unit_membership: dict[str, set[str]] = {}
    qid_membership: dict[str, set[str]] = {}
    for annotator_id, rows in bins.items():
        units = {str(row["unit_id"]) for row in rows}
        qids = {qid for unit_id in units if (qid := str(key_by_id[unit_id].get("qid", "")))}
        unit_membership[annotator_id] = units
        qid_membership[annotator_id] = qids
        for unit_id in units:
            features = features_by_unit[unit_id]
            annotator_feature_counts[annotator_id].update(features)
            total_feature_counts.update(features)

    annotator_count = len(annotator_ids)
    rng = random.Random(f"{seed}:global-assignment-balance")
    total_assignments = sum(len(rows) for rows in bins.values())
    attempts = max(
        50_000,
        total_assignments * BALANCE_SEARCH_ATTEMPTS_PER_ASSIGNMENT,
    )
    no_improvement = 0
    for _ in range(attempts):
        left_id, right_id = rng.sample(annotator_ids, 2)
        left_index = rng.randrange(len(bins[left_id]))
        right_index = rng.randrange(len(bins[right_id]))
        left_row = bins[left_id][left_index]
        right_row = bins[right_id][right_index]
        left_unit = str(left_row["unit_id"])
        right_unit = str(right_row["unit_id"])
        if left_unit == right_unit:
            continue
        left_qid = str(key_by_id[left_unit].get("qid", ""))
        right_qid = str(key_by_id[right_unit].get("qid", ""))
        if right_unit in unit_membership[left_id] or left_unit in unit_membership[right_id]:
            continue
        if right_qid and right_qid != left_qid and right_qid in qid_membership[left_id]:
            continue
        if left_qid and left_qid != right_qid and left_qid in qid_membership[right_id]:
            continue

        left_features = features_by_unit[left_unit]
        right_features = features_by_unit[right_unit]
        delta = _swap_balance_delta(
            left_counts=annotator_feature_counts[left_id],
            right_counts=annotator_feature_counts[right_id],
            left_features=left_features,
            right_features=right_features,
            totals=total_feature_counts,
            annotator_count=annotator_count,
        )
        if delta >= 0:
            no_improvement += 1
            if no_improvement >= max(25_000, total_assignments * 100):
                break
            continue

        no_improvement = 0
        bins[left_id][left_index], bins[right_id][right_index] = right_row, left_row
        _replace_membership(unit_membership[left_id], remove=left_unit, add=right_unit)
        _replace_membership(unit_membership[right_id], remove=right_unit, add=left_unit)
        if left_qid != right_qid:
            _replace_membership(qid_membership[left_id], remove=left_qid, add=right_qid)
            _replace_membership(qid_membership[right_id], remove=right_qid, add=left_qid)
        annotator_feature_counts[left_id].subtract(left_features)
        annotator_feature_counts[left_id].update(right_features)
        annotator_feature_counts[right_id].subtract(right_features)
        annotator_feature_counts[right_id].update(left_features)


def _swap_balance_delta(
    *,
    left_counts: Counter[tuple[str, str]],
    right_counts: Counter[tuple[str, str]],
    left_features: tuple[tuple[str, str], ...],
    right_features: tuple[tuple[str, str], ...],
    totals: Counter[tuple[str, str]],
    annotator_count: int,
) -> int:
    left_set = set(left_features)
    right_set = set(right_features)
    changed = left_set ^ right_set
    delta = 0
    for feature in changed:
        left_before = left_counts[feature]
        right_before = right_counts[feature]
        left_after = left_before - int(feature in left_set) + int(feature in right_set)
        right_after = right_before - int(feature in right_set) + int(feature in left_set)
        weight = _BALANCE_WEIGHTS[feature[0]]
        total = totals[feature]
        delta += weight * (
            _scaled_squared_error(left_after, total=total, annotators=annotator_count)
            + _scaled_squared_error(right_after, total=total, annotators=annotator_count)
            - _scaled_squared_error(left_before, total=total, annotators=annotator_count)
            - _scaled_squared_error(right_before, total=total, annotators=annotator_count)
        )
    return delta


def _scaled_squared_error(count: int, *, total: int, annotators: int) -> int:
    deviation = count * annotators - total
    return deviation * deviation


def _replace_membership(values: set[str], *, remove: str, add: str) -> None:
    if remove:
        values.remove(remove)
    if add:
        values.add(add)


def _balance_features(
    *,
    row: dict[str, object],
    key: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    features = list(_core_balance_features(row=row, key=key))
    features.extend(
        [
            ("path", "present" if row.get("graph_paths") else "absent"),
            (
                "refusal",
                "present"
                if "response_decision_appropriate" in row.get("applicable_fields", [])
                else "absent",
            ),
            ("detail", _stratum(key)),
        ]
    )
    return tuple(features)


def _core_balance_features(
    *,
    row: dict[str, object],
    key: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    family = str(key.get("dataset_family", "unknown"))
    source = str(key.get("source_kind", "unknown"))
    features = [("family", family), ("source", source)]
    if source == "system_output":
        features.append(("system", str(key.get("system_name", "unknown"))))
    return tuple(features)


def _stratified_overlap_ids(
    key_rows: list[dict[str, object]],
    target: int,
    *,
    annotators: int,
    seed: int,
) -> list[str]:
    if target == 0:
        return []
    groups: defaultdict[str, list[str]] = defaultdict(list)
    qid_by_unit: dict[str, str] = {}
    for row in key_rows:
        unit_id = str(row["unit_id"])
        groups[_stratum(row)].append(unit_id)
        qid_by_unit[unit_id] = str(row.get("qid", ""))
    qid_counts = defaultdict(int)
    for qid in qid_by_unit.values():
        if qid:
            qid_counts[qid] += 1

    def permits_strict_qid_placement(unit_id: str) -> bool:
        qid = qid_by_unit[unit_id]
        return not qid or qid_counts[qid] < annotators

    quotas = proportional_quotas(
        {key: float(len(values)) for key, values in groups.items()}, target
    )
    selected: list[str] = []
    used_qids: set[str] = set()
    for key in sorted(groups):
        values = sorted(
            groups[key],
            key=lambda unit_id: _hash(f"{seed}:overlap:{unit_id}"),
        )
        for unit_id in values:
            if not permits_strict_qid_placement(unit_id):
                continue
            qid = qid_by_unit[unit_id]
            if qid and qid in used_qids:
                continue
            selected.append(unit_id)
            if qid:
                used_qids.add(qid)
            selected_in_stratum = sum(
                _stratum(row) == key and str(row["unit_id"]) in selected for row in key_rows
            )
            if selected_in_stratum >= quotas[key]:
                break
    if len(selected) < target:
        remaining = sorted(
            (str(row["unit_id"]) for row in key_rows if str(row["unit_id"]) not in selected),
            key=lambda unit_id: (
                not permits_strict_qid_placement(unit_id),
                _hash(f"{seed}:overlap-fallback:{unit_id}"),
            ),
        )
        for unit_id in remaining:
            qid = qid_by_unit[unit_id]
            selected.append(unit_id)
            if qid:
                used_qids.add(qid)
            if len(selected) == target:
                break
    if len(selected) != target:
        raise RuntimeError(f"Selected {len(selected)} overlap units; expected {target}")
    return sorted(
        selected,
        key=lambda unit_id: _hash(f"{seed}:overlap-order:{unit_id}"),
    )


def _match_additional_copies(
    *,
    unit_ids: list[str],
    annotator_ids: list[str],
    bins: dict[str, list[dict[str, object]]],
    membership: dict[str, set[str]],
    qid_membership: dict[str, set[str]],
    key_by_id: dict[str, dict[str, object]],
    assignments_per_annotator: int,
    seed: int,
    layer: str,
) -> dict[str, str]:
    if not unit_ids:
        return {}
    slots = [
        (annotator_id, slot)
        for annotator_id in annotator_ids
        for slot in range(assignments_per_annotator - len(bins[annotator_id]))
    ]

    def solve(*, avoid_qid_reuse: bool) -> dict[str, str] | None:
        matched_slot: dict[tuple[str, int], str] = {}

        def duplicate_qids(annotator_id: str, *, excluding: tuple[str, int]) -> set[str]:
            return {
                str(key_by_id[matched_unit].get("qid", ""))
                for slot, matched_unit in matched_slot.items()
                if slot[0] == annotator_id and slot != excluding
            }

        def place(unit_id: str, seen: set[tuple[str, int]]) -> bool:
            qid = str(key_by_id[unit_id].get("qid", ""))
            candidates: list[tuple[str, int]] = []
            for slot in slots:
                annotator_id = slot[0]
                if unit_id in membership[annotator_id] or slot in seen:
                    continue
                repeats_qid = bool(
                    qid
                    and (
                        qid in qid_membership[annotator_id]
                        or qid in duplicate_qids(annotator_id, excluding=slot)
                    )
                )
                if avoid_qid_reuse and repeats_qid:
                    continue
                candidates.append(slot)
            candidates.sort(
                key=lambda slot: (
                    bool(
                        qid
                        and (
                            qid in qid_membership[slot[0]]
                            or qid in duplicate_qids(slot[0], excluding=slot)
                        )
                    ),
                    _hash(f"{seed}:{layer}-match:{unit_id}:{slot[0]}:{slot[1]}"),
                )
            )
            for slot in candidates:
                seen.add(slot)
                incumbent = matched_slot.get(slot)
                if incumbent is None or place(incumbent, seen):
                    matched_slot[slot] = unit_id
                    return True
            return False

        for unit_id in unit_ids:
            if not place(unit_id, set()):
                return None
        placement = {unit_id: slot[0] for slot, unit_id in matched_slot.items()}
        if len(placement) != len(unit_ids):
            return None
        return placement

    placement = solve(avoid_qid_reuse=True)
    if placement is None:
        placement = solve(avoid_qid_reuse=False)
    if placement is None:
        raise RuntimeError(f"Could not place every {layer} annotation in a distinct annotator bin")
    return placement


def _uniform_random_sample(unit_ids: list[str], *, target: int, seed: int) -> list[str]:
    if target == 0:
        return []
    if target < 0 or target > len(unit_ids):
        raise ValueError("Random third-annotation sample is outside the unit population")
    population = sorted(unit_ids)
    return random.Random(seed).sample(population, target)


def _question_uniqueness_is_feasible(
    bins: dict[str, list[dict[str, object]]],
    *,
    key_by_id: dict[str, dict[str, object]],
    assignments_per_annotator: int,
    annotator_count: int,
) -> bool:
    qid_counts = Counter(
        str(key_by_id[str(row["unit_id"])].get("qid", "")) for rows in bins.values() for row in rows
    )
    qid_counts.pop("", None)
    return len(qid_counts) >= assignments_per_annotator and all(
        count <= annotator_count for count in qid_counts.values()
    )


def _repair_duplicate_questions(
    bins: dict[str, list[dict[str, object]]],
    *,
    key_by_id: dict[str, dict[str, object]],
    seed: int,
) -> bool:
    def qid(row: dict[str, object]) -> str:
        return str(key_by_id[str(row["unit_id"])].get("qid", ""))

    def duplicate_positions() -> list[tuple[str, int]]:
        duplicates: list[tuple[str, int]] = []
        for annotator_id, rows in sorted(bins.items()):
            seen: set[str] = set()
            for index, row in enumerate(rows):
                value = qid(row)
                if value and value in seen:
                    duplicates.append((annotator_id, index))
                seen.add(value)
        return duplicates

    while violations := duplicate_positions():
        annotator_id, index = violations[0]
        row = bins[annotator_id][index]
        row_qid = qid(row)
        row_unit_id = str(row["unit_id"])
        source_qids = {
            qid(candidate)
            for candidate_index, candidate in enumerate(bins[annotator_id])
            if candidate_index != index
        }
        source_units = {
            str(candidate["unit_id"])
            for candidate_index, candidate in enumerate(bins[annotator_id])
            if candidate_index != index
        }
        candidates: list[tuple[str, int]] = []
        for other_id, other_rows in bins.items():
            if other_id == annotator_id:
                continue
            for other_index, other_row in enumerate(other_rows):
                other_qid = qid(other_row)
                other_unit_id = str(other_row["unit_id"])
                other_qids = {
                    qid(candidate)
                    for candidate_index, candidate in enumerate(other_rows)
                    if candidate_index != other_index
                }
                other_units = {
                    str(candidate["unit_id"])
                    for candidate_index, candidate in enumerate(other_rows)
                    if candidate_index != other_index
                }
                if (
                    not other_qid
                    or other_qid in source_qids
                    or row_qid in other_qids
                    or other_unit_id in source_units
                    or row_unit_id in other_units
                ):
                    continue
                candidates.append((other_id, other_index))
        candidates.sort(
            key=lambda candidate: (
                _stratum(key_by_id[str(bins[candidate[0]][candidate[1]]["unit_id"])])
                != _stratum(key_by_id[row_unit_id]),
                _hash(f"{seed}:qid-repair:{annotator_id}:{index}:{candidate[0]}:{candidate[1]}"),
            )
        )
        if not candidates:
            return False
        other_id, other_index = candidates[0]
        bins[annotator_id][index], bins[other_id][other_index] = (
            bins[other_id][other_index],
            bins[annotator_id][index],
        )
    return True


def _stratum(row: dict[str, object]) -> str:
    source_kind = str(row.get("source_kind", "unknown"))
    detail = row.get("system_name") if source_kind == "system_output" else row.get("variant_type")
    return f"{row.get('dataset_family')}|{source_kind}|{detail}"


def _key_breakdown(
    rows: list[dict[str, object]],
    field: str,
) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))


def _append(
    bins: dict[str, list[dict[str, object]]],
    membership: dict[str, set[str]],
    qid_membership: dict[str, set[str]],
    annotator_id: str,
    row: dict[str, object],
    *,
    qid: str,
) -> None:
    unit_id = str(row["unit_id"])
    bins[annotator_id].append(row)
    membership[annotator_id].add(unit_id)
    if qid:
        qid_membership[annotator_id].add(qid)


def _progressive_order(
    rows: list[dict[str, object]],
    *,
    seed: str,
) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            annotation_complexity(row),
            _hash(f"{seed}:complexity-tie:{row['unit_id']}"),
        ),
    )


def _assignment_complexity_report(
    assignments: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    sequences = {
        annotator_id: [annotation_complexity(row) for row in rows]
        for annotator_id, rows in assignments.items()
    }
    all_scores = [score for scores in sequences.values() for score in scores]
    all_breakdowns = [
        annotation_complexity_breakdown(row)
        for rows in assignments.values()
        for row in rows
    ]
    component_names = tuple(all_breakdowns[0]) if all_breakdowns else ()
    return {
        "minimum_score": min(all_scores),
        "maximum_score": max(all_scores),
        "all_assignments_monotonic": all(
            scores == sorted(scores) for scores in sequences.values()
        ),
        "per_annotator_score_range": {
            annotator_id: [scores[0], scores[-1]]
            for annotator_id, scores in sorted(sequences.items())
        },
        "component_ranges": {
            component: [
                min(breakdown[component] for breakdown in all_breakdowns),
                max(breakdown[component] for breakdown in all_breakdowns),
            ]
            for component in component_names
        },
    }


def _assignment_row(
    row: dict[str, object],
    *,
    annotator_id: str,
    assignment_index: int,
) -> dict[str, object]:
    public_row = {
        key: value for key, value in row.items() if key not in {"annotator_id", "assignment_index"}
    }
    return {
        "annotator_id": annotator_id,
        "assignment_index": assignment_index,
        **public_row,
    }


def _validate_plan(
    plan: AnnotationAssignmentPlan,
    *,
    expected_unit_ids: set[str],
    key_by_id: dict[str, dict[str, object]],
) -> None:
    counts = plan.annotation_counts
    if set(counts) != expected_unit_ids:
        raise RuntimeError("Assignment plan does not cover exactly the public unit population")
    if any(count < 1 or count > MAX_ANNOTATIONS_PER_UNIT for count in counts.values()):
        raise RuntimeError("Assignment plan produced an invalid annotation multiplicity")
    expected_second = {unit_id for unit_id, count in counts.items() if count >= 2}
    expected_third = {unit_id for unit_id, count in counts.items() if count >= 3}
    if expected_second != set(plan.second_coded_unit_ids):
        raise RuntimeError("Second-coded unit metadata does not match assignment rows")
    if expected_third != set(plan.third_coded_unit_ids):
        raise RuntimeError("Third-coded unit metadata does not match assignment rows")

    raters_by_unit: defaultdict[str, set[str]] = defaultdict(set)
    for annotator_id, rows in plan.assignments.items():
        unit_ids = [str(row["unit_id"]) for row in rows]
        if len(unit_ids) != len(set(unit_ids)):
            raise RuntimeError(f"{annotator_id} received a repeated annotation unit")
        qids = [str(key_by_id[unit_id].get("qid", "")) for unit_id in unit_ids]
        nonempty_qids = [qid for qid in qids if qid]
        if plan.question_uniqueness_achieved and len(nonempty_qids) != len(set(nonempty_qids)):
            raise RuntimeError(f"{annotator_id} received more than one answer to the same question")
        if any(str(row.get("annotator_id", "")) != annotator_id for row in rows):
            raise RuntimeError(f"{annotator_id} contains a row assigned to another annotator")
        if [row.get("assignment_index") for row in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"{annotator_id} has invalid assignment indexes")
        for unit_id in unit_ids:
            raters_by_unit[unit_id].add(annotator_id)

    if any(len(raters_by_unit[unit_id]) != count for unit_id, count in counts.items()):
        raise RuntimeError("A unit was assigned repeatedly to the same annotator")
    if plan.question_uniqueness_feasible and not plan.question_uniqueness_achieved:
        raise RuntimeError("Question-unique workloads were feasible but not achieved")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
