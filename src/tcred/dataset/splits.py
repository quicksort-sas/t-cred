from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping

BASE_SPLITS = ("train", "dev", "test_auto", "generator_audit_reserve")


def stable_group_splits(
    scenario_to_group: Mapping[str, str],
    *,
    namespace: str,
) -> dict[str, list[str]]:
    """Assign immutable source groups to deterministic hash buckets.

    Related scenarios receive one split even when the release later grows. The
    human pool is an explicit subset of held-out test scenarios, not a competing
    partition.
    """
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for scenario_id, group_id in scenario_to_group.items():
        groups[group_id].append(scenario_id)

    result = {name: [] for name in BASE_SPLITS}
    result["human_pool"] = []
    for group_id, scenario_ids in sorted(groups.items()):
        bucket = _bucket(namespace, group_id)
        if bucket < 600:
            split = "train"
        elif bucket < 750:
            split = "dev"
        elif bucket < 950:
            split = "test_auto"
        else:
            split = "generator_audit_reserve"
        result[split].extend(scenario_ids)
        if 750 <= bucket < 800:
            result["human_pool"].extend(scenario_ids)
    return {name: sorted(ids) for name, ids in result.items()}


def _bucket(namespace: str, group_id: str) -> int:
    digest = hashlib.sha256(f"tcred-split-v2:{namespace}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 1000
