from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Literal

from tcred.dataset.generator import SyntheticDatasetGenerator
from tcred.dataset.models import DatasetBundle
from tcred.dataset.source_grounded import SourceGroundedDatasetGenerator


def estimate_generation_runtime(
    *,
    target_scenarios: int,
    sample_scenarios: int,
    seed: int,
    generator_mode: Literal["source_grounded", "legacy_template"] = "source_grounded",
) -> dict[str, str]:
    started = time.perf_counter()
    generator = (
        SourceGroundedDatasetGenerator(seed=seed)
        if generator_mode == "source_grounded"
        else SyntheticDatasetGenerator(seed=seed)
    )
    bundle = generator.generate(
        scenario_count=sample_scenarios,
        questions_per_scenario=4,
    )
    elapsed = time.perf_counter() - started
    per_scenario = elapsed / sample_scenarios
    estimated = per_scenario * target_scenarios
    return {
        "generator_mode": generator_mode,
        "sample_scenarios": str(sample_scenarios),
        "sample_questions": str(len(bundle.questions)),
        "sample_seconds": f"{elapsed:.3f}",
        "seconds_per_scenario": f"{per_scenario:.4f}",
        "target_scenarios": str(target_scenarios),
        "estimated_seconds": f"{estimated:.1f}",
        "estimated_minutes": f"{estimated / 60:.2f}",
        "note": "Deterministic local generation only; LLM paraphrasing adds API latency.",
    }


def write_generation_report(bundle: DatasetBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "generation_report.md"
    domain_counts = Counter(scenario.domain for scenario in bundle.scenarios)
    operator_counts = Counter(question.temporal_operator for question in bundle.questions)
    system_counts = Counter(question.system_difficulty for question in bundle.questions)
    eval_counts = Counter(question.eval_difficulty for question in bundle.questions)
    variant_counts = Counter(variant.variant_type for variant in bundle.answer_variants)
    source_family_counts = Counter(
        fact.source_type.split(":", 1)[0] for fact in bundle.facts if ":" in fact.source_type
    )

    lines = [
        "# T-CRED Synthetic Generation Report",
        "",
        "## Counts",
        "",
        f"- Scenarios: {len(bundle.scenarios)}",
        f"- Entities: {len(bundle.entities)}",
        f"- Facts/evidence items: {len(bundle.facts)}",
        f"- Questions: {len(bundle.questions)}",
        f"- Graph paths: {len(bundle.graph_paths)}",
        f"- Context packs: {len(bundle.context_packs)}",
        f"- Controlled answer variants: {len(bundle.answer_variants)}",
        "",
        "## Domains",
        "",
        *_counter_lines(domain_counts),
        "",
        "## Temporal Operators",
        "",
        *_counter_lines(operator_counts),
        "",
        "## System Difficulty",
        "",
        *_counter_lines(system_counts),
        "",
        "## Evaluation-Capture Difficulty",
        "",
        *_counter_lines(eval_counts),
        "",
        "## Answer Variant Types",
        "",
        *_counter_lines(variant_counts),
        "",
        "## Source Grounding",
        "",
        *(
            _counter_lines(source_family_counts)
            if source_family_counts
            else ["- Source-family prefixes not present; legacy template generation was used."]
        ),
        "",
        "## Notes",
        "",
        "- Gold labels are produced by a deterministic symbolic solver over generated facts.",
        "- Evaluated systems should not receive question programs, fact roles, or gold labels.",
        "- LLM paraphrasing should be run only as a validated post-processing step.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _counter_lines(counter: Counter[str]) -> list[str]:
    total = sum(counter.values()) or 1
    return [f"- {key}: {value} ({value / total:.1%})" for key, value in sorted(counter.items())]
