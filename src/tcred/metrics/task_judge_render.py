from __future__ import annotations

from html import escape

from tcred.metrics.task_judge_models import (
    JudgeEvidence,
    JudgeGraphPath,
    JudgeStage,
    TaskJudgeInput,
    VisibleInterval,
)


def render_evidence_stage(row: TaskJudgeInput) -> str:
    return _render(row, stage="evidence", include_reference=False)


def render_answer_stage(row: TaskJudgeInput) -> str:
    return _render(row, stage="answer", include_reference=True)


def _render(row: TaskJudgeInput, *, stage: JudgeStage, include_reference: bool) -> str:
    sections = [
        _tag("applicable_fields", ", ".join(row.stage_fields(stage))),
        _tag("question", row.question),
    ]
    if include_reference:
        sections.append(_tag("reference_answer", row.reference_answer))
    sections.extend(
        [
            _tag("candidate_answer", row.candidate_answer),
            _tag("context_note", row.context_note or "(none)"),
            _tag("displayed_evidence", _render_evidence(row)),
            _tag("graph_paths", _render_paths(row.graph_paths)),
        ]
    )
    return "\n\n".join(sections)


def _render_evidence(row: TaskJudgeInput) -> str:
    evidence = row.displayed_evidence()
    if not evidence:
        return "(none)"
    cited = set(row.cited_evidence_ids)
    return "\n\n".join(
        _render_evidence_item(item, cited=item.evidence_id in cited) for item in evidence
    )


def _render_evidence_item(item: JudgeEvidence, *, cited: bool) -> str:
    publication = item.publication_time or "unknown"
    return "\n".join(
        [
            f"[{escape(item.evidence_id)}] cited={'yes' if cited else 'no'}",
            f"Text: {escape(item.text)}",
            f"Valid time: {_interval(item.valid_time)}",
            f"Publication time: {escape(publication)}",
        ]
    )


def _render_paths(paths: list[JudgeGraphPath]) -> str:
    if not paths:
        return "(none)"
    return "\n\n".join(_render_path(path) for path in paths)


def _render_path(path: JudgeGraphPath) -> str:
    if not path.edges:
        return f"[{escape(path.path_id)}]\n(no edges)"
    edges = []
    for index, edge in enumerate(path.edges, start=1):
        connector = "<-->" if edge.symmetric else "-->"
        relation = edge.relation_label or edge.relation or "unknown relation"
        edges.append(
            " ".join(
                [
                    f"{index}.",
                    f"{escape(edge.source.label)} ({escape(edge.source.id)})",
                    connector,
                    f"{escape(edge.target.label)} ({escape(edge.target.id)});",
                    f"relation={escape(relation)};",
                    f"traversal={escape(edge.traversal_direction)};",
                    f"valid_time={_interval(edge.valid_time)};",
                    f"evidence_id={escape(edge.fact_id or 'unknown')}",
                ]
            )
        )
    return f"[{escape(path.path_id)}]\n" + "\n".join(edges)


def _interval(value: VisibleInterval) -> str:
    if value.type == "unknown" or (value.start is None and value.end is None):
        return "unknown"
    start = value.start or "open"
    end = value.end or "open"
    granularity = f", granularity={value.granularity}" if value.granularity else ""
    return f"type={value.type}, start={escape(start)}, end={escape(end)}{granularity}"


def _tag(name: str, value: str) -> str:
    rendered = value if name in {"displayed_evidence", "graph_paths"} else escape(value)
    return f"<{name}>\n{rendered}\n</{name}>"
