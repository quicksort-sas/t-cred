"""SABET-QA reproduction and external metric-evaluation support."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AnswerMetricRecord",
    "ReleasedEvaluation",
    "SabetPredictionRecord",
    "parse_released_log",
    "score_prediction",
]

_LAZY_EXPORTS = {
    "AnswerMetricRecord": ("tcred.external_evaluations.sabet_tkgqa.schema", "AnswerMetricRecord"),
    "ReleasedEvaluation": ("tcred.external_evaluations.sabet_tkgqa.schema", "ReleasedEvaluation"),
    "SabetPredictionRecord": (
        "tcred.external_evaluations.sabet_tkgqa.schema",
        "SabetPredictionRecord",
    ),
    "parse_released_log": (
        "tcred.external_evaluations.sabet_tkgqa.released_logs",
        "parse_released_log",
    ),
    "score_prediction": (
        "tcred.external_evaluations.sabet_tkgqa.evaluation",
        "score_prediction",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
