from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson


class CalibrationCollector:
    def __init__(self) -> None:
        self.values: defaultdict[str, list[tuple[Any, Any]]] = defaultdict(list)

    def add_batch(self, *, task: str, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        import torch

        if task == "answer":
            targets = torch.stack(
                (batch["answer_u1"], batch["answer_u2"], batch["equivalence"]), dim=1
            )
            self.values["answer"].append(
                (outputs["logits"].detach().float().cpu(), targets.detach().float().cpu())
            )
            return
        if task in {"support", "temporal"}:
            self.values[f"{task}_class"].append(
                (
                    outputs["class_logits"].detach().float().cpu(),
                    batch["class_target"].detach().float().cpu(),
                )
            )
            self.values[f"{task}_supported"].append(
                (
                    outputs["supported_logit"].detach().float().cpu(),
                    batch["supported"].detach().float().cpu(),
                )
            )
            return
        if task == "relevance":
            self.values["relevance"].append(
                (outputs["logit"].detach().float().cpu(), batch["relevance"].float().cpu())
            )
            return
        if task == "answerability":
            self.values["answerability"].append(
                (outputs["logit"].detach().float().cpu(), batch["answerable"].float().cpu())
            )
            return
        if task == "citation":
            self.values["citation_class"].append(
                (
                    outputs["class_logits"].detach().float().cpu(),
                    batch["class_target"].detach().float().cpu(),
                )
            )
            return
        raise ValueError(f"Unknown calibration task: {task}")

    def fit(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "tcred-sl-temperature-calibration-v1",
            "method": "held-out scalar temperature scaling",
            "heads": {},
        }
        for name, batches in sorted(self.values.items()):
            logits = _concatenate([batch[0] for batch in batches])
            targets = _concatenate([batch[1] for batch in batches])
            if name == "answer":
                fitted = _fit_answer_temperatures(logits, targets)
            elif name.endswith("_class"):
                fitted = _fit_class_temperature(logits, targets)
            else:
                fitted = _fit_binary_temperature(logits, targets)
            if fitted is not None:
                result["heads"][name] = fitted
        return result


def fit_model_calibration(
    *,
    model: Any,
    loader: Any,
    output_path: Path,
) -> dict[str, Any]:
    import torch

    collector = CalibrationCollector()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                task=batch["task"],
            )
            collector.add_batch(task=batch["task"], outputs=outputs, batch=batch)
    result = collector.fit()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(output_path)
    return result


def apply_temperature(
    outputs: dict[str, Any],
    *,
    task: str,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch

    if not calibration:
        return outputs
    heads = calibration.get("heads", {})
    calibrated = dict(outputs)
    if task == "answer" and "answer" in heads:
        values = heads["answer"]["temperatures"]
        logits = outputs["logits"]
        u1 = torch.sigmoid(logits[:, 0] / values[0])
        u2 = u1 * torch.sigmoid(logits[:, 1] / values[1])
        equivalence = torch.sigmoid(logits[:, 2] / values[2])
        calibrated.update(
            {
                "u1": u1,
                "u2": u2,
                "equivalence": equivalence,
                "score": (u1 + u2) / 2.0,
            }
        )
        return calibrated
    class_key = f"{task}_class"
    if "class_logits" in outputs and class_key in heads:
        temperature = heads[class_key]["temperatures"][0]
        calibrated["class_probabilities"] = torch.softmax(
            outputs["class_logits"] / temperature, dim=-1
        )
    binary_key = f"{task}_supported" if task in {"support", "temporal"} else task
    if binary_key in heads:
        temperature = heads[binary_key]["temperatures"][0]
        logit_name = "supported_logit" if "supported_logit" in outputs else "logit"
        probability_name = (
            "supported"
            if logit_name == "supported_logit"
            else "answerable"
            if task == "answerability"
            else task
        )
        calibrated[probability_name] = torch.sigmoid(outputs[logit_name] / temperature)
    return calibrated


def _fit_answer_temperatures(logits: Any, targets: Any) -> dict[str, Any] | None:
    import torch
    from torch.nn import functional as functional

    if not (targets >= 0).any():
        return None
    log_temperature = torch.zeros(3, requires_grad=True)

    def objective() -> Any:
        temperatures = log_temperature.exp()
        u1 = torch.sigmoid(logits[:, 0] / temperatures[0])
        u2 = u1 * torch.sigmoid(logits[:, 1] / temperatures[1])
        equivalence = torch.sigmoid(logits[:, 2] / temperatures[2])
        probabilities = (u1, u2, equivalence)
        terms = []
        for index, probability in enumerate(probabilities):
            mask = targets[:, index] >= 0
            if mask.any():
                terms.append(
                    functional.binary_cross_entropy(
                        probability[mask].clamp(1e-6, 1 - 1e-6), targets[mask, index]
                    )
                )
        return torch.stack(terms).mean()

    before = float(objective().detach())
    _optimize_temperature(log_temperature, objective)
    after = float(objective().detach())
    return {
        "n": int((targets >= 0).any(dim=1).sum()),
        "temperatures": [float(value) for value in log_temperature.detach().exp()],
        "loss_before": before,
        "loss_after": after,
    }


def _fit_class_temperature(logits: Any, targets: Any) -> dict[str, Any] | None:
    import torch

    mask = targets[:, 0] >= 0
    if not mask.any():
        return None
    logits = logits[mask]
    targets = targets[mask]
    log_temperature = torch.zeros(1, requires_grad=True)

    def objective() -> Any:
        log_probabilities = torch.log_softmax(logits / log_temperature.exp(), dim=-1)
        return -(targets * log_probabilities).sum(dim=-1).mean()

    before = float(objective().detach())
    _optimize_temperature(log_temperature, objective)
    after = float(objective().detach())
    return {
        "n": int(mask.sum()),
        "temperatures": [float(log_temperature.detach().exp())],
        "loss_before": before,
        "loss_after": after,
    }


def _fit_binary_temperature(logits: Any, targets: Any) -> dict[str, Any] | None:
    from torch.nn import functional as functional

    mask = targets >= 0
    if not mask.any():
        return None
    logits = logits[mask]
    targets = targets[mask]
    import torch

    log_temperature = torch.zeros(1, requires_grad=True)

    def objective() -> Any:
        return functional.binary_cross_entropy_with_logits(
            logits / log_temperature.exp(), targets
        )

    before = float(objective().detach())
    _optimize_temperature(log_temperature, objective)
    after = float(objective().detach())
    return {
        "n": int(mask.sum()),
        "temperatures": [float(log_temperature.detach().exp())],
        "loss_before": before,
        "loss_after": after,
    }


def _optimize_temperature(parameter: Any, objective: Any) -> None:
    import torch

    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        parameter.clamp_(-3.0, 3.0)


def _concatenate(values: list[Any]) -> Any:
    import torch

    return torch.cat(values, dim=0)
