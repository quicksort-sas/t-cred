from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SemanticLoss:
    total: Any
    components: dict[str, Any]
    probabilities: dict[str, Any]


class TCredSLModel:
    """Factory namespace kept import-light until Torch is available."""

    @staticmethod
    def create(
        *,
        backbone_dir: Path,
        tokenizer_size: int,
        dropout: float = 0.10,
    ) -> Any:
        import torch
        from torch import nn
        from transformers import AutoModel

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(backbone_dir, local_files_only=True)
                self.encoder.resize_token_embeddings(tokenizer_size, mean_resizing=False)
                hidden_size = int(self.encoder.config.hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.answer_head = nn.Linear(hidden_size, 3)
                self.support_head = nn.Linear(hidden_size, 3)
                self.support_binary_head = nn.Linear(hidden_size, 1)
                self.relevance_head = nn.Linear(hidden_size, 1)
                self.temporal_head = nn.Linear(hidden_size, 3)
                self.temporal_binary_head = nn.Linear(hidden_size, 1)
                self.answerability_head = nn.Linear(hidden_size, 1)
                self.citation_head = nn.Linear(hidden_size, 3)
                self.model_metadata = {
                    "schema_version": "tcred-sl-model-v1",
                    "backbone_dir": str(backbone_dir),
                    "tokenizer_size": tokenizer_size,
                    "dropout": dropout,
                    "hidden_size": hidden_size,
                }

            def forward(
                self,
                *,
                input_ids: Any,
                attention_mask: Any,
                task: str,
                token_type_ids: Any | None = None,
            ) -> dict[str, Any]:
                encoded = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pooled = self.dropout(encoded.last_hidden_state[:, 0])
                if task == "answer":
                    logits = self.answer_head(pooled)
                    u1 = torch.sigmoid(logits[:, 0])
                    u2 = u1 * torch.sigmoid(logits[:, 1])
                    return {
                        "logits": logits,
                        "u1": u1,
                        "u2": u2,
                        "equivalence": torch.sigmoid(logits[:, 2]),
                        "score": (u1 + u2) / 2.0,
                    }
                if task == "support":
                    class_logits = self.support_head(pooled)
                    supported_logit = self.support_binary_head(pooled).squeeze(-1)
                    return {
                        "class_logits": class_logits,
                        "supported_logit": supported_logit,
                        "class_probabilities": torch.softmax(class_logits, dim=-1),
                        "supported": torch.sigmoid(supported_logit),
                    }
                if task == "relevance":
                    logit = self.relevance_head(pooled).squeeze(-1)
                    return {"logit": logit, "relevance": torch.sigmoid(logit)}
                if task == "temporal":
                    class_logits = self.temporal_head(pooled)
                    supported_logit = self.temporal_binary_head(pooled).squeeze(-1)
                    return {
                        "class_logits": class_logits,
                        "supported_logit": supported_logit,
                        "class_probabilities": torch.softmax(class_logits, dim=-1),
                        "supported": torch.sigmoid(supported_logit),
                    }
                if task == "answerability":
                    logit = self.answerability_head(pooled).squeeze(-1)
                    return {"logit": logit, "answerable": torch.sigmoid(logit)}
                if task == "citation":
                    logits = self.citation_head(pooled)
                    return {
                        "class_logits": logits,
                        "class_probabilities": torch.softmax(logits, dim=-1),
                    }
                raise ValueError(f"Unknown semantic task: {task}")

        return _Model()


def semantic_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    pair_margin: float,
    paired_loss_weight: float,
    invariance_loss_weight: float,
    calibration_loss_weight: float,
) -> SemanticLoss:
    import torch
    from torch.nn import functional as functional

    task = str(batch["task"])
    components: dict[str, Any] = {}
    probabilities: dict[str, Any] = {}
    supervised_probabilities: list[tuple[Any, Any]] = []

    if task == "answer":
        for name in ("u1", "u2", "equivalence"):
            target_name = "answer_u1" if name == "u1" else "answer_u2" if name == "u2" else name
            loss = _masked_probability_bce(outputs[name], batch[target_name])
            if loss is not None:
                components[name] = loss
                supervised_probabilities.append((outputs[name], batch[target_name]))
        rating_mask = batch["scalar_rating"] >= 0
        if rating_mask.any():
            components["scalar_rating"] = functional.mse_loss(
                outputs["score"][rating_mask], batch["scalar_rating"][rating_mask]
            )
        probabilities = {
            "answer_u1": outputs["u1"],
            "answer_u2": outputs["u2"],
            "equivalence": outputs["equivalence"],
            "score": outputs["score"],
        }
        pair_score = outputs["score"]
    elif task in {"support", "temporal"}:
        class_loss = _soft_class_loss(outputs["class_logits"], batch["class_target"])
        if class_loss is not None:
            components["class"] = class_loss
        binary_loss = _masked_logit_bce(outputs["supported_logit"], batch["supported"])
        if binary_loss is not None:
            components["supported"] = binary_loss
            supervised_probabilities.append((outputs["supported"], batch["supported"]))
        probabilities = {
            "class_probabilities": outputs["class_probabilities"],
            "supported": outputs["supported"],
        }
        binary_mask = batch["supported"] >= 0
        pair_score = torch.where(
            binary_mask,
            outputs["supported"],
            outputs["class_probabilities"][:, 0],
        )
    elif task == "relevance":
        components["relevance"] = _required_logit_bce(
            outputs["logit"], batch["relevance"], "relevance"
        )
        probabilities = {"relevance": outputs["relevance"]}
        supervised_probabilities.append((outputs["relevance"], batch["relevance"]))
        pair_score = outputs["relevance"]
    elif task == "answerability":
        components["answerable"] = _required_logit_bce(
            outputs["logit"], batch["answerable"], "answerable"
        )
        probabilities = {"answerable": outputs["answerable"]}
        supervised_probabilities.append((outputs["answerable"], batch["answerable"]))
        pair_score = outputs["answerable"]
    elif task == "citation":
        class_loss = _soft_class_loss(outputs["class_logits"], batch["class_target"])
        if class_loss is None:
            raise ValueError("citation batch has no class supervision")
        components["class"] = class_loss
        probabilities = {"class_probabilities": outputs["class_probabilities"]}
        pair_score = outputs["class_probabilities"][:, 0]
    else:
        raise ValueError(f"Unknown task in loss: {task}")

    if not components:
        raise ValueError(f"Batch for {task} has no applicable supervised targets")
    base_loss = torch.stack(list(components.values())).mean()
    ranking, invariance = _paired_losses(
        pair_score,
        pair_ids=batch["pair_ids"],
        pair_roles=batch["pair_roles"],
        margin=pair_margin,
    )
    if ranking is not None:
        components["paired_margin"] = ranking
        base_loss = base_loss + paired_loss_weight * ranking
    if invariance is not None:
        components["invariance"] = invariance
        base_loss = base_loss + invariance_loss_weight * invariance
    brier_terms = [
        ((probability[target >= 0] - target[target >= 0]) ** 2).mean()
        for probability, target in supervised_probabilities
        if (target >= 0).any()
    ]
    if brier_terms and calibration_loss_weight:
        brier = torch.stack(brier_terms).mean()
        components["brier"] = brier
        base_loss = base_loss + calibration_loss_weight * brier
    return SemanticLoss(total=base_loss, components=components, probabilities=probabilities)


def save_model_bundle(model: Any, output_dir: Path, *, metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    values = dict(getattr(model, "model_metadata", {})) | metadata
    (output_dir / "model_config.json").write_text(
        json.dumps(values, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_model_bundle(
    *,
    bundle_dir: Path,
    backbone_dir: Path,
    tokenizer_size: int,
    dropout: float,
) -> Any:
    from safetensors.torch import load_file

    model = TCredSLModel.create(
        backbone_dir=backbone_dir,
        tokenizer_size=tokenizer_size,
        dropout=dropout,
    )
    model.load_state_dict(load_file(str(bundle_dir / "model.safetensors")), strict=True)
    return model


def _masked_probability_bce(probability: Any, target: Any) -> Any | None:
    from torch.nn import functional as functional

    mask = target >= 0
    if not mask.any():
        return None
    return functional.binary_cross_entropy(probability[mask].clamp(1e-6, 1 - 1e-6), target[mask])


def _masked_logit_bce(logit: Any, target: Any) -> Any | None:
    from torch.nn import functional as functional

    mask = target >= 0
    if not mask.any():
        return None
    return functional.binary_cross_entropy_with_logits(logit[mask], target[mask])


def _required_logit_bce(logit: Any, target: Any, name: str) -> Any:
    loss = _masked_logit_bce(logit, target)
    if loss is None:
        raise ValueError(f"batch has no {name} supervision")
    return loss


def _soft_class_loss(logits: Any, targets: Any) -> Any | None:
    import torch

    mask = targets[:, 0] >= 0
    if not mask.any():
        return None
    log_probabilities = torch.log_softmax(logits[mask], dim=-1)
    return -(targets[mask] * log_probabilities).sum(dim=-1).mean()


def _paired_losses(
    scores: Any,
    *,
    pair_ids: list[str],
    pair_roles: list[str],
    margin: float,
) -> tuple[Any | None, Any | None]:
    import torch

    pairs: dict[str, dict[str, int]] = {}
    for index, (pair_id, role) in enumerate(zip(pair_ids, pair_roles, strict=True)):
        if pair_id and role:
            pairs.setdefault(pair_id, {})[role] = index
    ranking_terms = []
    invariance_terms = []
    for roles in pairs.values():
        if "positive" in roles and "negative" in roles:
            ranking_terms.append(
                torch.relu(margin - scores[roles["positive"]] + scores[roles["negative"]])
            )
        if "invariant_a" in roles and "invariant_b" in roles:
            invariance_terms.append(
                (scores[roles["invariant_a"]] - scores[roles["invariant_b"]]) ** 2
            )
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else None
    invariance = torch.stack(invariance_terms).mean() if invariance_terms else None
    return ranking, invariance
