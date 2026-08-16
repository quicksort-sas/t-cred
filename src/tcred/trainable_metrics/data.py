from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


def load_tokenized_split(
    tokenized_dir: Path,
    *,
    partition: str,
    stage: str,
) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    path = tokenized_dir / f"{partition}.{stage}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing tokenized split: {path}")
    return Dataset.from_parquet(str(path))


class HomogeneousBatchSampler:
    """Yield source/task-homogeneous batches while keeping supervised pairs intact."""

    def __init__(
        self,
        *,
        tasks: Sequence[str],
        sources: Sequence[str],
        pair_ids: Sequence[str],
        batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if not (len(tasks) == len(sources) == len(pair_ids)):
            raise ValueError("sampler metadata columns must have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        grouped: defaultdict[tuple[str, str], defaultdict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (task, source, pair_id) in enumerate(
            zip(tasks, sources, pair_ids, strict=True)
        ):
            packet_id = f"pair:{pair_id}" if pair_id else f"single:{index}"
            grouped[(task, source)][packet_id].append(index)
        self._packets = {
            key: tuple(tuple(indexes) for _, indexes in sorted(packets.items()))
            for key, packets in grouped.items()
        }
        oversize = [
            (key, len(packet))
            for key, packets in self._packets.items()
            for packet in packets
            if len(packet) > batch_size
        ]
        if oversize:
            raise ValueError(f"paired packet exceeds batch size: {oversize[:3]}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for key in sorted(self._packets):
            packets = list(self._packets[key])
            if self.shuffle:
                rng.shuffle(packets)
            current: list[int] = []
            for packet in packets:
                if current and len(current) + len(packet) > self.batch_size:
                    batches.append(current)
                    current = []
                current.extend(packet)
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches


class SemanticBatchCollator:
    def __init__(self, tokenizer: Any, *, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        tasks = {str(row["task"]) for row in rows}
        sources = {str(row["source_dataset"]) for row in rows}
        if len(tasks) != 1 or len(sources) != 1:
            raise ValueError("semantic training batches must be task/source homogeneous")
        padded = self.tokenizer.pad(
            [
                {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
                for row in rows
            ],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch: dict[str, Any] = dict(padded)
        batch.update(
            {
                "task": next(iter(tasks)),
                "source_dataset": next(iter(sources)),
                "unit_ids": [str(row["unit_id"]) for row in rows],
                "pair_ids": [str(row["pair_id"]) for row in rows],
                "pair_roles": [str(row["pair_role"]) for row in rows],
                "invariance_group_ids": [str(row["invariance_group_id"]) for row in rows],
                "class_target": torch.tensor(
                    [row["class_target"] for row in rows], dtype=torch.float32
                ),
            }
        )
        for name in (
            "answer_u1",
            "answer_u2",
            "equivalence",
            "supported",
            "relevance",
            "answerable",
            "scalar_rating",
        ):
            batch[name] = torch.tensor([row[name] for row in rows], dtype=torch.float32)
        return batch
