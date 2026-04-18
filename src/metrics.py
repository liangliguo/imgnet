from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[torch.Tensor]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    batch_size = target.size(0)
    return [
        correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size)
        for k in topk
    ]


def append_csv_row(path: str | Path, fieldnames: Iterable[str], row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
