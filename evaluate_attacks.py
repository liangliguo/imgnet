from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from tqdm import tqdm

from src.attacks import fgsm, pgd_linf
from src.metrics import AverageMeter, accuracy
from src.resnet import build_resnet18
from src.tiny_imagenet import build_val_dataset


def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean, FGSM, and PGD accuracy on Tiny ImageNet validation."
    )
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--checkpoint", default="runs/resnet18_tinyimagenet/best.pt")
    parser.add_argument("--output", default="runs/resnet18_tinyimagenet/attack_report.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--eps", type=parse_fraction, default=parse_fraction("8/255"))
    parser.add_argument("--pgd-alpha", type=parse_fraction, default=parse_fraction("2/255"))
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    attack: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    max_batches: int = 0,
) -> dict[str, float]:
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.eval()
    for batch_index, (images, targets) in enumerate(tqdm(loader, leave=False)):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if attack is not None:
            images = attack(images, targets)

        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, targets)
            top1, top5 = accuracy(logits, targets, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        top1_meter.update(float(top1.item()), batch_size)
        top5_meter.update(float(top5.item()), batch_size)

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    checkpoint_args = checkpoint.get("args", {})

    model = build_resnet18(
        num_classes=len(class_to_idx),
        cifar_stem=not checkpoint_args.get("no_cifar_stem", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_dataset = build_val_dataset(
        args.data_root,
        class_to_idx=class_to_idx,
        image_size=args.image_size,
    )
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    criterion = nn.CrossEntropyLoss()

    clean = evaluate_loader(
        model,
        loader,
        criterion,
        device,
        attack=None,
        max_batches=args.max_batches,
    )

    fgsm_metrics = evaluate_loader(
        model,
        loader,
        criterion,
        device,
        attack=lambda images, targets: fgsm(model, images, targets, args.eps, criterion),
        max_batches=args.max_batches,
    )

    pgd_metrics = evaluate_loader(
        model,
        loader,
        criterion,
        device,
        attack=lambda images, targets: pgd_linf(
            model,
            images,
            targets,
            eps=args.eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
            criterion=criterion,
        ),
        max_batches=args.max_batches,
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "dataset": "Tiny ImageNet validation",
        "eps": args.eps,
        "pgd_alpha": args.pgd_alpha,
        "pgd_steps": args.pgd_steps,
        "max_batches": args.max_batches,
        "clean": clean,
        "fgsm": fgsm_metrics,
        "pgd": pgd_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
