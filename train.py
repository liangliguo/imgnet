from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.metrics import AverageMeter, accuracy, append_csv_row
from src.resnet import build_resnet18, count_parameters
from src.tiny_imagenet import make_dataloaders


CSV_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_top1",
    "train_top5",
    "val_loss",
    "val_top1",
    "val_top5",
    "epoch_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on Tiny ImageNet and export quality metrics."
    )
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--output-dir", default="runs/resnet18_tinyimagenet")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cifar-stem", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use torch.nn.DataParallel when multiple CUDA GPUs are available.",
    )
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        top1, top5 = accuracy(logits.detach(), targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        top1_meter.update(float(top1.item()), batch_size)
        top5_meter.update(float(top5.item()), batch_size)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            top1=f"{top1_meter.avg:.2f}",
            top5=f"{top5_meter.avg:.2f}",
        )

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    progress = tqdm(loader, desc=f"val {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        top1_meter.update(float(top1.item()), batch_size)
        top5_meter.update(float(top5.item()), batch_size)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            top1=f"{top1_meter.avg:.2f}",
            top5=f"{top5_meter.avg:.2f}",
        )

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_top1: float,
    best_epoch: int,
    best_metrics: dict[str, float],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_top1": best_val_top1,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def write_quality_report(
    output_dir: Path,
    args: argparse.Namespace,
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: dict[str, float],
    train_size: int,
    val_size: int,
    num_parameters: int,
) -> None:
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "ResNet-18",
        "dataset": "Tiny ImageNet",
        "num_classes": 200,
        "train_size": train_size,
        "val_size": val_size,
        "num_parameters": num_parameters,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "final_epoch": final_metrics,
        "report_metrics": [
            "validation top-1 accuracy",
            "validation top-5 accuracy",
            "validation cross-entropy loss",
            "training top-1/top-5 accuracy",
        ],
        "args": vars(args),
    }
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    markdown = f"""# Model Quality Report

| Item | Value |
| --- | --- |
| Dataset | Tiny ImageNet |
| Model | ResNet-18 |
| Classes | 200 |
| Train images | {train_size} |
| Validation images | {val_size} |
| Parameters | {num_parameters:,} |
| Best epoch | {best_epoch} |
| Best val top-1 | {best_metrics.get("top1", 0.0):.2f}% |
| Best val top-5 | {best_metrics.get("top5", 0.0):.2f}% |
| Best val loss | {best_metrics.get("loss", 0.0):.4f} |
| Final train top-1 | {final_metrics.get("train_top1", 0.0):.2f}% |
| Final train top-5 | {final_metrics.get("train_top5", 0.0):.2f}% |
| Final val top-1 | {final_metrics.get("val_top1", 0.0):.2f}% |
| Final val top-5 | {final_metrics.get("val_top5", 0.0):.2f}% |

Use `metrics.csv` for the training/validation loss and accuracy curves. Use
`best.pt` for FGSM/PGD attack evaluation.
"""
    with (output_dir / "quality_report.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)


def load_resume(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float, int, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_top1", -1.0)),
        int(checkpoint.get("best_epoch", 0)),
        checkpoint.get("best_metrics", {"loss": 0.0, "top1": 0.0, "top5": 0.0}),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    train_loader, val_loader, class_to_idx = make_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18(
        num_classes=len(class_to_idx),
        cifar_stem=not args.no_cifar_stem,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    best_epoch = 0
    best_val_top1 = -1.0
    best_metrics = {"loss": 0.0, "top1": 0.0, "top5": 0.0}
    if args.resume:
        start_epoch, best_val_top1, best_epoch, best_metrics = load_resume(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
        )

    num_parameters = count_parameters(model)
    final_row: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        lr_used = current_lr(optimizer)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        epoch_seconds = time.time() - started

        final_row = {
            "epoch": epoch,
            "lr": lr_used,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_seconds": epoch_seconds,
        }
        append_csv_row(output_dir / "metrics.csv", CSV_FIELDS, final_row)

        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_top1,
                best_epoch,
                best_metrics,
                class_to_idx,
                args,
                final_row,
            )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_top1,
            best_epoch,
            best_metrics,
            class_to_idx,
            args,
            final_row,
        )
        write_quality_report(
            output_dir=output_dir,
            args=args,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            final_metrics=final_row,
            train_size=len(train_loader.dataset),
            val_size=len(val_loader.dataset),
            num_parameters=num_parameters,
        )

        print(
            "epoch "
            f"{epoch:03d}/{args.epochs:03d} "
            f"train_top1={train_metrics['top1']:.2f} "
            f"val_top1={val_metrics['top1']:.2f} "
            f"val_top5={val_metrics['top5']:.2f} "
            f"best_top1={best_val_top1:.2f}"
        )


if __name__ == "__main__":
    main()
