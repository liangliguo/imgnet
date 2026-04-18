from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and attack metrics for the Tiny ImageNet run."
    )
    parser.add_argument("--run-dir", default="runs/resnet18_tinyimagenet")
    parser.add_argument("--metrics", default="", help="Path to metrics.csv.")
    parser.add_argument("--attack-report", default="", help="Path to attack_report.json.")
    parser.add_argument("--output-dir", default="", help="Directory for generated figures.")
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})

    if not rows:
        raise RuntimeError(f"No metric rows found in: {path}")
    return rows


def save_line_plot(
    output_path: Path,
    x: list[float],
    series: Iterable[tuple[str, list[float]]],
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for label, values in series:
        ax.plot(x, values, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_metrics(rows: list[dict[str, float]], output_dir: Path) -> list[Path]:
    epochs = [row["epoch"] for row in rows]
    outputs: list[Path] = []

    accuracy_path = output_dir / "accuracy_curves.png"
    save_line_plot(
        accuracy_path,
        epochs,
        [
            ("train top-1", [row["train_top1"] for row in rows]),
            ("val top-1", [row["val_top1"] for row in rows]),
            ("train top-5", [row["train_top5"] for row in rows]),
            ("val top-5", [row["val_top5"] for row in rows]),
        ],
        "Tiny ImageNet Accuracy",
        "Accuracy (%)",
    )
    outputs.append(accuracy_path)

    loss_path = output_dir / "loss_curves.png"
    save_line_plot(
        loss_path,
        epochs,
        [
            ("train loss", [row["train_loss"] for row in rows]),
            ("val loss", [row["val_loss"] for row in rows]),
        ],
        "Tiny ImageNet Cross-Entropy Loss",
        "Loss",
    )
    outputs.append(loss_path)

    lr_path = output_dir / "learning_rate.png"
    save_line_plot(
        lr_path,
        epochs,
        [("learning rate", [row["lr"] for row in rows])],
        "Learning Rate Schedule",
        "Learning Rate",
    )
    outputs.append(lr_path)

    time_path = output_dir / "epoch_time.png"
    save_line_plot(
        time_path,
        epochs,
        [("epoch seconds", [row["epoch_seconds"] for row in rows])],
        "Epoch Runtime",
        "Seconds",
    )
    outputs.append(time_path)

    return outputs


def plot_attack_report(path: Path, output_dir: Path) -> list[Path]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    methods = [method for method in ("clean", "fgsm", "pgd") if method in report]
    if not methods:
        return []

    top1 = [float(report[method]["top1"]) for method in methods]
    top5 = [float(report[method]["top5"]) for method in methods]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = list(range(len(methods)))
    width = 0.36
    ax.bar([value - width / 2 for value in x], top1, width=width, label="top-1")
    ax.bar([value + width / 2 for value in x], top5, width=width, label="top-5")
    ax.set_title("Clean vs FGSM vs PGD Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([method.upper() for method in methods])
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "attack_accuracy.png"
    fig.savefig(output_path)
    plt.close(fig)
    return [output_path]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.csv"
    attack_report_path = (
        Path(args.attack_report) if args.attack_report else run_dir / "attack_report.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(metrics_path)
    outputs = plot_training_metrics(rows, output_dir)
    outputs.extend(plot_attack_report(attack_report_path, output_dir))

    print("generated figures:", flush=True)
    for output in outputs:
        print(output, flush=True)


if __name__ == "__main__":
    main()
