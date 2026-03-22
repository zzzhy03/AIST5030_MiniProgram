#!/usr/bin/env python3
"""Plot a training loss curve from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plot_loss.py. Install the dependencies with "
        "'pip install -r requirements.txt' first."
    ) from exc


LOSS_CANDIDATES = ("loss", "train_loss", "training_loss", "value")
STEP_CANDIDATES = ("step", "global_step", "iteration", "iter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV file containing training logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", default="Training Loss", help="Plot title.")
    parser.add_argument("--step-column", default=None, help="Optional explicit column name for step.")
    parser.add_argument("--loss-column", default=None, help="Optional explicit column name for loss.")
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=25,
        help="Optional moving-average window. Use 0 or 1 to disable smoothing.",
    )
    return parser.parse_args()


def find_column(fieldnames: list[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Column '{requested}' not found in CSV. Available columns: {fieldnames}")
        return requested

    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(f"Could not detect a suitable column in CSV. Available columns: {fieldnames}")


def load_points(csv_path: Path, step_column: str | None, loss_column: str | None) -> tuple[list[float], list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file must contain a header row.")

        step_key = find_column(reader.fieldnames, step_column, STEP_CANDIDATES)
        loss_key = find_column(reader.fieldnames, loss_column, LOSS_CANDIDATES)

        steps: list[float] = []
        losses: list[float] = []
        for row in reader:
            if not row.get(step_key) or not row.get(loss_key):
                continue
            steps.append(float(row[step_key]))
            losses.append(float(row[loss_key]))

    if not steps:
        raise ValueError("No valid step/loss pairs were found in the CSV file.")

    return steps, losses


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values

    rolling: list[float] = []
    current = sum(values[:window])
    rolling.append(current / window)
    for index in range(window, len(values)):
        current += values[index] - values[index - window]
        rolling.append(current / window)
    return rolling


def plot_loss(
    steps: list[float],
    losses: list[float],
    title: str,
    output_path: Path,
    moving_average_window: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    if moving_average_window > 1 and len(losses) >= moving_average_window:
        smoothed_losses = moving_average(losses, moving_average_window)
        smoothed_steps = steps[moving_average_window - 1 :]
        plt.plot(steps, losses, linewidth=1, alpha=0.35, label="Raw Loss")
        plt.plot(smoothed_steps, smoothed_losses, linewidth=2.5, label=f"Moving Average ({moving_average_window})")
        plt.legend()
    else:
        plt.plot(steps, losses, linewidth=2, label="Loss")
        plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main() -> None:
    args = parse_args()
    steps, losses = load_points(args.input, args.step_column, args.loss_column)
    plot_loss(steps, losses, args.title, args.output, args.moving_average_window)
    print(f"Saved loss plot to {args.output}")


if __name__ == "__main__":
    main()
