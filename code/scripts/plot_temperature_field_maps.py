#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot relative continuous FPV module temperature fields from prediction samples.

The FPV metadata provides relative plot locations (SE/SW/M/NW/NE, or
NW/N/NE/M/S) and A/B/C plot-location numbers, but not surveyed module
coordinates. The generated field maps therefore use inferred relative
coordinates and should be described as relative interpolated temperature-field
visualizations, not true geometric maps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


TEMP_PREFIX_RE = re.compile(r"^(true|pred)_(.+)$")

CARDINAL_COORDS: Dict[str, Tuple[float, float]] = {
    "NW": (-1.0, 1.0),
    "N": (0.0, 1.0),
    "NE": (1.0, 1.0),
    "W": (-1.0, 0.0),
    "M": (0.0, 0.0),
    "MD": (0.0, 0.0),
    "E": (1.0, 0.0),
    "SW": (-1.0, -1.0),
    "S": (0.0, -1.0),
    "SE": (1.0, -1.0),
}

REPL_OFFSETS: Dict[str, Tuple[float, float]] = {
    "A": (-0.16, -0.06),
    "B": (0.0, 0.08),
    "C": (0.16, -0.06),
}

COL_PREFIX_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("SEPTM", "SE"),
    ("SWPTM", "SW"),
    ("NWPTM", "NW"),
    ("NEPTM", "NE"),
    ("NOPTM", "N"),
    ("SOPTM", "S"),
    ("MDPTM", "M"),
)


def infer_location_from_channel(channel: str) -> Tuple[str, str]:
    for prefix, loc in COL_PREFIX_ALIASES:
        if channel.startswith(prefix):
            repl = channel[-1] if channel[-1] in REPL_OFFSETS else "B"
            return loc, repl
    if channel.startswith("MODT"):
        try:
            number = int(channel.replace("MODT", ""))
        except ValueError:
            number = 1
        row = 0 if number <= 8 else -1
        col = ((number - 1) % 8) / 7.0 * 2.0 - 1.0
        return f"RTB{number}", "B"
    raise ValueError(f"Cannot infer relative location for channel: {channel}")


def relative_xy(channel: str) -> Tuple[float, float]:
    if channel.startswith("MODT"):
        number = int(channel.replace("MODT", ""))
        row = 0.5 if number <= 8 else -0.5
        col = ((number - 1) % 8) / 7.0 * 2.0 - 1.0
        return col, row
    loc, repl = infer_location_from_channel(channel)
    x, y = CARDINAL_COORDS[loc]
    dx, dy = REPL_OFFSETS.get(repl, (0.0, 0.0))
    return x + dx, y + dy


def prediction_channels(df: pd.DataFrame) -> List[str]:
    channels: List[str] = []
    for col in df.columns:
        match = TEMP_PREFIX_RE.match(col)
        if match and match.group(1) == "true":
            channel = match.group(2)
            if f"pred_{channel}" in df.columns:
                channels.append(channel)
    return channels


def choose_row(df: pd.DataFrame, channels: Iterable[str], row_mode: str, row_index: int) -> int:
    if row_mode == "index":
        return max(0, min(row_index, len(df) - 1))
    values = df[[f"true_{c}" for c in channels]].to_numpy(dtype=float)
    preds = df[[f"pred_{c}" for c in channels]].to_numpy(dtype=float)
    spread = np.nanpercentile(values, 95, axis=1) - np.nanpercentile(values, 5, axis=1)
    if row_mode == "max_spread":
        return int(np.nanargmax(spread))
    if row_mode == "median_spread":
        target = np.nanmedian(spread)
        return int(np.nanargmin(np.abs(spread - target)))
    if row_mode == "max_field_mae":
        field_mae = np.nanmean(np.abs(preds - values), axis=1)
        return int(np.nanargmax(field_mae))
    raise ValueError(f"Unknown row mode: {row_mode}")


def interpolate_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    triang = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(triang, z)
    gx = np.linspace(x.min() - 0.08, x.max() + 0.08, grid_n)
    gy = np.linspace(y.min() - 0.08, y.max() + 0.08, grid_n)
    xx, yy = np.meshgrid(gx, gy)
    zz = interp(xx, yy)
    return xx, yy, np.asarray(zz)


def plot_one(
    csv_path: Path,
    output_path: Path,
    row_mode: str,
    row_index: int,
    grid_n: int,
    title: str,
) -> None:
    df = pd.read_csv(csv_path)
    channels = prediction_channels(df)
    if not channels:
        raise ValueError(f"No true_/pred_ channel pairs found in {csv_path}")
    idx = choose_row(df, channels, row_mode, row_index)

    x = np.array([relative_xy(c)[0] for c in channels], dtype=float)
    y = np.array([relative_xy(c)[1] for c in channels], dtype=float)
    true = df.loc[idx, [f"true_{c}" for c in channels]].to_numpy(dtype=float)
    pred = df.loc[idx, [f"pred_{c}" for c in channels]].to_numpy(dtype=float)
    err = pred - true
    timestamp = str(df.loc[idx, "target_time"]) if "target_time" in df.columns else f"row {idx}"

    finite = np.isfinite(true) & np.isfinite(pred)
    x, y, true, pred, err = x[finite], y[finite], true[finite], pred[finite], err[finite]
    channels = [c for c, keep in zip(channels, finite) if keep]

    true_grid = interpolate_grid(x, y, true, grid_n)
    pred_grid = interpolate_grid(x, y, pred, grid_n)
    err_grid = interpolate_grid(x, y, err, grid_n)

    vmin = float(np.nanmin([np.nanmin(true), np.nanmin(pred)]))
    vmax = float(np.nanmax([np.nanmax(true), np.nanmax(pred)]))
    emax = float(np.nanmax(np.abs(err)))
    if emax < 1e-6:
        emax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    panels = [
        ("Observed", true_grid, true, "coolwarm", vmin, vmax),
        ("Predicted", pred_grid, pred, "coolwarm", vmin, vmax),
        ("Prediction error", err_grid, err, "RdBu_r", -emax, emax),
    ]
    for ax, (name, grid, points, cmap, lo, hi) in zip(axes, panels):
        xx, yy, zz = grid
        cf = ax.contourf(xx, yy, zz, levels=18, cmap=cmap, vmin=lo, vmax=hi)
        ax.scatter(x, y, c=points, s=42, cmap=cmap, vmin=lo, vmax=hi, edgecolors="black", linewidths=0.5)
        for xi, yi, label in zip(x, y, channels):
            ax.text(xi, yi + 0.055, label.replace("PTM", ""), ha="center", va="bottom", fontsize=7)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(cf, ax=ax, shrink=0.86, label="degC")

    suptitle = title or csv_path.parent.name
    fig.suptitle(f"{suptitle} | {timestamp} | relative interpolated field", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--row-mode",
        default="max_spread",
        choices=["max_spread", "median_spread", "max_field_mae", "index"],
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=140)
    parser.add_argument("--title", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_one(
        Path(args.pred_csv),
        Path(args.output),
        args.row_mode,
        args.row_index,
        args.grid_n,
        args.title,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
