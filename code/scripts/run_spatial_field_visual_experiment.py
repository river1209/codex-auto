#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch spatial visualization/evaluation for FPV temperature-field predictions.

This script uses relative sensor positions inferred from dataset metadata codes
and prediction CSVs. It produces interpolation figures for selected samples and
per-sensor spatial error summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_temperature_field_maps import (
    choose_row,
    plot_one,
    prediction_channels,
    relative_xy,
)


DEFAULT_CASES: Dict[str, Dict[str, str]] = {
    "oakville": {
        "model": "timexer_lite",
        "pred_csv": (
            "code/runs/hybrid_compare_oakville_h1_seed42_mse/"
            "pred_oakville_h1_xgb_topk_factorized_timexer_lite_seed42.csv"
        ),
        "title": "Oakville h=1 min TimeXer-lite",
    },
    "orlando": {
        "model": "crossformer_lite",
        "pred_csv": (
            "code/runs/hybrid_compare_orlando_h1_seed42_mse/"
            "pred_orlando_h1_xgb_topk_factorized_crossformer_lite_seed42.csv"
        ),
        "title": "Orlando h=1 min Crossformer-lite",
    },
    "windsor": {
        "model": "autoformer_lite",
        "pred_csv": (
            "code/runs/hybrid_compare_windsor_h1_seed42_mse/"
            "pred_windsor_h1_xgb_topk_factorized_autoformer_lite_seed42.csv"
        ),
        "title": "Windsor h=1 min Autoformer-lite",
    },
}


def channel_error_summary(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    rows = []
    for channel in channels:
        true = df[f"true_{channel}"].to_numpy(dtype=float)
        pred = df[f"pred_{channel}"].to_numpy(dtype=float)
        err = pred - true
        x, y = relative_xy(channel)
        rows.append(
            {
                "channel": channel,
                "rel_x": x,
                "rel_y": y,
                "MAE": float(np.nanmean(np.abs(err))),
                "RMSE": float(np.sqrt(np.nanmean(err**2))),
                "Bias": float(np.nanmean(err)),
                "n": int(np.isfinite(err).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_y", "rel_x"], ascending=[False, True])


def plot_sensor_error_map(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    sc = ax.scatter(
        summary["rel_x"],
        summary["rel_y"],
        c=summary["MAE"],
        s=260,
        cmap="magma",
        edgecolors="black",
        linewidths=0.7,
    )
    for _, row in summary.iterrows():
        ax.text(
            row["rel_x"],
            row["rel_y"] + 0.07,
            f"{row['channel']}\n{row['MAE']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="Per-sensor MAE (degC)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def sample_selection_summary(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    rows = []
    for mode in ["median_spread", "max_spread", "max_field_mae"]:
        idx = choose_row(df, channels, mode, 0)
        true = df.loc[idx, [f"true_{c}" for c in channels]].to_numpy(dtype=float)
        pred = df.loc[idx, [f"pred_{c}" for c in channels]].to_numpy(dtype=float)
        err = pred - true
        rows.append(
            {
                "row_mode": mode,
                "row_index": int(idx),
                "target_time": str(df.loc[idx, "target_time"]) if "target_time" in df.columns else "",
                "true_mean": float(np.nanmean(true)),
                "pred_mean": float(np.nanmean(pred)),
                "field_MAE": float(np.nanmean(np.abs(err))),
                "true_spread95": float(np.nanpercentile(true, 95) - np.nanpercentile(true, 5)),
                "pred_spread95": float(np.nanpercentile(pred, 95) - np.nanpercentile(pred, 5)),
                "max_abs_error": float(np.nanmax(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def run_case(site: str, spec: Dict[str, str], output_root: Path, grid_n: int) -> Dict[str, object]:
    pred_csv = Path(spec["pred_csv"])
    df = pd.read_csv(pred_csv)
    channels = prediction_channels(df)
    case_dir = output_root / site
    case_dir.mkdir(parents=True, exist_ok=True)

    sensor_summary = channel_error_summary(df, channels)
    sensor_summary.to_csv(case_dir / f"{site}_per_sensor_relative_error.csv", index=False)
    plot_sensor_error_map(
        sensor_summary,
        case_dir / f"{site}_per_sensor_mae_map.png",
        f"{spec['title']} | relative sensor MAE",
    )

    selection = sample_selection_summary(df, channels)
    selection.to_csv(case_dir / f"{site}_selected_samples.csv", index=False)
    for mode in selection["row_mode"]:
        plot_one(
            pred_csv,
            case_dir / f"{site}_{mode}_interpolated_field.png",
            str(mode),
            0,
            grid_n,
            spec["title"],
        )

    return {
        "site": site,
        "model": spec["model"],
        "pred_csv": str(pred_csv),
        "mean_sensor_MAE": float(sensor_summary["MAE"].mean()),
        "max_sensor_MAE": float(sensor_summary["MAE"].max()),
        "max_sensor_channel": str(sensor_summary.sort_values("MAE", ascending=False).iloc[0]["channel"]),
        "median_spread_field_MAE": float(selection.loc[selection["row_mode"].eq("median_spread"), "field_MAE"].iloc[0]),
        "max_spread_field_MAE": float(selection.loc[selection["row_mode"].eq("max_spread"), "field_MAE"].iloc[0]),
        "max_field_mae": float(selection.loc[selection["row_mode"].eq("max_field_mae"), "field_MAE"].iloc[0]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="code/figures/temperature_fields/spatial_experiment_20260509")
    parser.add_argument("--grid-n", type=int, default=140)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    records = []
    for site, spec in DEFAULT_CASES.items():
        records.append(run_case(site, spec, output_root, args.grid_n))
    pd.DataFrame(records).to_csv(output_root / "spatial_experiment_summary.csv", index=False)
    print(f"Saved spatial experiment outputs to: {output_root}")


if __name__ == "__main__":
    main()
