#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-site FPV sensorless temperature-field forecasting.

The single-site script evaluates each geographical site separately. This script
tests cross-region generalization by training on multiple sites and evaluating
on a held-out site.

Target layouts:
    common8   All four FPV sites, using common reliable locations:
              NW A/B/C, NE A/C, M A/B/C. NE B is excluded because Altamonte
              has insufficient chronological validation/test coverage.
    sesw14    Altamonte/Oakville/Orlando, using the SE/SW/M/NW/NE layout with
              NE B excluded for Altamonte compatibility.

Inputs are canonical meteorological/time features only. Historical module
temperature values are not used as inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import train_fpv_sensorless_field_models as base


TARGET_LAYOUTS: Dict[str, Dict[str, object]] = {
    "common8": {
        "sites": ("altamonte", "oakville", "orlando", "windsor"),
        "temp_cols": (
            "NWPTMA", "NWPTMB", "NWPTMC",
            "NEPTMA", "NEPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
        ),
    },
    "sesw14": {
        "sites": ("altamonte", "oakville", "orlando"),
        "temp_cols": (
            "SEPTMA", "SEPTMB", "SEPTMC",
            "SWPTMA", "SWPTMB", "SWPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
            "NWPTMA", "NWPTMB", "NWPTMC",
            "NEPTMA", "NEPTMC",
        ),
    },
}


CANONICAL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "poa_1": ("FPOAI1", "FPAIRR", "POAIR1"),
    "poa_2": ("FPOAI2", "FHZIRR", "POAIR2", "FPHIRR"),
    "poa_3": ("FSPIRR", "POAIR3"),
    "ambient_temp": ("FPVDBT", "FAMBTM"),
    "relative_humidity": ("FPV_RH",),
    "pressure": ("FRAIRP", "FBPRES"),
    "wind_avg": ("FWINDA", "FWINDS"),
    "wind_max": ("FWINDM",),
    "water_low": ("WTPLOW", "WTM1_5", "WTM0_3", "WTM1_0"),
    "water_mid": ("WTPMID", "WTM5_0", "WTM0_6", "WTM275"),
    "water_deep": ("WTPDEP", "WTM10_", "WTM1_4", "WTM4_5"),
}

WIND_DIR_ALIASES = ("FWDIRA", "FWINDD")


@dataclass
class CrossSiteConfig:
    target_layout: str = "common8"
    output_root: str = "code/runs"
    run_name: str = ""
    resample_minutes: int = 1
    lookback_minutes: int = 60
    horizon_minutes: int = 1
    models: Tuple[str, ...] = ("dlinear", "autoformer_lite", "crossformer_lite")
    seeds: Tuple[int, ...] = (42,)
    epochs: int = 25
    patience: int = 5
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_size: int = 96
    d_model: int = 96
    num_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.10
    loss: str = "mse"
    spatial_loss_weight: float = 0.10
    finite_prediction_clip: float = 8.0
    epoch_pause_sec: float = 0.0
    torch_num_threads: int = 0
    cuda_memory_fraction: float = 0.0
    short_gap_limit_steps: int = 2
    missing_feature_threshold: float = 0.40
    min_irradiance: float = 0.0
    input_min_mean_irradiance: float = 0.0
    max_train_samples_per_site: int = 60000
    max_val_samples_per_site: int = 12000
    max_test_samples: int = 60000
    prediction_sample_rows: int = 0
    target_outlier_threshold: float = 25.0
    scaler: str = "robust"

    @property
    def lookback_steps(self) -> int:
        if self.lookback_minutes % self.resample_minutes != 0:
            raise ValueError("lookback_minutes must be divisible by resample_minutes")
        return self.lookback_minutes // self.resample_minutes

    @property
    def horizon_steps(self) -> int:
        if self.horizon_minutes % self.resample_minutes != 0:
            raise ValueError("horizon_minutes must be divisible by resample_minutes")
        return self.horizon_minutes // self.resample_minutes

    @property
    def resample_rule(self) -> str:
        return f"{self.resample_minutes}min"


@dataclass
class SiteArrays:
    site: str
    df: pd.DataFrame
    x_raw: np.ndarray
    y_raw: np.ndarray
    feature_cols: List[str]
    temp_cols: List[str]
    irr_col: str
    anchors: np.ndarray


class CrossSiteWindowDataset(Dataset):
    def __init__(
        self,
        x_scaled_by_site: Sequence[np.ndarray],
        y_scaled_by_site: Sequence[np.ndarray],
        refs: Sequence[Tuple[int, int]],
        lookback_steps: int,
        horizon_steps: int,
    ) -> None:
        self.x_scaled_by_site = x_scaled_by_site
        self.y_scaled_by_site = y_scaled_by_site
        self.refs = list(refs)
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        site_idx, anchor = self.refs[idx]
        start = anchor - self.lookback_steps + 1
        target = anchor + self.horizon_steps
        x = self.x_scaled_by_site[site_idx][start : anchor + 1]
        y = self.y_scaled_by_site[site_idx][target]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def choose_alias(columns: Sequence[str], aliases: Sequence[str]) -> str:
    available = set(columns)
    return next((name for name in aliases if name in available), "")


def load_cross_site_dataframe(site: str, temp_cols: Sequence[str], cfg: CrossSiteConfig) -> pd.DataFrame:
    site_info = base.SITE_CONFIGS[site]
    data_file = Path(str(site_info["data_file"]))
    header = pd.read_csv(data_file, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing_targets = [c for c in temp_cols if c not in header]
    if missing_targets:
        raise ValueError(f"{site} is missing target columns: {missing_targets}")

    alias_cols: List[str] = []
    for aliases in CANONICAL_ALIASES.values():
        alias_cols.extend([c for c in aliases if c in header])
    alias_cols.extend([c for c in WIND_DIR_ALIASES if c in header])
    base_cols = list(dict.fromkeys(["DAY", "HOUR"] + list(temp_cols) + alias_cols))
    df = pd.read_csv(data_file, usecols=base_cols, encoding="utf-8-sig")
    df["datetime"] = base.infer_datetime(df)
    df = base.clean_numeric_frame(df, temp_cols, base.Config())
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime")

    numeric_cols = [c for c in df.columns if c not in ("DAY", "HOUR")]
    df = df[numeric_cols].resample(cfg.resample_rule).mean()
    df = df.interpolate(
        method="time",
        limit=cfg.short_gap_limit_steps,
        limit_direction="both",
    )

    out = pd.DataFrame(index=df.index)
    for name, aliases in CANONICAL_ALIASES.items():
        col = choose_alias(df.columns, aliases)
        out[name] = df[col] if col else np.nan

    wind_dir_col = choose_alias(df.columns, WIND_DIR_ALIASES)
    if wind_dir_col:
        radians = np.deg2rad(df[wind_dir_col])
        out["wind_dir_sin"] = np.sin(radians)
        out["wind_dir_cos"] = np.cos(radians)
    else:
        out["wind_dir_sin"] = np.nan
        out["wind_dir_cos"] = np.nan

    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24.0)
    doy = out.index.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)

    for col in list(CANONICAL_ALIASES.keys()):
        out[f"d{col}"] = out[col].diff()

    for col in temp_cols:
        out[col] = df[col]
    return out


def select_cross_features(site_frames: Dict[str, pd.DataFrame], cfg: CrossSiteConfig) -> List[str]:
    target_cols = set(TARGET_LAYOUTS[cfg.target_layout]["temp_cols"])  # type: ignore[index]
    candidates = [c for c in next(iter(site_frames.values())).columns if c not in target_cols]
    selected = []
    dropped: List[Tuple[str, float]] = []
    for col in candidates:
        max_missing = max(float(frame[col].isna().mean()) for frame in site_frames.values())
        if max_missing <= cfg.missing_feature_threshold:
            selected.append(col)
        else:
            dropped.append((col, max_missing))
    if dropped:
        print("Dropped cross-site high-missing canonical features:")
        for col, miss in dropped:
            print(f"  {col}: max missing {miss:.2%}")
    return selected


def build_site_arrays(site: str, frame: pd.DataFrame, feature_cols: List[str], temp_cols: List[str], cfg: CrossSiteConfig) -> SiteArrays:
    x_raw = frame[feature_cols].to_numpy(dtype=np.float32)
    y_raw = frame[temp_cols].to_numpy(dtype=np.float32)
    irr_col = "poa_1" if "poa_1" in feature_cols else ""
    anchor_cfg = base.Config(
        resample_minutes=cfg.resample_minutes,
        lookback_minutes=cfg.lookback_minutes,
        short_gap_limit_steps=cfg.short_gap_limit_steps,
        min_irradiance=cfg.min_irradiance,
        input_min_mean_irradiance=cfg.input_min_mean_irradiance,
    )
    anchors = base.build_anchors(frame, x_raw, y_raw, irr_col, cfg.horizon_steps, anchor_cfg)
    return SiteArrays(site, frame, x_raw, y_raw, feature_cols, temp_cols, irr_col, anchors)


def sample_refs(
    site_idx: int,
    anchors: np.ndarray,
    limit: int,
    rng: np.random.Generator,
    sort_result: bool = False,
) -> List[Tuple[int, int]]:
    anchors = np.asarray(anchors, dtype=np.int64)
    if limit > 0 and len(anchors) > limit:
        anchors = rng.choice(anchors, size=limit, replace=False)
    if sort_result:
        anchors = np.sort(anchors)
    return [(site_idx, int(a)) for a in anchors]


def make_loader(
    x_scaled_by_site: Sequence[np.ndarray],
    y_scaled_by_site: Sequence[np.ndarray],
    refs: Sequence[Tuple[int, int]],
    cfg: CrossSiteConfig,
    shuffle: bool,
) -> DataLoader:
    ds = CrossSiteWindowDataset(
        x_scaled_by_site,
        y_scaled_by_site,
        refs,
        cfg.lookback_steps,
        cfg.horizon_steps,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fit_global_scalers(
    sites: Sequence[SiteArrays],
    train_refs: Sequence[Tuple[int, int]],
    y_target_by_site: Sequence[np.ndarray],
    cfg: CrossSiteConfig,
) -> Tuple[base.ArrayScaler, base.ArrayScaler]:
    x_rows = []
    y_rows = []
    for site_idx, anchor in train_refs:
        x_rows.append(sites[site_idx].x_raw[anchor])
        y_rows.append(y_target_by_site[site_idx][anchor + cfg.horizon_steps])
    x_scaler = base.ArrayScaler().fit(np.vstack(x_rows).astype(np.float32))
    y_scaler = base.ArrayScaler().fit(np.vstack(y_rows).astype(np.float32))
    return x_scaler, y_scaler


def train_factorized_cross_site(
    model_name: str,
    sites: Sequence[SiteArrays],
    train_refs: Sequence[Tuple[int, int]],
    val_refs: Sequence[Tuple[int, int]],
    test_refs: Sequence[Tuple[int, int]],
    cfg: CrossSiteConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], float]:
    mean_raw_by_site = [np.mean(site.y_raw, axis=1, keepdims=True) for site in sites]
    resid_raw_by_site = [site.y_raw - mean_raw for site, mean_raw in zip(sites, mean_raw_by_site)]

    x_scaler, mean_scaler = fit_global_scalers(sites, train_refs, mean_raw_by_site, cfg)
    _, resid_scaler = fit_global_scalers(sites, train_refs, resid_raw_by_site, cfg)

    x_scaled = [x_scaler.transform(site.x_raw) for site in sites]
    mean_scaled = [mean_scaler.transform(mean_raw) for mean_raw in mean_raw_by_site]
    resid_scaled = [resid_scaler.transform(resid_raw) for resid_raw in resid_raw_by_site]

    model_cfg = base.Config(
        lookback_minutes=cfg.lookback_minutes,
        resample_minutes=cfg.resample_minutes,
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hidden_size=cfg.hidden_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        loss=cfg.loss,
        spatial_loss_weight=cfg.spatial_loss_weight,
        epoch_pause_sec=cfg.epoch_pause_sec,
        torch_num_threads=cfg.torch_num_threads,
        cuda_memory_fraction=cfg.cuda_memory_fraction,
    )

    mean_loaders = (
        make_loader(x_scaled, mean_scaled, train_refs, cfg, True),
        make_loader(x_scaled, mean_scaled, val_refs, cfg, False),
        make_loader(x_scaled, mean_scaled, test_refs, cfg, False),
    )
    resid_loaders = (
        make_loader(x_scaled, resid_scaled, train_refs, cfg, True),
        make_loader(x_scaled, resid_scaled, val_refs, cfg, False),
        make_loader(x_scaled, resid_scaled, test_refs, cfg, False),
    )

    mean_model = base.build_model(model_name, cfg.lookback_steps, sites[0].x_raw.shape[1], 1, model_cfg)
    resid_model = base.build_model(
        model_name,
        cfg.lookback_steps,
        sites[0].x_raw.shape[1],
        sites[0].y_raw.shape[1],
        model_cfg,
    )
    mean_model, mean_val = base.train_one_model(mean_model, mean_loaders[0], mean_loaders[1], model_cfg, device)
    resid_model, resid_val = base.train_one_model(resid_model, resid_loaders[0], resid_loaders[1], model_cfg, device)

    pred_mean = mean_scaler.inverse_transform(base.predict_model(mean_model, mean_loaders[2], device))
    pred_resid = resid_scaler.inverse_transform(base.predict_model(resid_model, resid_loaders[2], device))
    pred_resid = pred_resid - np.mean(pred_resid, axis=1, keepdims=True)
    y_pred = pred_mean + pred_resid

    y_true = []
    target_times: List[pd.Timestamp] = []
    for site_idx, anchor in test_refs:
        target = anchor + cfg.horizon_steps
        y_true.append(sites[site_idx].y_raw[target])
        target_times.append(sites[site_idx].df.index[target])
    return np.vstack(y_true).astype(np.float32), y_pred.astype(np.float32), target_times, float((mean_val + resid_val) / 2.0)


def save_prediction_sample(
    out_path: Path,
    target_times: Sequence[pd.Timestamp],
    temp_cols: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_rows: int,
) -> None:
    n = min(max_rows, len(y_true))
    data = {"target_time": [str(t) for t in target_times[:n]]}
    for idx, col in enumerate(temp_cols):
        data[f"true_{col}"] = y_true[:n, idx]
        data[f"pred_{col}"] = y_pred[:n, idx]
    pd.DataFrame(data).to_csv(out_path, index=False)


def run_cross_site(cfg: CrossSiteConfig) -> Path:
    if cfg.target_layout not in TARGET_LAYOUTS:
        raise ValueError(f"Unknown target layout: {cfg.target_layout}")
    layout = TARGET_LAYOUTS[cfg.target_layout]
    sites_to_use = list(layout["sites"])  # type: ignore[index]
    temp_cols = list(layout["temp_cols"])  # type: ignore[index]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"cross_site_{cfg.target_layout}_{stamp}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.torch_num_threads > 0:
        torch.set_num_threads(cfg.torch_num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, cfg.torch_num_threads)))
        except RuntimeError:
            pass

    print(f"Run dir: {run_dir}")
    print(f"Target layout: {cfg.target_layout} | sites: {sites_to_use}")
    print(f"Targets: {temp_cols}")

    site_frames = {
        site: load_cross_site_dataframe(site, temp_cols, cfg)
        for site in sites_to_use
    }
    feature_cols = select_cross_features(site_frames, cfg)
    site_arrays = [
        build_site_arrays(site, site_frames[site], feature_cols, temp_cols, cfg)
        for site in sites_to_use
    ]
    for site in site_arrays:
        print(
            f"{site.site}: rows={len(site.df)} anchors={len(site.anchors)} "
            f"features={len(feature_cols)}"
        )

    payload = asdict(cfg)
    payload.update(
        {
            "run_dir": str(run_dir),
            "sites": sites_to_use,
            "temp_cols": temp_cols,
            "feature_cols": feature_cols,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(min(cfg.cuda_memory_fraction, 1.0))
    records: List[Dict[str, object]] = []
    for seed in cfg.seeds:
        base.set_seed(seed)
        rng = np.random.default_rng(seed)
        for heldout_idx, heldout_site in enumerate(site_arrays):
            train_refs: List[Tuple[int, int]] = []
            val_refs: List[Tuple[int, int]] = []
            for site_idx, site in enumerate(site_arrays):
                if site_idx == heldout_idx:
                    continue
                tr, va, _ = base.chronological_split(site.anchors, len(site.df), cfg.horizon_steps)
                train_refs.extend(sample_refs(site_idx, tr, cfg.max_train_samples_per_site, rng))
                val_refs.extend(sample_refs(site_idx, va, cfg.max_val_samples_per_site, rng))
            test_refs = sample_refs(
                heldout_idx,
                heldout_site.anchors,
                cfg.max_test_samples,
                rng,
                sort_result=True,
            )
            if not train_refs or not val_refs or not test_refs:
                raise ValueError(
                    f"Empty split for heldout={heldout_site.site}: "
                    f"train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
                )
            random.Random(seed).shuffle(train_refs)
            random.Random(seed + 1).shuffle(val_refs)

            for model_name in cfg.models:
                print(
                    f"\nseed={seed} heldout={heldout_site.site} "
                    f"model={model_name} train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
                )
                start = time.time()
                y_true, y_pred, target_times, best_val = train_factorized_cross_site(
                    model_name,
                    site_arrays,
                    train_refs,
                    val_refs,
                    test_refs,
                    cfg,
                    device,
                )
                metrics = base.field_metrics(y_true, y_pred)
                rec: Dict[str, object] = {
                    "target_layout": cfg.target_layout,
                    "seed": seed,
                    "heldout_site": heldout_site.site,
                    "train_sites": ",".join(s.site for i, s in enumerate(site_arrays) if i != heldout_idx),
                    "model": model_name,
                    "horizon_min": cfg.horizon_minutes,
                    "resample_min": cfg.resample_minutes,
                    "lookback_min": cfg.lookback_minutes,
                    "n_targets": len(temp_cols),
                    "n_features": len(feature_cols),
                    "train_samples": len(train_refs),
                    "val_samples": len(val_refs),
                    "test_samples": len(test_refs),
                    "best_val_loss": best_val,
                    "elapsed_sec": round(time.time() - start, 2),
                }
                rec.update(metrics)
                records.append(rec)
                pd.DataFrame(records).to_csv(run_dir / "metrics_cross_site.csv", index=False)

                pred_name = f"pred_holdout_{heldout_site.site}_{cfg.target_layout}_{model_name}_seed{seed}.csv"
                save_prediction_sample(
                    run_dir / pred_name,
                    target_times,
                    temp_cols,
                    y_true,
                    y_pred,
                    cfg.prediction_sample_rows,
                )
                print(
                    f"field_MAE={metrics['field_MAE']:.3f} "
                    f"Tmean_MAE={metrics['Tmean_MAE']:.3f} "
                    f"Tstd_MAE={metrics['Tstd_MAE']:.3f} "
                    f"Spread95_MAE={metrics['Spread95_MAE']:.3f}"
                )
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-layout", default="common8", choices=sorted(TARGET_LAYOUTS.keys()))
    parser.add_argument("--output-root", default="code/runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resample-minutes", type=int, default=1)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--horizon-minutes", type=int, default=1)
    parser.add_argument("--models", default="dlinear,autoformer_lite,crossformer_lite")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--loss", default="mse", choices=["mse", "smoothl1", "huber", "mae"])
    parser.add_argument("--spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--epoch-pause-sec", type=float, default=0.0)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--short-gap-limit-steps", type=int, default=2)
    parser.add_argument("--missing-feature-threshold", type=float, default=0.40)
    parser.add_argument("--min-irradiance", type=float, default=0.0)
    parser.add_argument("--input-min-mean-irradiance", type=float, default=0.0)
    parser.add_argument("--max-train-samples-per-site", type=int, default=60000)
    parser.add_argument("--max-val-samples-per-site", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=60000)
    parser.add_argument("--prediction-sample-rows", type=int, default=5000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = CrossSiteConfig(
        target_layout=args.target_layout,
        output_root=args.output_root,
        run_name=args.run_name,
        resample_minutes=args.resample_minutes,
        lookback_minutes=args.lookback_minutes,
        horizon_minutes=args.horizon_minutes,
        models=base.parse_csv_tuple(args.models, str),
        seeds=base.parse_csv_tuple(args.seeds, int),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        loss=args.loss,
        spatial_loss_weight=args.spatial_loss_weight,
        epoch_pause_sec=args.epoch_pause_sec,
        torch_num_threads=args.torch_num_threads,
        cuda_memory_fraction=args.cuda_memory_fraction,
        short_gap_limit_steps=args.short_gap_limit_steps,
        missing_feature_threshold=args.missing_feature_threshold,
        min_irradiance=args.min_irradiance,
        input_min_mean_irradiance=args.input_min_mean_irradiance,
        max_train_samples_per_site=args.max_train_samples_per_site,
        max_val_samples_per_site=args.max_val_samples_per_site,
        max_test_samples=args.max_test_samples,
        prediction_sample_rows=args.prediction_sample_rows,
    )
    run_dir = run_cross_site(cfg)
    print(f"\nCompleted. Metrics: {run_dir / 'metrics_cross_site.csv'}")


if __name__ == "__main__":
    main()
