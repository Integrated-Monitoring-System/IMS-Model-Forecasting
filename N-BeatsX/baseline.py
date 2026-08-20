"""
baseline.py
===========
Baseline sederhana (naive & moving-average) untuk pembanding NBEATSx, dengan
metodologi split TANGGAL TETAP yang identik dengan train.py:
    TRAIN : semua data <= train_end
    TEST  : data dari test_start s/d test_end (dibatasi tanggal data terakhir
            yang tersedia, sama seperti split_train_test() di train.py)

Dua baseline yang dihitung:
    - naive_last_value : nilai terakhir yang diketahui di akhir TRAIN,
      diulang FLAT untuk seluruh periode TEST (tidak pakai info apapun
      selain 1 titik data terakhir).
    - moving_average    : rata-rata N hari terakhir di TRAIN (default 7),
      diulang FLAT untuk seluruh periode TEST.

Kedua baseline ini TIDAK memakai model apapun, TIDAK ada training, dan
TIDAK ada recursive forecasting -- sengaja dibuat sesederhana mungkin
supaya jadi tolok ukur minimum yang harus dilewati oleh NBEATSx.

Output-nya (metrics.csv, predictions.csv, prediction_plot.png, run_info.json)
sengaja dibuat dengan skema kolom yang SAMA seperti output train.py, supaya
gampang dibandingkan langsung berdampingan.

Cara pakai (CLI), dijalankan dari dalam folder N-BeatsX/:
    python baseline.py --data-dir "../dataset/N-BeatsX processed" \
        --output-dir artifacts_baseline --target pressure \
        --train-end "2026-06-30 23:59:59" \
        --test-start "2026-07-01 00:00:00" \
        --test-end "2026-07-31 23:59:59" \
        --ma-window 7

Bisa juga dipakai sebagai module:
    from baseline import Config, run_baseline
    cfg = Config(data_dir="../dataset/N-BeatsX processed", output_dir="artifacts_baseline", target="pressure")
    result = run_baseline(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")


# =============================================================================
# KONFIGURASI
# =============================================================================
@dataclass
class Config:
    data_dir: str
    output_dir: str
    target: str

    # Fixed temporal split -- HARUS sama dengan yang dipakai train.py supaya
    # perbandingan metrik-nya adil (apple-to-apple)
    train_end: str = "2026-06-30 23:59:59"
    test_start: str = "2026-07-01 00:00:00"
    test_end: str = "2026-07-31 23:59:59"

    # jumlah hari/titik terakhir di TRAIN yang dipakai untuk moving average
    ma_window: int = 7


# =============================================================================
# LOAD DATA (identik dengan train.py, supaya konsisten)
# =============================================================================
def load_data(cfg: Config):
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    data_dir = Path(cfg.data_dir)
    parquet_file = data_dir / "full_clean.parquet"
    metadata_file = data_dir / "metadata.json"

    if not parquet_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan:\n{parquet_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json tidak ditemukan:\n{metadata_file}")

    df = pd.read_parquet(parquet_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    datetime_col = metadata.get("datetime_col", "DateTime")
    if datetime_col not in df.columns:
        raise ValueError(f"Kolom datetime '{datetime_col}' tidak ditemukan.")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])

    if cfg.target not in metadata["target_cols"]:
        raise ValueError(
            f"Target '{cfg.target}' tidak tersedia.\nTarget tersedia: {metadata['target_cols']}"
        )
    if cfg.target not in df.columns:
        raise ValueError(f"Kolom target '{cfg.target}' tidak ditemukan di dataset.")

    group_col = metadata.get("group_col", "equipment_id")
    df = df.sort_values([group_col, datetime_col]).reset_index(drop=True)

    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    logger.info(f"Rows       : {len(df):,}")
    logger.info(f"Equipment  : {df[group_col].nunique()}")
    logger.info(f"Date range : {df[datetime_col].min()} -> {df[datetime_col].max()}")

    return df, metadata


# =============================================================================
# SPLIT (identik dengan split_train_test() di train.py)
# =============================================================================
def split_train_test(df: pd.DataFrame, cfg: Config, metadata: dict):
    logger.info("=" * 70)
    logger.info("FIXED TRAIN / TEST SPLIT")
    logger.info("=" * 70)

    datetime_col = metadata["datetime_col"]
    group_col = metadata["group_col"]

    train_end = pd.Timestamp(cfg.train_end)
    test_start = pd.Timestamp(cfg.test_start)
    test_end_requested = pd.Timestamp(cfg.test_end)

    actual_data_end = df[datetime_col].max()
    test_end = min(test_end_requested, actual_data_end)

    train_df = df[df[datetime_col] <= train_end].copy()
    test_df = df[(df[datetime_col] >= test_start) & (df[datetime_col] <= test_end)].copy()

    if len(train_df) == 0:
        raise ValueError("TRAIN kosong. Periksa train_end.")
    if len(test_df) == 0:
        raise ValueError("TEST kosong. Periksa test_start/test_end.")

    logger.info(f"TRAIN : {train_df[datetime_col].min()} -> {train_df[datetime_col].max()}")
    logger.info(f"TEST  : {test_df[datetime_col].min()} -> {test_df[datetime_col].max()}")
    logger.info(f"Train rows : {len(train_df):,}")
    logger.info(f"Test rows  : {len(test_df):,}")

    for uid, g in test_df.groupby(group_col):
        logger.info(f"  {uid}: {len(g)} points")

    return train_df, test_df


# =============================================================================
# HITUNG BASELINE (naive last-value & moving average)
# =============================================================================
def compute_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    cfg: Config,
) -> pd.DataFrame:
    """
    Untuk tiap equipment: ambil nilai terakhir (dan rata-rata N hari terakhir)
    dari TRAIN, lalu diulang FLAT untuk semua timestamp di TEST equipment itu.
    """
    datetime_col = metadata["datetime_col"]
    group_col = metadata["group_col"]

    rows = []
    for uid, g_test in test_df.groupby(group_col):
        g_train = train_df[train_df[group_col] == uid].sort_values(datetime_col)
        if g_train.empty:
            logger.warning(f"  -> equipment '{uid}' tidak punya data TRAIN, dilewati")
            continue

        last_value = float(g_train[cfg.target].iloc[-1])
        ma_value = float(g_train[cfg.target].tail(cfg.ma_window).mean())

        for _, row in g_test.sort_values(datetime_col).iterrows():
            rows.append({
                "unique_id": str(uid),
                "ds": row[datetime_col],
                "y": float(row[cfg.target]),
                "naive_last_value": last_value,
                "moving_average": ma_value,
            })

    result = pd.DataFrame(rows)
    logger.info(f"  -> {len(result):,} titik baseline dihitung untuk {result['unique_id'].nunique()} equipment")
    return result


# =============================================================================
# METRIK (identik dengan calculate_metrics() di train.py)
# =============================================================================
def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    mask = np.abs(y_true) > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_pct": float(mape),
        "r2": float(r2),
        "n_points": len(y_true),
    }


# =============================================================================
# PLOT
# =============================================================================
def save_plot(predictions_df: pd.DataFrame, output_dir: Path, target: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_file = output_dir / "prediction_plot.png"
    plt.figure(figsize=(14, 6))

    for uid, g in predictions_df.groupby("unique_id"):
        g = g.sort_values("ds")
        plt.plot(g["ds"], g["y"], marker="o", label=f"{uid} - Actual", color="black")
        plt.plot(g["ds"], g["naive_last_value"], marker="x", linestyle="--",
                  label=f"{uid} - Naive (last value)", color="tab:blue")
        plt.plot(g["ds"], g["moving_average"], marker="s", linestyle=":",
                  label=f"{uid} - Moving average", color="tab:green")

    plt.title(f"Baseline vs Actual - {target}")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot        : {plot_file}")


# =============================================================================
# SIMPAN OUTPUT (skema kolom SAMA seperti train.py, biar mudah dibandingkan)
# =============================================================================
def save_results(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    metrics_naive: dict,
    metrics_ma: dict,
    cfg: Config,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    metrics_combined = pd.DataFrame([
        {"method": "naive_last_value", **metrics_naive},
        {"method": "moving_average", **metrics_ma},
    ])
    metrics_combined.to_csv(output_dir / "metrics.csv", index=False)

    run_info = {
        "target": cfg.target,
        "train_end": cfg.train_end,
        "test_start": cfg.test_start,
        "test_end": cfg.test_end,
        "actual_test_end": str(predictions_df["ds"].max()) if len(predictions_df) else None,
        "ma_window": cfg.ma_window,
        "freq": metadata["freq"],
        "methods": ["naive_last_value", "moving_average"],
        "metrics": {"naive_last_value": metrics_naive, "moving_average": metrics_ma},
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    logger.info(f"Saved predictions : {output_dir / 'predictions.csv'}")
    logger.info(f"Saved metrics     : {output_dir / 'metrics.csv'}")
    logger.info(f"Saved run info    : {output_dir / 'run_info.json'}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_baseline(cfg: Config) -> dict:
    logger.info("=" * 70)
    logger.info(f"BASELINE FORECASTING - {cfg.target}")
    logger.info("=" * 70)

    df, metadata = load_data(cfg)
    train_df, test_df = split_train_test(df, cfg, metadata)

    logger.info("=" * 70)
    logger.info("MENGHITUNG BASELINE (naive last-value & moving average)")
    logger.info("=" * 70)
    predictions_df = compute_baselines(train_df, test_df, metadata, cfg)

    metrics_naive = calculate_metrics(predictions_df["y"], predictions_df["naive_last_value"])
    metrics_ma = calculate_metrics(predictions_df["y"], predictions_df["moving_average"])

    logger.info("")
    logger.info("=" * 70)
    logger.info("HASIL BASELINE")
    logger.info("=" * 70)
    logger.info(
        f"[naive_last_value] MAE={metrics_naive['mae']:.4f}  RMSE={metrics_naive['rmse']:.4f}  "
        f"MAPE={metrics_naive['mape_pct']:.2f}%  R2={metrics_naive['r2']:.4f}  n={metrics_naive['n_points']}"
    )
    logger.info(
        f"[moving_average]   MAE={metrics_ma['mae']:.4f}  RMSE={metrics_ma['rmse']:.4f}  "
        f"MAPE={metrics_ma['mape_pct']:.2f}%  R2={metrics_ma['r2']:.4f}  n={metrics_ma['n_points']}"
    )

    output_dir = Path(cfg.output_dir)
    save_results(output_dir, predictions_df, metrics_naive, metrics_ma, cfg, metadata)
    save_plot(predictions_df, output_dir, cfg.target)

    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE SELESAI")
    logger.info("=" * 70)

    return {
        "predictions": predictions_df,
        "metrics_naive": metrics_naive,
        "metrics_moving_average": metrics_ma,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Baseline naive/moving-average untuk pembanding NBEATSx (fixed date split)"
    )

    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", required=True, choices=["pressure", "flowrate", "temperature"])

    p.add_argument("--train-end", default="2026-06-30 23:59:59")
    p.add_argument("--test-start", default="2026-07-01 00:00:00")
    p.add_argument("--test-end", default="2026-07-31 23:59:59")

    p.add_argument("--ma-window", type=int, default=7)

    args = p.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target=args.target,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        ma_window=args.ma_window,
    )


if __name__ == "__main__":
    config = parse_args()
    logger.info(f"Konfigurasi:\n{json.dumps(asdict(config), indent=2)}")
    run_baseline(config)