"""
train.py (skenario direct / non-recursive)
===========================================
NBEATSx Forecasting dengan fixed temporal split, TANPA recursive forecasting.

Beda dengan versi sebelumnya (recursive):
    Versi recursive melatih model dengan horizon KECIL (mis. 3 hari), lalu
    memanggil predict() berulang kali per blok horizon, memasukkan hasil
    prediksi sebelumnya sebagai "histori palsu" untuk blok berikutnya. Ini
    rawan COMPOUNDING ERROR -- kesalahan menumpuk dan membesar makin jauh
    forecast-nya dari titik awal.

    Versi ini (direct) melatih model dengan horizon LANGSUNG SEPANJANG
    seluruh periode test (mis. 22 hari), lalu memanggil predict() SATU KALI
    SAJA untuk seluruh periode itu sekaligus. Tidak ada hasil prediksi yang
    dimasukkan balik jadi histori -- jadi tidak ada compounding error.
    Trade-off: horizon yang lebih panjang biasanya bikin model lebih sulit
    dilatih (NBEATSx harus belajar memprediksi jauh ke depan sekaligus),
    tapi ini cara yang lebih "jujur" untuk mengetahui kemampuan asli model,
    tanpa bias dari akumulasi error recursive.

Tujuan:
    TRAIN:
        Semua data sampai dengan train_end (default: 30 Juni 2026).

    TEST:
        Data mulai test_start (default: 1 Juli 2026) sampai data terakhir
        yang tersedia di dataset atau test_end (yang lebih awal).

    Model:
        NBEATSx, dilatih dengan h = jumlah titik test (dihitung otomatis
        dari panjang periode test, kecuali di-override via --horizon).

    Target:
        pressure / flowrate / temperature

    Exogenous:
        FUTURE KNOWN ONLY:
            - hour, dayofweek, day, month, is_weekend
        (flowrate/temperature TIDAK dipakai sebagai hist_exog, sama seperti
        versi sebelumnya -- alasannya sama: nilai masa depannya belum
        diketahui saat forecasting, jadi akan menyebabkan leakage kalau
        dipaksakan.)

    Forecast:
        SATU KALI prediksi langsung untuk seluruh horizon (non-recursive).

    Evaluasi:
        Actual periode test dibandingkan dengan prediksi pada tanggal yang
        sama (inner join berdasarkan unique_id + ds).
    
    Cara Pakai:
        python train_direct.py --data-dir "../dataset/N-BeatsX processed" --output-dir artifacts_direct --target pressure --train-end "2026-06-30 23:59:59" --test-start "2026-07-01 00:00:00" --test-end "2026-07-31 23:59:59" --max-steps 300 --learning-rate 0.001
        
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_direct")


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    data_dir: str
    output_dir: str
    target: str

    # Fixed temporal split -- default SAMA dengan versi recursive, supaya
    # metrik-nya bisa dibandingkan apple-to-apple
    train_end: str = "2026-06-30 23:59:59"
    test_start: str = "2026-07-01 00:00:00"
    test_end: str = "2026-07-31 23:59:59"

    # horizon=None -> otomatis diisi = jumlah titik test yang tersedia
    # (dihitung setelah split, lihat resolve_horizon()). Override manual
    # kalau mau horizon spesifik (mis. lebih pendek dari panjang test).
    horizon: int | None = None
    input_size: int | None = None

    scaler_type: str = "standard"

    max_steps: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 32

    val_check_steps: int = 25
    early_stop_patience_steps: int = 5

    random_seed: int = 42


# =============================================================================
# RANDOM SEED
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =============================================================================
# LOAD DATA (identik dengan versi recursive)
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
# CREATE CALENDAR FEATURES (identik dengan versi recursive)
# =============================================================================
def create_calendar_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])

    df["hour"] = dt.dt.hour.astype(float)
    df["dayofweek"] = dt.dt.dayofweek.astype(float)
    df["day"] = dt.dt.day.astype(float)
    df["month"] = dt.dt.month.astype(float)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

    return df


# =============================================================================
# FIXED TRAIN / TEST SPLIT (identik dengan versi recursive)
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
    logger.info("Test points/equipment:")
    for uid, g in test_df.groupby(group_col):
        logger.info(f"  {uid}: {len(g)} points")

    return train_df, test_df


# =============================================================================
# PREPARE NIXTLA FORMAT (identik dengan versi recursive)
# =============================================================================
def prepare_nixtla_dataframe(df: pd.DataFrame, metadata: dict, target: str) -> pd.DataFrame:
    datetime_col = metadata["datetime_col"]
    group_col = metadata["group_col"]

    df = df.copy()
    df = create_calendar_features(df, datetime_col)

    nf_df = pd.DataFrame()
    nf_df["unique_id"] = df[group_col].astype(str)
    nf_df["ds"] = pd.to_datetime(df[datetime_col])
    nf_df["y"] = pd.to_numeric(df[target], errors="coerce")

    future_exog = ["hour", "dayofweek", "day", "month", "is_weekend"]
    for col in future_exog:
        nf_df[col] = pd.to_numeric(df[col], errors="coerce")

    nf_df = nf_df.dropna(subset=["unique_id", "ds", "y"])
    nf_df = nf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return nf_df


# =============================================================================
# RESOLVE HORIZON -- otomatis = panjang periode test (kecuali di-override)
# =============================================================================
def resolve_horizon(test_nf: pd.DataFrame, cfg: Config, metadata: dict) -> int:
    """
    Untuk skenario direct (non-recursive), horizon model HARUS sama dengan
    jumlah titik yang ingin diprediksi sekaligus. Default: otomatis dihitung
    dari jumlah tanggal unik di periode test hasil split. Bisa di-override
    manual lewat cfg.horizon kalau kamu sengaja mau horizon lebih pendek
    (evaluasi hanya sebagian awal test) atau nilai spesifik lain.
    """
    n_test_dates = test_nf["ds"].nunique()

    if cfg.horizon is not None:
        horizon = cfg.horizon
        if horizon > n_test_dates:
            logger.warning(
                f"--horizon ({horizon}) lebih besar dari jumlah titik test yang tersedia "
                f"({n_test_dates}). Sebagian hasil forecast tidak akan punya nilai aktual "
                "untuk dibandingkan (akan otomatis di-skip saat evaluasi)."
            )
        elif horizon < n_test_dates:
            logger.warning(
                f"--horizon ({horizon}) lebih pendek dari jumlah titik test yang tersedia "
                f"({n_test_dates}). Evaluasi hanya mencakup {horizon} hari pertama periode test."
            )
    else:
        horizon = n_test_dates
        logger.info(f"horizon otomatis diisi = jumlah titik test yang tersedia ({n_test_dates})")

    return horizon


# =============================================================================
# VALIDATE TRAIN SERIES (identik dengan versi recursive)
# =============================================================================
def validate_training_series(train_nf: pd.DataFrame, input_size: int, horizon: int):
    minimum_required = input_size + horizon
    lengths = train_nf.groupby("unique_id").size()

    logger.info(f"Minimum training samples per equipment: {minimum_required}")

    too_short = lengths[lengths < minimum_required]
    if len(too_short) > 0:
        raise ValueError(
            "Ada equipment yang datanya terlalu pendek:\n"
            f"{too_short.to_dict()}\n"
            "Catatan: pada skenario direct, horizon = panjang periode test, jadi "
            "input_size + horizon bisa jauh lebih besar dibanding skenario recursive. "
            "Kalau ini masalahnya, coba perkecil input_size, atau perpendek periode "
            "test (--test-end), atau kembali ke skenario recursive."
        )


# =============================================================================
# METRICS (identik dengan versi recursive)
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
# TRAIN NBEATSX (h = horizon penuh, bukan blok kecil)
# =============================================================================
def train_model(train_nf: pd.DataFrame, horizon: int, input_size: int, cfg: Config, freq: str):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    logger.info("=" * 70)
    logger.info("TRAINING NBEATSx (DIRECT, h = horizon penuh)")
    logger.info("=" * 70)
    logger.info(f"Target       : {cfg.target}")
    logger.info(f"Input size   : {input_size}")
    logger.info(f"Horizon      : {horizon}  (model memprediksi seluruh periode ini SEKALIGUS)")
    logger.info(f"Max steps    : {cfg.max_steps}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info("Historical exogenous: NONE")
    logger.info("Future exogenous: ['hour', 'dayofweek', 'day', 'month', 'is_weekend']")

    model = NBEATSx(
        h=horizon,
        input_size=input_size,
        futr_exog_list=["hour", "dayofweek", "day", "month", "is_weekend"],
        hist_exog_list=None,
        scaler_type=cfg.scaler_type,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        val_check_steps=cfg.val_check_steps,
        early_stop_patience_steps=cfg.early_stop_patience_steps,
        random_seed=cfg.random_seed,
        alias="NBEATSx",
    )

    nf = NeuralForecast(models=[model], freq=freq)

    # val_size kecil hanya untuk early stopping/monitoring selama training,
    # BUKAN evaluasi akhir -- evaluasi akhir tetap dari periode test terpisah.
    val_size = min(horizon, max(1, len(train_nf) // 10))
    nf.fit(df=train_nf, val_size=val_size)

    logger.info("Training selesai.")
    return nf


# =============================================================================
# BUILD FUTURE DATAFRAME (identik logic-nya dengan versi recursive, tapi
# dipanggil SATU KALI SAJA untuk seluruh horizon, bukan per-blok)
# =============================================================================
def build_future_dataframe(history_nf: pd.DataFrame, steps: int, freq: str) -> pd.DataFrame:
    rows = []
    for uid, g in history_nf.groupby("unique_id"):
        g = g.sort_values("ds")
        last_date = g["ds"].max()

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)

        future = pd.DataFrame({"unique_id": uid, "ds": future_dates})
        future["hour"] = future["ds"].dt.hour.astype(float)
        future["dayofweek"] = future["ds"].dt.dayofweek.astype(float)
        future["day"] = future["ds"].dt.day.astype(float)
        future["month"] = future["ds"].dt.month.astype(float)
        future["is_weekend"] = (future["ds"].dt.dayofweek >= 5).astype(float)

        rows.append(future)

    if not rows:
        raise ValueError("Tidak dapat membuat future dataframe.")

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# DIRECT FORECAST (satu kali prediksi, TANPA loop / TANPA feed-back)
# =============================================================================
def direct_forecast(nf, train_nf: pd.DataFrame, test_nf: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    """
    Beda dengan recursive_forecast(): fungsi ini memanggil nf.predict() TEPAT
    SATU KALI untuk seluruh horizon sekaligus. Tidak ada hasil prediksi yang
    dimasukkan balik jadi histori, jadi tidak ada compounding error.
    """
    logger.info("=" * 70)
    logger.info("DIRECT FORECAST (non-recursive, satu kali prediksi)")
    logger.info("=" * 70)

    future_df = build_future_dataframe(history_nf=train_nf, steps=horizon, freq=freq)
    logger.info(f"future_df dibangun: {future_df['ds'].min()} -> {future_df['ds'].max()} ({horizon} titik)")

    forecast_df = nf.predict(df=train_nf, futr_df=future_df)
    forecast_df = forecast_df[["unique_id", "ds", "NBEATSx"]].rename(columns={"NBEATSx": "prediction"})

    logger.info(f"Forecast selesai: {len(forecast_df):,} baris prediksi dihasilkan")
    return forecast_df


# =============================================================================
# EVALUATION -- inner join predictions vs actual test (identik dengan versi
# recursive), otomatis abaikan tanggal predict yang tidak ada actual-nya
# (relevan kalau horizon di-override lebih besar dari panjang test)
# =============================================================================
def evaluate_forecast(predictions: pd.DataFrame, test_nf: pd.DataFrame):
    merged = pd.merge(
        test_nf[["unique_id", "ds", "y"]],
        predictions,
        on=["unique_id", "ds"],
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("Tidak ada timestamp yang berhasil di-merge antara prediction dan actual.")

    if len(merged) != len(test_nf):
        logger.warning(f"Jumlah actual test : {len(test_nf)}")
        logger.warning(f"Jumlah ter-evaluasi: {len(merged)}  (sisanya tidak ter-cover oleh horizon)")

    metrics = calculate_metrics(merged["y"].values, merged["prediction"].values)
    return merged, metrics


# =============================================================================
# SAVE RESULTS (skema kolom sama seperti versi recursive & baseline.py,
# supaya gampang dibandingkan berdampingan)
# =============================================================================
def save_results(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    metrics: dict,
    cfg: Config,
    horizon: int,
    input_size: int,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    run_info = {
        "method": "direct_non_recursive",
        "target": cfg.target,
        "train_end": cfg.train_end,
        "test_start": cfg.test_start,
        "test_end": cfg.test_end,
        "actual_test_end": str(predictions_df["ds"].max()) if len(predictions_df) else None,
        "horizon": horizon,
        "input_size": input_size,
        "freq": metadata["freq"],
        "scaler_type": cfg.scaler_type,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "random_seed": cfg.random_seed,
        "hist_exog_list": [],
        "futr_exog_list": ["hour", "dayofweek", "day", "month", "is_weekend"],
        "data_leakage_policy": (
            "Prediksi dilakukan satu kali langsung untuk seluruh horizon "
            "(non-recursive). Tidak ada nilai actual test maupun hasil "
            "prediksi yang dimasukkan balik ke histori."
        ),
        "metrics": metrics,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    logger.info(f"Saved predictions : {output_dir / 'predictions.csv'}")
    logger.info(f"Saved metrics     : {output_dir / 'metrics.csv'}")
    logger.info(f"Saved run info    : {output_dir / 'run_info.json'}")


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
        plt.plot(g["ds"], g["y"], marker="o", label=f"{uid} - Actual")
        plt.plot(g["ds"], g["prediction"], marker="x", linestyle="--", label=f"{uid} - NBEATSx (direct)")

    plt.title(f"NBEATSx Forecast vs Actual (DIRECT, non-recursive) - {target}")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot        : {plot_file}")


# =============================================================================
# SAVE MODEL
# =============================================================================
def save_model(nf, output_dir: Path) -> None:
    model_dir = output_dir / "nf_checkpoint"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        nf.save(path=str(model_dir), overwrite=True, save_dataset=False)
        logger.info(f"Saved model       : {model_dir}")
    except Exception as e:
        logger.warning(f"Gagal menyimpan NeuralForecast checkpoint: {e}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def run_training(cfg: Config) -> dict:
    set_seed(cfg.random_seed)

    logger.info("CONFIGURATION:")
    logger.info(json.dumps(asdict(cfg), indent=2))
    logger.info("")

    logger.info("=" * 70)
    logger.info(f"NBEATSx FORECASTING (DIRECT / NON-RECURSIVE) - {cfg.target}")
    logger.info("=" * 70)

    df, metadata = load_data(cfg)
    datetime_col = metadata["datetime_col"]
    freq = metadata["freq"]

    train_df, test_df = split_train_test(df, cfg, metadata)

    train_nf = prepare_nixtla_dataframe(train_df, metadata, cfg.target)
    test_nf = prepare_nixtla_dataframe(test_df, metadata, cfg.target)

    horizon = resolve_horizon(test_nf, cfg, metadata)
    input_size = cfg.input_size if cfg.input_size is not None else metadata["input_size"]

    logger.info(f"Input size (dipakai) : {input_size}")
    logger.info(f"Horizon (dipakai)    : {horizon}")

    validate_training_series(train_nf, input_size, horizon)

    nf = train_model(train_nf=train_nf, horizon=horizon, input_size=input_size, cfg=cfg, freq=freq)

    predictions = direct_forecast(nf=nf, train_nf=train_nf, test_nf=test_nf, horizon=horizon, freq=freq)

    predictions_df, metrics = evaluate_forecast(predictions, test_nf)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL TEST RESULTS (DIRECT / NON-RECURSIVE)")
    logger.info("=" * 70)
    logger.info(f"MAE      : {metrics['mae']:.6f}")
    logger.info(f"MSE      : {metrics['mse']:.6f}")
    logger.info(f"RMSE     : {metrics['rmse']:.6f}")
    logger.info(f"MAPE (%) : {metrics['mape_pct']:.6f}")
    logger.info(f"R2       : {metrics['r2']:.6f}")
    logger.info(f"N Points : {metrics['n_points']}")

    output_dir = Path(cfg.output_dir)
    save_results(output_dir, predictions_df, metrics, cfg, horizon, input_size, metadata)
    save_plot(predictions_df, output_dir, cfg.target)
    save_model(nf, output_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PREDICTIONS")
    logger.info("=" * 70)
    logger.info("\n" + predictions_df.to_string(index=False))

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING + TESTING SELESAI (DIRECT / NON-RECURSIVE)")
    logger.info("=" * 70)

    return {"model": nf, "predictions": predictions_df, "metrics": metrics}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="NBEATSx forecasting dengan fixed train/test split, DIRECT (non-recursive)."
    )

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", required=True, choices=["pressure", "flowrate", "temperature"])

    parser.add_argument("--train-end", default="2026-06-30 23:59:59")
    parser.add_argument("--test-start", default="2026-07-01 00:00:00")
    parser.add_argument("--test-end", default="2026-07-31 23:59:59")

    parser.add_argument("--horizon", type=int, default=None,
                         help="override horizon; default: otomatis = panjang periode test")
    parser.add_argument("--input-size", type=int, default=None)

    parser.add_argument("--scaler-type", default="standard")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-check-steps", type=int, default=25)
    parser.add_argument("--early-stop-patience-steps", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target=args.target,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        horizon=args.horizon,
        input_size=args.input_size,
        scaler_type=args.scaler_type,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=args.early_stop_patience_steps,
        random_seed=args.random_seed,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config = parse_args()
    run_training(config)