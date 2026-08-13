"""
train.py
========
Training & evaluasi model NBEATSx (via library `neuralforecast`) untuk satu
parameter target (pressure / flowrate / temperature), menggunakan output dari
preproc.py (full_clean.parquet + metadata.json).

Alur:
    1. Evaluasi: cross_validation() -- fit di porsi train+val, uji di porsi
       test yang di-hold-out -- menghasilkan metrics.csv, predictions.csv,
       prediction_plot.png.
    2. Refit final: fit ulang model dengan hyperparameter sama di SELURUH
       data (train+val+test) supaya modelnya benar-benar "melihat" histori
       terbaru sebelum dipakai forecast ke depan sungguhan -- menghasilkan
       nbeatsx_model.ckpt (bobot akhir) dan nbeatsx_best.ckpt (checkpoint
       dengan validation loss terbaik selama refit ini).
    3. scaler.pkl: statistik referensi (mean/std per equipment per target,
       dihitung dari porsi train) -- BUKAN dipakai oleh model (NBEATSx sudah
       menormalisasi datanya sendiri lewat parameter scaler_type), murni
       untuk keperluan inspeksi/reporting manual di luar training.

Target lain (flowrate/temperature/pressure) otomatis dipakai sebagai
hist_exog (exogenous yang hanya diketahui historis), dan fitur kalender dari
preproc.py (hour/dayofweek/day/month/is_weekend) dipakai sebagai futr_exog
(exogenous yang diketahui di masa depan).

Cara pakai (CLI), dijalankan dari dalam folder N-BeatsX/:
    # satu target saja, output flat ke artifacts/ (sesuai struktur project)
    python train.py --data-dir "../dataset/N-BeatsX processed" \
        --output-dir artifacts --target pressure \
        --max-steps 300 --learning-rate 0.001

    # semua target sekaligus (loop otomatis), tiap target dapat subfolder
    python train.py --data-dir "../dataset/N-BeatsX processed" \
        --output-dir artifacts --target all --max-steps 300

Bisa juga dipakai sebagai module:
    from train import Config, run_training
    cfg = Config(data_dir="../dataset/N-BeatsX processed", output_dir="artifacts", target="pressure")
    result = run_training(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from preproc import to_nixtla_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# =============================================================================
# 1. KONFIGURASI
# =============================================================================
@dataclass
class Config:
    # --- I/O ---
    data_dir: str    # folder output preproc.py, berisi full_clean.parquet + metadata.json
    output_dir: str  # folder artifacts, tempat model/metrics/plot disimpan
    target: str      # "pressure" / "flowrate" / "temperature" / dst. sesuai target_cols di metadata

    # --- override parameter windowing (default: ambil dari metadata.json hasil preproc) ---
    horizon: int | None = None
    input_size: int | None = None

    # --- hyperparameter model & training ---
    scaler_type: str = "standard"   # "standard", "robust", "minmax", "identity", dll (lihat dok neuralforecast)
    max_steps: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 32
    val_check_steps: int = 25
    early_stop_patience_steps: int = 5
    random_seed: int = 1

    # --- evaluasi ---
    n_test_windows: int = 2  # panjang test_size = horizon * n_test_windows (dalam satuan timestep)

    def __post_init__(self):
        if self.n_test_windows < 1:
            raise ValueError("n_test_windows minimal 1")


# =============================================================================
# 2. LOAD DATA HASIL PREPROC
# =============================================================================
def load_preproc_output(cfg: Config) -> tuple[pd.DataFrame, dict]:
    data_dir = Path(cfg.data_dir)
    df_path = data_dir / "full_clean.parquet"
    meta_path = data_dir / "metadata.json"

    if not df_path.exists():
        raise FileNotFoundError(
            f"Tidak ditemukan {df_path}. Pastikan --data-dir menunjuk ke folder output "
            "preproc.py (yang berisi full_clean.parquet & metadata.json)."
        )

    logger.info(f"Load data dari: {df_path}")
    df = pd.read_parquet(df_path)
    metadata = json.loads(meta_path.read_text())

    if cfg.target not in metadata["target_cols"]:
        raise ValueError(
            f"Target '{cfg.target}' tidak ada di metadata (target_cols: {metadata['target_cols']}). "
            "Cek lagi nama parameter yang ingin diforecast."
        )

    logger.info(f"  -> {len(df):,} baris, {df['equipment_id'].nunique()} equipment, target='{cfg.target}'")
    return df, metadata


# =============================================================================
# 3. SUSUN EXOGENOUS LIST & FORMAT NIXTLA
# =============================================================================
def build_exog_lists(metadata: dict, target: str) -> tuple[list, list]:
    """
    hist_exog: parameter lain (target_cols selain target ini sendiri) -- hanya
        diketahui secara historis, tidak diketahui di masa depan.
    futr_exog: fitur kalender dari metadata['time_varying_known_reals'] -- bisa
        dihitung untuk tanggal manapun (masa lalu maupun masa depan), minus
        'time_idx' karena sudah terwakili oleh kolom 'ds' di format Nixtla.
    """
    hist_exog = [c for c in metadata["target_cols"] if c != target]
    futr_exog = [c for c in metadata["time_varying_known_reals"] if c != "time_idx"]
    return hist_exog, futr_exog


def build_nixtla_df(df: pd.DataFrame, metadata: dict, target: str, hist_exog: list, futr_exog: list) -> pd.DataFrame:
    nf_df = to_nixtla_format(
        df, target_col=target, datetime_col=metadata["datetime_col"],
        exog_cols=hist_exog + futr_exog,
    )
    nf_df["unique_id"] = nf_df["unique_id"].astype(str)
    nf_df = nf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return nf_df


# =============================================================================
# 4. VALIDASI PANJANG DATA VS PARAMETER EVALUASI
# =============================================================================
def resolve_horizon_input_size(metadata: dict, cfg: Config) -> tuple[int, int]:
    horizon = cfg.horizon if cfg.horizon is not None else metadata["horizon"]
    input_size = cfg.input_size if cfg.input_size is not None else metadata["input_size"]
    return horizon, input_size


def check_series_length(nf_df: pd.DataFrame, horizon: int, input_size: int, n_test_windows: int) -> None:
    test_size = horizon * n_test_windows
    val_size = horizon
    min_required = input_size + val_size + test_size

    lengths = nf_df.groupby("unique_id").size()
    too_short = lengths[lengths < min_required]
    if len(too_short):
        raise ValueError(
            f"Equipment berikut punya data lebih pendek dari input_size + val_size + test_size "
            f"({min_required} timestep): {too_short.to_dict()}. Perkecil --n-test-windows, "
            "--horizon, atau --input-size, atau pakai freq yang lebih kasar saat preproc."
        )
    logger.info(f"  -> panjang data cukup untuk semua {len(lengths)} equipment (butuh >= {min_required} timestep)")


# =============================================================================
# 5. SCALER REFERENSI (bukan dipakai model, murni untuk reporting)
# =============================================================================
def compute_reference_scaler(nf_df: pd.DataFrame, target: str, test_size: int) -> dict:
    """Mean/std per equipment, dihitung dari porsi TRAIN saja (sebelum test_size terakhir)."""
    scalers = {}
    for uid, g in nf_df.groupby("unique_id"):
        g = g.sort_values("ds")
        train_part = g.iloc[: max(len(g) - test_size, 1)]
        mean, std = float(train_part["y"].mean()), float(train_part["y"].std())
        scalers[uid] = {"target": target, "mean": mean, "std": std if std and not np.isnan(std) else 1.0}
    return scalers


# =============================================================================
# 6. METRIK EVALUASI
# =============================================================================
def compute_metrics(cv_df: pd.DataFrame, model_col: str = "NBEATSx") -> pd.DataFrame:
    y_true = cv_df["y"].values
    y_pred = cv_df[model_col].values

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = np.abs(y_true) > 1e-6  # hindari pembagian dengan nilai mendekati nol
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
        logger.warning("Semua nilai aktual mendekati nol -- MAPE tidak dihitung (pembagian tidak stabil)")

    return pd.DataFrame([{"mae": mae, "rmse": rmse, "mape_pct": mape, "n_points": len(cv_df)}])


# =============================================================================
# 7. PLOT PREDIKSI VS AKTUAL
# =============================================================================
def save_prediction_plot(cv_df: pd.DataFrame, target: str, output_path: Path, model_col: str = "NBEATSx") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for uid, g in cv_df.groupby("unique_id"):
        g = g.sort_values("ds")
        ax.plot(g["ds"], g["y"], marker="o", label=f"{uid} - aktual", color="black", linewidth=1.5)
        ax.plot(g["ds"], g[model_col], marker="x", linestyle="--", label=f"{uid} - prediksi NBEATSx", color="tab:red")

    ax.set_title(f"Prediksi vs Aktual ({target}) -- evaluasi hold-out test")
    ax.set_xlabel("Waktu")
    ax.set_ylabel(target)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 8. REFIT FINAL DI SELURUH DATA (untuk model yang benar-benar dipakai forecast ke depan)
# =============================================================================
def refit_final_model(nf_df: pd.DataFrame, hist_exog: list, futr_exog: list, horizon: int, input_size: int,
                       freq: str, cfg: Config, checkpoint_dir: Path):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from pytorch_lightning.callbacks import ModelCheckpoint

    logger.info("Refit final model di SELURUH data (train+val+test) untuk deployment")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="nbeatsx_best",
        monitor="valid_loss", mode="min", save_top_k=1,
    )

    model = NBEATSx(
        h=horizon, input_size=input_size,
        futr_exog_list=futr_exog, hist_exog_list=hist_exog,
        scaler_type=cfg.scaler_type,
        max_steps=cfg.max_steps, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        val_check_steps=cfg.val_check_steps, early_stop_patience_steps=cfg.early_stop_patience_steps,
        random_seed=cfg.random_seed, alias="NBEATSx",
        callbacks=[best_ckpt_cb], enable_checkpointing=True,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=nf_df, val_size=horizon)  # val_size kecil hanya untuk early stopping/monitoring, bukan evaluasi akhir

    return nf, best_ckpt_cb


def save_model_artifacts(nf, checkpoint_dir: Path, output_dir: Path) -> None:
    """
    Simpan model final ke output_dir:
      - nbeatsx_model.ckpt : salinan bobot akhir hasil refit (untuk kemudahan akses langsung)
      - nbeatsx_best.ckpt  : checkpoint dengan validation loss terbaik selama refit (dari callback)
      - nf_checkpoint/     : folder lengkap ala neuralforecast (checkpoint + configuration.pkl +
                              alias_to_model.pkl) -- WAJIB dipakai (bukan file .ckpt di atas) kalau
                              mau me-load ulang model lewat NeuralForecast.load(path=...)
    """
    nf_checkpoint_dir = output_dir / "nf_checkpoint"
    nf.save(path=str(nf_checkpoint_dir), overwrite=True, save_dataset=False)

    # salin file .ckpt utama dari folder nf_checkpoint ke nama flat "nbeatsx_model.ckpt"
    # (HANYA untuk kemudahan melihat ukuran/tanggal model; untuk reload sesungguhnya tetap
    # pakai NeuralForecast.load(path=".../nf_checkpoint") supaya configuration.pkl ikut terbaca)
    ckpt_candidates = list(nf_checkpoint_dir.glob("*.ckpt"))
    if ckpt_candidates:
        shutil.copy(ckpt_candidates[0], output_dir / "nbeatsx_model.ckpt")

    # NB: atribut ModelCheckpoint.best_model_path tidak selalu ter-update lewat referensi
    # Python biasa di setup ini (isu internal PyTorch Lightning/neuralforecast), jadi kita
    # cari langsung file-nya dari folder checkpoint tempat callback menulis, bukan bergantung
    # pada atribut tersebut.
    best_candidates = list(checkpoint_dir.glob("nbeatsx_best*.ckpt"))
    if best_candidates:
        shutil.copy(best_candidates[0], output_dir / "nbeatsx_best.ckpt")
    else:
        logger.warning(
            f"Checkpoint 'best' tidak ditemukan di {checkpoint_dir} -- kemungkinan training terlalu "
            "singkat (validation belum sempat jalan). nbeatsx_model.ckpt tetap tersedia sebagai bobot akhir."
        )

    logger.info(f"  -> nbeatsx_model.ckpt, nbeatsx_best.ckpt (kalau ada), nf_checkpoint/ tersimpan di {output_dir}")


# =============================================================================
# MAIN PIPELINE (satu target)
# =============================================================================
def run_training(cfg: Config) -> dict:
    logger.info("=" * 70)
    logger.info(f"TRAINING NBEATSx -- target: {cfg.target}")
    logger.info("=" * 70)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_preproc_output(cfg)
    horizon, input_size = resolve_horizon_input_size(metadata, cfg)
    freq = metadata["freq"]

    hist_exog, futr_exog = build_exog_lists(metadata, cfg.target)
    logger.info(f"hist_exog_list: {hist_exog}")
    logger.info(f"futr_exog_list: {futr_exog}")

    nf_df = build_nixtla_df(df, metadata, cfg.target, hist_exog, futr_exog)
    check_series_length(nf_df, horizon, input_size, cfg.n_test_windows)

    # --- 1. evaluasi (cross_validation) ---
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    test_size = horizon * cfg.n_test_windows
    val_size = horizon
    logger.info(f"Evaluasi via cross_validation: val_size={val_size}, test_size={test_size}")

    eval_model = NBEATSx(
        h=horizon, input_size=input_size,
        futr_exog_list=futr_exog, hist_exog_list=hist_exog,
        scaler_type=cfg.scaler_type,
        max_steps=cfg.max_steps, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        val_check_steps=cfg.val_check_steps, early_stop_patience_steps=cfg.early_stop_patience_steps,
        random_seed=cfg.random_seed, alias="NBEATSx",
    )
    nf_eval = NeuralForecast(models=[eval_model], freq=freq)
    cv_df = nf_eval.cross_validation(df=nf_df, val_size=val_size, test_size=test_size, step_size=horizon, n_windows=None)

    metrics_df = compute_metrics(cv_df)
    logger.info(f"Hasil evaluasi:\n{metrics_df.to_string(index=False)}")

    cv_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    save_prediction_plot(cv_df, cfg.target, output_dir / "prediction_plot.png")

    # --- 2. scaler referensi ---
    scalers = compute_reference_scaler(nf_df, cfg.target, test_size)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scalers, f)

    # --- 3. refit final di seluruh data, simpan model deployable ---
    best_ckpt_dir = output_dir / "_tmp_best_ckpt"
    nf_final, _ = refit_final_model(nf_df, hist_exog, futr_exog, horizon, input_size, freq, cfg, best_ckpt_dir)
    save_model_artifacts(nf_final, best_ckpt_dir, output_dir)
    shutil.rmtree(best_ckpt_dir, ignore_errors=True)

    # --- ringkasan run ---
    run_info = {
        "target": cfg.target,
        "hist_exog_list": hist_exog,
        "futr_exog_list": futr_exog,
        "horizon": horizon,
        "input_size": input_size,
        "freq": freq,
        "scaler_type": cfg.scaler_type,
        "max_steps": cfg.max_steps,
        "n_test_windows": cfg.n_test_windows,
        "metrics": metrics_df.to_dict(orient="records")[0],
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info(f"TRAINING SELESAI -- target: {cfg.target}")
    logger.info(f"Artifacts tersimpan di: {output_dir.resolve()}")
    logger.info("=" * 70)

    return {"cv_df": cv_df, "metrics": metrics_df, "nf_final": nf_final, "run_info": run_info}


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def _parse_args() -> tuple[Config, str]:
    p = argparse.ArgumentParser(description="Training model NBEATSx per parameter target")

    p.add_argument("--data-dir", required=True, help="folder output preproc.py (berisi full_clean.parquet, metadata.json)")
    p.add_argument("--output-dir", required=True, help="folder artifacts tempat model/metrics/plot disimpan")
    p.add_argument(
        "--target", required=True,
        help="nama target ('pressure'/'flowrate'/'temperature'/dst.), atau 'all' untuk loop semua target di metadata",
    )

    p.add_argument("--horizon", type=int, default=None, help="override horizon dari metadata.json")
    p.add_argument("--input-size", type=int, default=None, help="override input_size dari metadata.json")

    p.add_argument("--scaler-type", default="standard")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-check-steps", type=int, default=25)
    p.add_argument("--early-stop-patience-steps", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--n-test-windows", type=int, default=2)

    args = p.parse_args()

    base_kwargs = dict(
        data_dir=args.data_dir,
        horizon=args.horizon,
        input_size=args.input_size,
        scaler_type=args.scaler_type,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=args.early_stop_patience_steps,
        random_seed=args.random_seed,
        n_test_windows=args.n_test_windows,
    )
    return base_kwargs, args.target, args.output_dir


if __name__ == "__main__":
    base_kwargs, target_arg, output_dir_arg = _parse_args()

    if target_arg.lower() == "all":
        # cek target_cols yang tersedia dari metadata.json tanpa perlu load ulang manual
        meta_path = Path(base_kwargs["data_dir"]) / "metadata.json"
        available_targets = json.loads(meta_path.read_text())["target_cols"]
        logger.info(f"Mode --target all -- akan training untuk: {available_targets}")

        for t in available_targets:
            cfg = Config(output_dir=str(Path(output_dir_arg) / t), target=t, **base_kwargs)
            logger.info(f"Konfigurasi target '{t}':\n{json.dumps(asdict(cfg), indent=2, default=str)}")
            run_training(cfg)
    else:
        cfg = Config(output_dir=output_dir_arg, target=target_arg, **base_kwargs)
        logger.info(f"Konfigurasi:\n{json.dumps(asdict(cfg), indent=2, default=str)}")
        run_training(cfg)