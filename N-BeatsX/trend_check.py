"""
check_trend_shift.py
=====================
Diagnostik: apakah periode TEST (default Juli 2026) punya level/tren yang
BEDA SECARA SISTEMATIS dari histori TRAIN, atau ini cuma noise biasa?

Kenapa ini penting:
    Kalau NBEATSx maupun baseline sederhana (naive/moving average) sama-sama
    dapat R2 negatif di periode test, itu bisa berarti dua hal:
        (a) Datanya memang sulit diprediksi / modelnya kurang bagus, ATAU
        (b) Periode test punya karakteristik yang beda secara sistematis dari
            histori train -- dalam kasus ini TIDAK ADA model (secanggih
            apapun) yang bisa "menebak" pergeseran itu hanya dari pola masa
            lalu, kecuali dikasih tahu penyebabnya (event eksternal, dsb).
    Script ini membantu membedakan (a) dan (b) secara kuantitatif.

Tiga analisis yang dilakukan:
    1. Ekstrapolasi tren linear dari N hari terakhir sebelum train_end,
       diproyeksikan ke periode test. Actual test dibandingkan ke proyeksi
       ini, dinormalisasi jadi z-score pakai volatilitas historis (residual
       rolling trend di dalam TRAIN) -- ini kasih ukuran "seberapa jauh dari
       yang diharapkan", dalam satuan yang bisa dibandingkan antar parameter.
    2. Perbandingan level test terhadap bulan yang sama di tahun-tahun
       sebelumnya (kalau historinya cukup panjang) -- untuk membedakan pola
       musiman berulang vs pergeseran baru yang belum pernah terjadi.
    3. Ringkasan angka + plot untuk inspeksi visual.

Cara pakai (CLI), dijalankan dari dalam folder N-BeatsX/:
    python check_trend_shift.py --data-dir "../dataset/N-BeatsX processed" \
        --output-dir artifacts_trend_check --target pressure \
        --train-end "2026-06-30 23:59:59" \
        --test-start "2026-07-01 00:00:00" \
        --test-end "2026-07-31 23:59:59" \
        --trend-window-days 90 --volatility-window-days 30
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
logger = logging.getLogger("check_trend_shift")


# =============================================================================
# KONFIGURASI
# =============================================================================
@dataclass
class Config:
    data_dir: str
    output_dir: str
    target: str

    # split HARUS sama dengan train.py/baseline.py supaya diagnosisnya
    # relevan dengan periode test yang sama-sama dievaluasi
    train_end: str = "2026-06-30 23:59:59"
    test_start: str = "2026-07-01 00:00:00"
    test_end: str = "2026-07-31 23:59:59"

    # berapa hari terakhir SEBELUM train_end dipakai untuk fit garis tren
    trend_window_days: int = 90
    # berapa hari dipakai untuk hitung volatilitas historis (residual rolling)
    volatility_window_days: int = 30


# =============================================================================
# LOAD DATA & SPLIT (konsisten dengan train.py/baseline.py)
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
        metadata = json.loads(metadata_file.read_text()) if False else json.load(f)

    datetime_col = metadata.get("datetime_col", "DateTime")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])

    if cfg.target not in metadata["target_cols"]:
        raise ValueError(f"Target '{cfg.target}' tidak tersedia. Target tersedia: {metadata['target_cols']}")

    group_col = metadata.get("group_col", "equipment_id")
    df = df.sort_values([group_col, datetime_col]).reset_index(drop=True)
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    logger.info(f"Rows       : {len(df):,}")
    logger.info(f"Equipment  : {df[group_col].nunique()}")
    logger.info(f"Date range : {df[datetime_col].min()} -> {df[datetime_col].max()}")

    return df, metadata


def split_train_test(df: pd.DataFrame, cfg: Config, metadata: dict):
    datetime_col = metadata["datetime_col"]

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

    return train_df, test_df


# =============================================================================
# 1. EKSTRAPOLASI TREN LINEAR
# =============================================================================
def fit_linear_trend(train_df: pd.DataFrame, datetime_col: str, target: str, window_days: int):
    """
    Fit garis lurus (least squares) pada N hari terakhir TRAIN. Return slope,
    intercept, dan basis tanggal (ordinal) supaya bisa dipakai ekstrapolasi.
    """
    train_df = train_df.sort_values(datetime_col)
    cutoff = train_df[datetime_col].max() - pd.Timedelta(days=window_days)
    recent = train_df[train_df[datetime_col] >= cutoff]

    if len(recent) < 5:
        logger.warning(
            f"Cuma {len(recent)} titik data di {window_days} hari terakhir TRAIN -- "
            "fit tren mungkin kurang stabil."
        )

    x = recent[datetime_col].map(pd.Timestamp.toordinal).values.astype(float)
    y = recent[target].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)

    logger.info(
        f"Tren linear ({window_days} hari terakhir TRAIN): slope={slope:.4f} /hari, "
        f"intercept={intercept:.2f}"
    )
    return slope, intercept, recent


def extrapolate_trend(slope: float, intercept: float, dates: pd.Series) -> np.ndarray:
    x = pd.to_datetime(dates).map(pd.Timestamp.toordinal).values.astype(float)
    return slope * x + intercept


# =============================================================================
# 2. VOLATILITAS HISTORIS (buat kontekstualisasi seberapa besar deviasi test)
# =============================================================================
def compute_historical_volatility(train_df: pd.DataFrame, datetime_col: str, target: str, window_days: int) -> float:
    """
    Residual dari rolling mean (bukan dari garis tren tunggal) di seluruh
    TRAIN -- merepresentasikan fluktuasi "wajar" yang historisnya sudah
    pernah terjadi. Dipakai sebagai denominator z-score.
    """
    s = train_df.sort_values(datetime_col).set_index(datetime_col)[target]
    rolling_mean = s.rolling(f"{window_days}D", min_periods=max(3, window_days // 3)).mean()
    residual = (s - rolling_mean).dropna()

    std = float(residual.std())
    if std == 0 or np.isnan(std):
        std = float(s.std()) or 1.0

    logger.info(f"Volatilitas historis (residual rolling {window_days} hari): std={std:.4f}")
    return std


# =============================================================================
# 3. BANDINGKAN TEST VS TREN TEREKSTRAPOLASI
# =============================================================================
def compare_test_vs_trend(test_df: pd.DataFrame, datetime_col: str, target: str, slope: float, intercept: float, historical_std: float) -> pd.DataFrame:
    test_df = test_df.sort_values(datetime_col).copy()
    test_df["trend_extrapolated"] = extrapolate_trend(slope, intercept, test_df[datetime_col])
    test_df["residual"] = test_df[target] - test_df["trend_extrapolated"]
    test_df["z_score"] = test_df["residual"] / historical_std if historical_std else np.nan
    return test_df


# =============================================================================
# 4. PERBANDINGAN SAME-MONTH TAHUN SEBELUMNYA (cek pola musiman vs baru)
# =============================================================================
def compare_same_month_previous_years(df: pd.DataFrame, datetime_col: str, target: str, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    """
    Ambil rata-rata target di bulan yang SAMA (mis. Juli) pada tahun-tahun
    sebelumnya, dibandingkan dengan rata-rata bulan tsb di tahun test.
    """
    target_month = test_start.month
    rows = []
    for year in sorted(df[datetime_col].dt.year.unique()):
        mask = (df[datetime_col].dt.year == year) & (df[datetime_col].dt.month == target_month)
        subset = df.loc[mask, target]
        if len(subset) == 0:
            continue
        rows.append({
            "year": int(year),
            "month": target_month,
            "mean": float(subset.mean()),
            "std": float(subset.std()) if len(subset) > 1 else np.nan,
            "n_points": len(subset),
            "is_test_period": year == test_start.year,
        })
    return pd.DataFrame(rows)


# =============================================================================
# PLOT
# =============================================================================
def save_plot(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_comparison: pd.DataFrame,
    trend_recent: pd.DataFrame,
    datetime_col: str,
    target: str,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- panel 1: histori penuh + garis tren + actual test ---
    ax = axes[0]
    ax.plot(df[datetime_col], df[target], color="black", alpha=0.5, linewidth=0.8, label="Histori (seluruh data)")
    ax.plot(
        trend_recent[datetime_col], trend_recent[target],
        color="tab:blue", linewidth=1.5, label=f"{len(trend_recent)} hari terakhir TRAIN (basis fit tren)"
    )
    ax.plot(
        test_comparison[datetime_col], test_comparison["trend_extrapolated"],
        color="tab:orange", linestyle="--", linewidth=2, label="Tren terekstrapolasi ke periode test"
    )
    ax.plot(
        test_comparison[datetime_col], test_comparison[target],
        color="tab:red", marker="o", linewidth=2, label="Actual periode test"
    )
    ax.set_title(f"Histori vs Ekstrapolasi Tren - {target}")
    ax.set_ylabel(target)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- panel 2: z-score residual per titik test ---
    ax2 = axes[1]
    colors = ["tab:red" if abs(z) > 2 else ("tab:orange" if abs(z) > 1 else "tab:green")
              for z in test_comparison["z_score"]]
    ax2.bar(test_comparison[datetime_col], test_comparison["z_score"], color=colors, width=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(2, color="red", linestyle=":", linewidth=1, label="z = +2 (deviasi besar)")
    ax2.axhline(-2, color="red", linestyle=":", linewidth=1)
    ax2.set_title("Seberapa jauh actual test dari tren yang diharapkan (z-score)")
    ax2.set_ylabel("z-score (residual / volatilitas historis)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plot_file = output_dir / "trend_diagnostic_plot.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {plot_file}")


# =============================================================================
# MAIN
# =============================================================================
def run_check(cfg: Config) -> dict:
    logger.info("=" * 70)
    logger.info(f"CEK TREND SHIFT - {cfg.target}")
    logger.info("=" * 70)

    df, metadata = load_data(cfg)
    datetime_col = metadata["datetime_col"]

    train_df, test_df = split_train_test(df, cfg, metadata)

    slope, intercept, trend_recent = fit_linear_trend(train_df, datetime_col, cfg.target, cfg.trend_window_days)
    historical_std = compute_historical_volatility(train_df, datetime_col, cfg.target, cfg.volatility_window_days)

    test_comparison = compare_test_vs_trend(test_df, datetime_col, cfg.target, slope, intercept, historical_std)

    test_start = pd.Timestamp(cfg.test_start)
    test_end = pd.Timestamp(cfg.test_end)
    same_month_df = compare_same_month_previous_years(df, datetime_col, cfg.target, test_start, test_end)

    # --- ringkasan ---
    mean_abs_z = float(test_comparison["z_score"].abs().mean())
    max_abs_z = float(test_comparison["z_score"].abs().max())
    pct_extreme = float((test_comparison["z_score"].abs() > 2).mean() * 100)
    mean_residual = float(test_comparison["residual"].mean())

    logger.info("")
    logger.info("=" * 70)
    logger.info("RINGKASAN")
    logger.info("=" * 70)
    logger.info(f"Rata-rata |z-score|      : {mean_abs_z:.2f}")
    logger.info(f"Maksimum |z-score|       : {max_abs_z:.2f}")
    logger.info(f"% titik dengan |z| > 2   : {pct_extreme:.1f}%")
    logger.info(f"Rata-rata residual       : {mean_residual:.2f} (positif = test lebih tinggi dari tren, negatif = lebih rendah)")
    logger.info("")
    logger.info("Perbandingan same-month tahun sebelumnya:")
    logger.info("\n" + same_month_df.to_string(index=False))

    # interpretasi otomatis (heuristik sederhana, bukan uji statistik formal)
    if mean_abs_z > 2 or pct_extreme > 50:
        verdict = (
            "INDIKASI KUAT trend shift: actual test menyimpang jauh (rata-rata |z| > 2 atau "
            "sebagian besar titik ekstrem) dari yang diharapkan berdasarkan tren historis."
        )
    elif mean_abs_z > 1:
        verdict = (
            "INDIKASI SEDANG: ada penyimpangan dari tren historis, tapi belum ekstrem. "
            "Bisa jadi kombinasi noise + sedikit pergeseran."
        )
    else:
        verdict = (
            "TIDAK ADA indikasi kuat trend shift dari analisis ini -- penyimpangan test dari "
            "tren historis masih dalam rentang volatilitas normal. Performa model yang jelek "
            "kemungkinan besar bukan karena shift, lebih ke arah keterbatasan model/data."
        )
    logger.info("")
    logger.info(f"KESIMPULAN: {verdict}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_comparison.to_csv(output_dir / "trend_comparison.csv", index=False)
    same_month_df.to_csv(output_dir / "same_month_previous_years.csv", index=False)

    summary = {
        "target": cfg.target,
        "train_end": cfg.train_end,
        "test_start": cfg.test_start,
        "test_end": cfg.test_end,
        "trend_window_days": cfg.trend_window_days,
        "volatility_window_days": cfg.volatility_window_days,
        "trend_slope_per_day": float(slope),
        "trend_intercept": float(intercept),
        "historical_volatility_std": float(historical_std),
        "mean_abs_z_score": mean_abs_z,
        "max_abs_z_score": max_abs_z,
        "pct_points_extreme_z": pct_extreme,
        "mean_residual": mean_residual,
        "verdict": verdict,
    }
    with open(output_dir / "trend_check_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_plot(df, train_df, test_comparison, trend_recent, datetime_col, cfg.target, output_dir)

    logger.info("")
    logger.info(f"Semua output tersimpan di: {output_dir.resolve()}")
    logger.info("=" * 70)

    return {"test_comparison": test_comparison, "same_month": same_month_df, "summary": summary}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Cek apakah periode test punya trend shift dibanding histori TRAIN")

    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", required=True, choices=["pressure", "flowrate", "temperature"])

    p.add_argument("--train-end", default="2026-06-30 23:59:59")
    p.add_argument("--test-start", default="2026-07-01 00:00:00")
    p.add_argument("--test-end", default="2026-07-31 23:59:59")

    p.add_argument("--trend-window-days", type=int, default=90)
    p.add_argument("--volatility-window-days", type=int, default=30)

    args = p.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target=args.target,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        trend_window_days=args.trend_window_days,
        volatility_window_days=args.volatility_window_days,
    )


if __name__ == "__main__":
    config = parse_args()
    logger.info(f"Konfigurasi:\n{json.dumps(asdict(config), indent=2)}")
    run_check(config)