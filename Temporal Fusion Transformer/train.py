import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import lightning as L

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# =========================================================
# CONFIG
# =========================================================

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3

# TFT window
ENCODER_LENGTH = 96      # histori 96 menit
PREDICTION_LENGTH = 24   # prediksi 24 menit

L.seed_everything(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_CSV = BASE_DIR.parent / 'dataset' / 'TFT processed' / 'tft_train.csv'
EVAL_CSV  = BASE_DIR.parent / 'dataset' / 'TFT processed' / 'tft_july_2026.csv'

ARTIFACT_DIR = BASE_DIR / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_CSV, parse_dates=['time'])
eval_df  = pd.read_csv(EVAL_CSV, parse_dates=['time'])

print('Train shape:', train_df.shape)
print('Eval shape :', eval_df.shape)

# gabungkan untuk membuat dataset kontinu
full_df = (
    pd.concat([train_df, eval_df], ignore_index=True)
    .sort_values('time_idx')
    .reset_index(drop=True)
)

# =========================================================
# TIME SERIES DATASET
# =========================================================

training_cutoff = train_df['time_idx'].max()

training = TimeSeriesDataSet(
    full_df[full_df.time_idx <= training_cutoff],

    time_idx='time_idx',
    target='pressure',
    group_ids=['group_id'],

    max_encoder_length=ENCODER_LENGTH,
    max_prediction_length=PREDICTION_LENGTH,

    static_categoricals=['group_id'],

    time_varying_known_reals=[
        'time_idx',
        'minute',
        'hour',
        'day',
        'day_of_week',
        'month',
        'is_weekend'
    ],

    time_varying_unknown_reals=['pressure'],

    target_normalizer=None,

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# validation = Juli 2026
validation = TimeSeriesDataSet.from_dataset(
    training,
    full_df,
    predict=True,
    stop_randomization=True
)

# dataloader
train_loader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=0
)

val_loader = validation.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches  : {len(val_loader)}')

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-4,
    mode='min'
)

# =========================================================
# TRAINER
# =========================================================

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='auto',
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop],
    logger=False,
    enable_checkpointing=True
)

# =========================================================
# TFT MODEL
# =========================================================

tft = TemporalFusionTransformer.from_dataset(
    training,

    learning_rate=LEARNING_RATE,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,

    loss=QuantileLoss(),

    log_interval=10,
    reduce_on_plateau_patience=3,
)

print(f'Number of parameters: {tft.size()/1e3:.1f}k')

# =========================================================
# TRAIN
# =========================================================

print('\\nStarting TFT training...')

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = ARTIFACT_DIR / 'tft_model.ckpt'

trainer.save_checkpoint(MODEL_PATH)

print(f'\\nModel saved: {MODEL_PATH}')

# =========================================================
# PREDICTION
# =========================================================

print('\\nGenerating predictions...')

# prediksi yang sudah aligned dengan target
raw_predictions = tft.predict(
    val_loader,
    mode='raw',
    return_x=True
)

# ambil prediksi median (quantile 0.5)
y_pred = raw_predictions.output.prediction[..., 1].cpu().numpy().reshape(-1)

# ambil target yang sesuai
y_true = raw_predictions.x['decoder_target'].cpu().numpy().reshape(-1)

# buang NaN jika ada
mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
y_true = y_true[mask]
y_pred = y_pred[mask]

print('Aligned samples:', len(y_true))
print('y_true shape:', y_true.shape)
print('y_pred shape:', y_pred.shape)

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-8
mape = np.mean(
    np.abs((y_true - y_pred) / (y_true + epsilon))
) * 100

r2 = r2_score(y_true, y_pred)

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'R-Squared'],
    'Value': [mae, mse, rmse, mape, r2]
})

print('\\n' + '='*60)
print('TEMPORAL FUSION TRANSFORMER RESULTS')
print('='*60)
print(results.to_string(index=False))

# =========================================================
# SAVE RESULTS
# =========================================================

# =========================================================
# SAVE RESULTS
# =========================================================

results.to_csv(ARTIFACT_DIR / 'metrics.csv', index=False)

# gunakan jumlah sample yang benar-benar aligned
n_samples = len(y_true)

# ambil timestamp sebanyak sample aligned
time_aligned = eval_df['time'].iloc[:n_samples].reset_index(drop=True)

pred_df = pd.DataFrame({
    'time': time_aligned,
    'actual': y_true,
    'predicted': y_pred
})

pred_df.to_csv(ARTIFACT_DIR / 'predictions.csv', index=False)

print('\\nSaved:')
print('- metrics.csv')
print('- predictions.csv')
print(f'- aligned samples: {n_samples}')

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14,5))

n_plot = min(500, len(pred_df))

plt.plot(
    pred_df['actual'].iloc[:n_plot],
    label='Actual',
    linewidth=1.5
)

plt.plot(
    pred_df['predicted'].iloc[:n_plot],
    label='TFT Prediction',
    linewidth=1.5
)

plt.title('TFT: Actual vs Predicted Pressure (July 2026)')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'prediction_plot.png', dpi=300)
plt.show()

print('\\nTraining completed successfully!')