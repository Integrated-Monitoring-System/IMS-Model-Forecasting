import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_tcn import TCN

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# CONFIG
# =========================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# forecasting window
CONTEXT_LENGTH = 96      # histori 96 menit
PREDICTION_LENGTH = 24   # prediksi 24 menit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {DEVICE}')

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / 'dataset' / 'TCN processed'

TRAIN_PATH = DATA_DIR / 'train.npy'
EVAL_PATH = DATA_DIR / 'eval.npy'
SCALER_PATH = DATA_DIR / 'scaler.pkl'
EVAL_TIME_PATH = DATA_DIR / 'eval_timestamps.csv'

ARTIFACT_DIR = BASE_DIR / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

train_data = np.load(TRAIN_PATH)
eval_data = np.load(EVAL_PATH)

scaler = joblib.load(SCALER_PATH)

eval_time = pd.read_csv(EVAL_TIME_PATH)

print(f'Train shape: {train_data.shape}')
print(f'Eval shape : {eval_data.shape}')

# =========================================================
# DATASET
# =========================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.data = data.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[
            idx + self.context_length :
            idx + self.context_length + self.prediction_length
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

# =========================================================
# CREATE DATASET
# =========================================================

train_dataset = TimeSeriesDataset(
    train_data,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH
)

eval_dataset = TimeSeriesDataset(
    eval_data,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH
)

print(f'Train samples: {len(train_dataset)}')
print(f'Eval samples : {len(eval_dataset)}')

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# TCN MODEL
# =========================================================

class TCNForecaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.tcn = TCN(
            num_inputs=1,
            num_channels=[32, 32, 64],
            kernel_size=3,
            dropout=0.2,
            causal=True
        )

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, PREDICTION_LENGTH)
        )

    def forward(self, x):
        # x: [B, L, 1]

        # TCN expects [B, C, L]
        x = x.permute(0, 2, 1)  # [B, 1, L]

        # TCN output: [B, 64, L]
        features = self.tcn(x)

        # ambil representasi waktu terakhir
        last = features[:, :, -1]  # [B, 64]

        # forecast horizon
        out = self.head(last)  # [B, 24]

        return out.unsqueeze(-1)  # [B, 24, 1]

model = TCNForecaster().to(DEVICE)

print(model)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

# =========================================================
# TRAINING
# =========================================================

train_losses = []

print('\\nStarting TCN training...')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(
        f'Epoch [{epoch+1}/{EPOCHS}] '
        f'Loss: {avg_loss:.6f}'
    )

# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = ARTIFACT_DIR / 'tcn_model.pth'

torch.save(model.state_dict(), MODEL_PATH)

print(f'\\nModel saved: {MODEL_PATH}')

# =========================================================
# EVALUATION
# =========================================================

model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for x, y in eval_loader:
        x = x.to(DEVICE)

        pred = model(x)

        predictions.append(pred.cpu().numpy())
        actuals.append(y.numpy())

# concatenate
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

print(f'Predictions shape: {predictions.shape}')
print(f'Actuals shape    : {actuals.shape}')

# =========================================================
# FLATTEN
# =========================================================

y_pred = predictions.reshape(-1, 1)
y_true = actuals.reshape(-1, 1)

# =========================================================
# INVERSE SCALING
# =========================================================

y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y_true)

print('ACTUAL')
print('min :', y_true_inv.min())
print('max :', y_true_inv.max())
print('mean:', y_true_inv.mean())

print('\\nPREDICTION')
print('min :', y_pred_inv.min())
print('max :', y_pred_inv.max())
print('mean:', y_pred_inv.mean())

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_true_inv, y_pred_inv)
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)

# SMAPE lebih stabil
smape = np.mean(
    2 * np.abs(y_pred_inv - y_true_inv) /
    (np.abs(y_true_inv) + np.abs(y_pred_inv) + 1e-8)
) * 100

r2 = r2_score(y_true_inv, y_pred_inv)

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'SMAPE (%)', 'R-Squared'],
    'Value': [mae, mse, rmse, smape, r2]
})

# =========================================================
# SAVE RESULTS
# =========================================================

results.to_csv(ARTIFACT_DIR / 'metrics.csv', index=False)

# flatten
actual_flat = y_true_inv.flatten()
pred_flat = y_pred_inv.flatten()

# jumlah sample hasil forecasting
n_samples = len(actual_flat)

# buat time index sintetis untuk hasil forecast
pred_df = pd.DataFrame({
    'step': np.arange(n_samples),
    'actual': actual_flat,
    'predicted': pred_flat
})

pred_df.to_csv(ARTIFACT_DIR / 'predictions.csv', index=False)

np.save(ARTIFACT_DIR / 'predictions.npy', y_pred_inv)
np.save(ARTIFACT_DIR / 'actuals.npy', y_true_inv)

print('\\nSaved:')
print('- metrics.csv')
print('- predictions.csv')
print('- predictions.npy')
print('- actuals.npy')

# =========================================================
# PLOT PREDICTION
# =========================================================

plt.figure(figsize=(14, 5))

n_plot = min(500, len(pred_df))

plt.plot(
    pred_df['actual'].iloc[:n_plot],
    label='Actual',
    linewidth=1.5
)

plt.plot(
    pred_df['predicted'].iloc[:n_plot],
    label='TCN Prediction',
    linewidth=1.5
)

plt.title('TCN: Actual vs Predicted Pressure (July 2026)')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'prediction_plot.png', dpi=300)
plt.show()

# =========================================================
# PLOT TRAINING LOSS
# =========================================================

plt.figure(figsize=(8, 4))
plt.plot(train_losses)
plt.title('TCN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'training_loss.png', dpi=300)
plt.show()

print('\\nTraining completed successfully!')