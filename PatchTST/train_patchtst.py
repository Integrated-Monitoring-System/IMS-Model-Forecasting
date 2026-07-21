import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# window
CONTEXT_LENGTH = 96      # 96 menit histori
PREDICTION_LENGTH = 24   # prediksi 24 menit ke depan

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {DEVICE}')

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR.parent / 'dataset' / 'PatchTST processed' / 'train.npy'
TEST_PATH = BASE_DIR.parent / 'dataset' / 'PatchTST processed' / 'test.npy'
SCALER_PATH = BASE_DIR / 'artifacts' / 'scaler.pkl'

MODEL_DIR = BASE_DIR / 'artifacts'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

scaler = joblib.load(SCALER_PATH)

print(f'Train shape: {train_data.shape}')
print(f'Test shape : {test_data.shape}')

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

test_dataset = TimeSeriesDataset(
    test_data,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH
)

# guard jika data terlalu sedikit
if len(train_dataset) <= 0:
    raise ValueError(
        f'Train dataset terlalu kecil. '
        f'Jumlah data: {len(train_data)}, '
        f'context: {CONTEXT_LENGTH}, '
        f'prediction: {PREDICTION_LENGTH}'
    )

if len(test_dataset) <= 0:
    raise ValueError(
        f'Test dataset terlalu kecil. '
        f'Jumlah data: {len(test_data)}, '
        f'context: {CONTEXT_LENGTH}, '
        f'prediction: {PREDICTION_LENGTH}'
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f'Train samples: {len(train_dataset)}')
print(f'Test samples : {len(test_dataset)}')

# =========================================================
# PATCHTST MODEL
# =========================================================

config = PatchTSTConfig(
    num_input_channels=1,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    patch_length=16,
    patch_stride=8,
    d_model=64,
    num_attention_heads=4,
    num_hidden_layers=3,
    ffn_dim=128,
    dropout=0.1
)

model = PatchTSTForPrediction(config).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

# =========================================================
# TRAINING LOOP
# =========================================================

train_losses = []

print('\\nStarting training...')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)  # [B, L, 1]
        y = y.to(DEVICE)  # [B, H, 1]

        optimizer.zero_grad()

        # x: [B, L, 1] -> [B, 1, L]
        outputs = model(
            past_values=x
        )
        # PatchTST output
        pred = outputs.prediction_outputs  # [B, H, 1]

        loss = criterion(pred, y)

        loss.backward()
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

MODEL_PATH = MODEL_DIR / 'patchtst_model.pth'

torch.save(model.state_dict(), MODEL_PATH)

print(f'\\nModel saved to: {MODEL_PATH}')

# =========================================================
# EVALUATION
# =========================================================

model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)

        # x: [B, L, 1] -> [B, 1, L]
        outputs = model(
            past_values=x
        )

        pred = outputs.prediction_outputs

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

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_true_inv, y_pred_inv)
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)

epsilon = 1e-8
mape = np.mean(
    np.abs((y_true_inv - y_pred_inv) / (y_true_inv + epsilon))
) * 100

r2 = r2_score(y_true_inv, y_pred_inv)

# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'R-Squared'],
    'Value': [mae, mse, rmse, mape, r2]
})

print('\\n' + '=' * 60)
print('PATCHTST EVALUATION RESULTS')
print('=' * 60)
print(results.to_string(index=False))

# =========================================================
# SAVE PREDICTIONS
# =========================================================

np.save(MODEL_DIR / 'predictions.npy', y_pred_inv)
np.save(MODEL_DIR / 'actuals.npy', y_true_inv)

results.to_csv(MODEL_DIR / 'metrics.csv', index=False)

print('\\nSaved:')
print('- predictions.npy')
print('- actuals.npy')
print('- metrics.csv')

# =========================================================
# PLOT PREDICTION
# =========================================================

plt.figure(figsize=(14, 5))

n_plot = min(500, len(y_true_inv))

plt.plot(
    y_true_inv[:n_plot],
    label='Actual',
    linewidth=1.5
)

plt.plot(
    y_pred_inv[:n_plot],
    label='PatchTST Prediction',
    linewidth=1.5
)

plt.title('PatchTST: Actual vs Predicted Pressure')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'prediction_plot.png', dpi=300)
plt.show()

# =========================================================
# PLOT TRAINING LOSS
# =========================================================

plt.figure(figsize=(8, 4))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'training_loss.png', dpi=300)
plt.show()

print('\\nTraining completed successfully!')