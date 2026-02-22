"""
Phase 2: Verify existing training results.
Re-runs the train.py pipeline from scratch and compares reproduced metrics to saved metrics.json.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("PHASE 2 — VERIFY EXISTING RESULTS")
print("=" * 70)

# ── Load saved metrics ────────────────────────────────────────────────────────
with open("models/metrics.json") as f:
    saved_metrics = json.load(f)

print("\nSaved metrics (from models/metrics.json):")
for model_name in ["logistic_regression", "lstm"]:
    m = saved_metrics[model_name]
    print(f"  {model_name}:")
    print(f"    test_auc   = {m['test_auc']:.6f}")
    print(f"    test_brier = {m['test_brier']:.6f}")
    print(f"    val_auc    = {m['val_auc']:.6f}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading data/processed/dataset.parquet...")
df = pd.read_parquet("data/processed/dataset.parquet")
print(f"  Raw shape: {df.shape}")

# Exclude future matches (same as what train.py would have seen)
today = pd.Timestamp("2026-02-22")
future_mask = df["Date"] > today
if future_mask.sum() > 0:
    print(f"  NOTE: Excluding {future_mask.sum()} future-date rows (> {today.date()})")
    df = df[~future_mask].copy()

print(f"  Shape after future-date filter: {df.shape}")

feature_cols = [
    "home_gf_r5", "home_ga_r5", "home_gd_r5",
    "away_gf_r5", "away_ga_r5", "away_gd_r5",
    "log_home_win_odds", "log_draw_odds", "log_away_win_odds",
    "home_days_since_last", "away_days_since_last", "month"
]
target_col = "y_ht_draw"

df_clean = df.dropna(subset=feature_cols + [target_col]).sort_values("Date").reset_index(drop=True)
print(f"  After dropping NaN rows: {len(df_clean):,}")
print(f"  Draw rate: {df_clean[target_col].mean():.2%}")
print(f"  Date range: {df_clean['Date'].min().date()} → {df_clean['Date'].max().date()}")

# ── Temporal split ────────────────────────────────────────────────────────────
n = len(df_clean)
train_end = int(0.70 * n)
val_end   = train_end + int(0.15 * n)

train_df = df_clean.iloc[:train_end]
val_df   = df_clean.iloc[train_end:val_end]
test_df  = df_clean.iloc[val_end:]

print(f"\n  Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
print(f"  Train dates: {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"  Val dates:   {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")
print(f"  Test dates:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")

X_train = train_df[feature_cols].values; y_train = train_df[target_col].values
X_val   = val_df[feature_cols].values;   y_val   = val_df[target_col].values
X_test  = test_df[feature_cols].values;  y_test  = test_df[target_col].values

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── Logistic Regression ───────────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
lr.fit(X_train_s, y_train)

repro_lr = {
    "train_auc":   roc_auc_score(y_train, lr.predict_proba(X_train_s)[:, 1]),
    "val_auc":     roc_auc_score(y_val,   lr.predict_proba(X_val_s)[:, 1]),
    "test_auc":    roc_auc_score(y_test,  lr.predict_proba(X_test_s)[:, 1]),
    "train_brier": brier_score_loss(y_train, lr.predict_proba(X_train_s)[:, 1]),
    "val_brier":   brier_score_loss(y_val,   lr.predict_proba(X_val_s)[:, 1]),
    "test_brier":  brier_score_loss(y_test,  lr.predict_proba(X_test_s)[:, 1]),
}
print(f"  Reproduced  — test_auc={repro_lr['test_auc']:.6f}, test_brier={repro_lr['test_brier']:.6f}")
print(f"  Saved       — test_auc={saved_metrics['logistic_regression']['test_auc']:.6f}, test_brier={saved_metrics['logistic_regression']['test_brier']:.6f}")

# ── LSTM (same architecture as train.py) ─────────────────────────────────────
class HalfTimeDrawLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        return self.sigmoid(self.fc2(out)).squeeze()

print("\nTraining LSTM (seq_len=1, same as original pipeline)...")
device = torch.device("cpu")
input_size = len(feature_cols)
model = HalfTimeDrawLSTM(input_size).to(device)

# Reshape to (N, 1, F) — seq_len=1
Xtr = torch.FloatTensor(X_train_s.reshape(-1, 1, input_size))
ytr = torch.FloatTensor(y_train)
Xv  = torch.FloatTensor(X_val_s.reshape(-1, 1, input_size))
yv  = torch.FloatTensor(y_val)
Xte = torch.FloatTensor(X_test_s.reshape(-1, 1, input_size))

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xv, yv),   batch_size=64)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_auc = 0
best_state = None
patience_counter = 0
patience = 10

for epoch in range(50):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        criterion(model(bx), by).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        vp = model(Xv).numpy()
    val_auc = roc_auc_score(y_val, vp)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    repro_lstm = {
        "train_auc":   roc_auc_score(y_train, model(Xtr).numpy()),
        "val_auc":     roc_auc_score(y_val,   model(Xv).numpy()),
        "test_auc":    roc_auc_score(y_test,  model(Xte).numpy()),
        "train_brier": brier_score_loss(y_train, model(Xtr).numpy()),
        "val_brier":   brier_score_loss(y_val,   model(Xv).numpy()),
        "test_brier":  brier_score_loss(y_test,  model(Xte).numpy()),
        "best_val_auc": best_val_auc,
    }

print(f"  Reproduced  — test_auc={repro_lstm['test_auc']:.6f}, test_brier={repro_lstm['test_brier']:.6f}")
print(f"  Saved       — test_auc={saved_metrics['lstm']['test_auc']:.6f}, test_brier={saved_metrics['lstm']['test_brier']:.6f}")

# ── Comparison Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: Saved vs Reproduced")
print("=" * 70)

tol = 0.005  # 0.5% tolerance for floating point / seed differences

discrepancies = []
for model_name, repro in [("logistic_regression", repro_lr), ("lstm", repro_lstm)]:
    saved = saved_metrics[model_name]
    print(f"\n{model_name}:")
    print(f"  {'Metric':<20} {'Saved':>10} {'Repro':>10} {'Delta':>10} {'Match?':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for metric in ["train_auc", "val_auc", "test_auc", "train_brier", "val_brier", "test_brier"]:
        s_val = saved.get(metric, float("nan"))
        r_val = repro.get(metric, float("nan"))
        delta = abs(s_val - r_val)
        match = "OK" if delta <= tol else "DIFF"
        if match == "DIFF":
            discrepancies.append(f"{model_name}.{metric}: saved={s_val:.6f} repro={r_val:.6f} Δ={delta:.6f}")
        print(f"  {metric:<20} {s_val:>10.6f} {r_val:>10.6f} {delta:>+10.6f} {match:>8}")

print("\n" + "=" * 70)
if discrepancies:
    print(f"DISCREPANCIES ({len(discrepancies)}):")
    for d in discrepancies:
        print(f"  ⚠️  {d}")
    print("\nNote: LSTM discrepancies are expected due to stochastic training.")
    print("Logistic Regression should be fully deterministic — any LR discrepancy")
    print("indicates the original metrics were from a different dataset split.")
else:
    print("✅ All metrics match within tolerance (±0.005).")
print("=" * 70)

# Save reproduced results
repro_all = {
    "logistic_regression": repro_lr,
    "lstm": repro_lstm,
    "dataset_info": {
        "rows_total": int(len(df_clean)),
        "rows_train": int(len(train_df)),
        "rows_val":   int(len(val_df)),
        "rows_test":  int(len(test_df)),
        "future_rows_excluded": int(future_mask.sum()),
        "draw_rate": float(df_clean[target_col].mean()),
    },
    "discrepancies": discrepancies,
}
with open("models/metrics_reproduced.json", "w") as f:
    json.dump(repro_all, f, indent=2)
print("\n  Saved models/metrics_reproduced.json")
