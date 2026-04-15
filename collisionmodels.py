# ==========================================================
# MODEL COMPARISON TABLE GENERATION
# ==========================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

# ==========================================================
# METRICS FUNCTION
# ==========================================================
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ==========================================================
# DUMMY DATA (Replace with your real dataset)
# ==========================================================
np.random.seed(42)
X = np.random.rand(500, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.05, 500)

# Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================================
# LINEAR REGRESSION
# ==========================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr, mae_lr, r2_lr = compute_metrics(y_test, y_pred_lr)

# ==========================================================
# MOVING AVERAGE
# ==========================================================
def moving_average(series, window=3):
    return np.convolve(series, np.ones(window)/window, mode='same')

y_pred_ma = moving_average(y_test)
rmse_ma, mae_ma, r2_ma = compute_metrics(y_test, y_pred_ma)

# ==========================================================
# SIMPLE RNN
# ==========================================================
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# ==========================================================
# GRU
# ==========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ==========================================================
# TCN (Simplified)
# ==========================================================
class TCN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.relu(self.conv1(x))
        out = out.mean(dim=2)
        return self.fc(out)

# ==========================================================
# TRAIN + EVALUATE DL MODELS
# ==========================================================
def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = loss_fn(output, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        preds = model(X_test_t).squeeze().numpy()

    return compute_metrics(y_test, preds)

# ==========================================================
# RUN MULTIPLE TIMES (FOR STD)
# ==========================================================
def run_multiple(model_class, runs=5):
    rmse_list, mae_list, r2_list = [], [], []

    for _ in range(runs):
        model = model_class(input_size=5)
        rmse, mae, r2 = train_and_eval(model, X_train, y_train, X_test, y_test)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    return np.mean(rmse_list), np.std(rmse_list), np.mean(mae_list), np.mean(r2_list)

# ==========================================================
# RESULTS
# ==========================================================
gru_rmse, gru_std, gru_mae, gru_r2 = run_multiple(GRUModel)
rnn_rmse, rnn_std, rnn_mae, rnn_r2 = run_multiple(RNNModel)
tcn_rmse, tcn_std, tcn_mae, tcn_r2 = run_multiple(TCN)

# ==========================================================
# CREATE TABLE
# ==========================================================
data = [
    ["GRU", f"{gru_rmse:.4f}", f"±{gru_std:.4f}", f"{gru_mae:.4f}", f"{gru_r2:.3f}"],
    ["Linear Regression", f"{rmse_lr:.4f}", "—", f"{mae_lr:.4f}", f"{r2_lr:.3f}"],
    ["Moving Average", f"{rmse_ma:.4f}", "—", f"{mae_ma:.4f}", f"{r2_ma:.3f}"],
    ["RNN", f"{rnn_rmse:.4f}", f"±{rnn_std:.4f}", f"{rnn_mae:.4f}", f"{rnn_r2:.3f}"],
    ["TCN", f"{tcn_rmse:.4f}", f"±{tcn_std:.4f}", f"{tcn_mae:.4f}", f"{tcn_r2:.3f}"],
]

df = pd.DataFrame(data, columns=["Model", "Mean RMSE", "Std", "MAE", "R²"])

print("\n===== MODEL COMPARISON TABLE =====")
print(df.to_string(index=False))

# Save as CSV (for paper)
df.to_csv("model_comparison.csv", index=False)
