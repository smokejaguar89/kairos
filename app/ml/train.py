import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path

# ==========================================
# 1. DATA ENGINEERING (With Hardware Heuristic)
# ==========================================


def load_and_engineer_data(filepath):
    print(
        "Loading data, applying hardware heuristics, and engineering targets..."
    )
    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    if "Reading" in df.columns:
        df.drop(columns=["Reading"], inplace=True)

    df_clean = df.resample("15min").mean().interpolate(method="time")

    # --- FIX 1: The 'Hours_Since_Watered' Feature ---
    current_moisture = df_clean["Water moisture"]
    previous_moisture = current_moisture.shift(1)

    # Custom Hardware Heuristic: Jump > 20pp AND crosses the 60% threshold
    is_massive_jump = (current_moisture - previous_moisture) >= 0.20
    is_now_wet = current_moisture >= 0.60
    was_below_60 = previous_moisture < 0.60

    watering_timestamps = df_clean[
        is_massive_jump & is_now_wet & was_below_60
    ].index

    # Create a column that tracks the timestamp of the LAST watering
    df_clean["Last_Watered_Time"] = np.nan
    df_clean.loc[watering_timestamps, "Last_Watered_Time"] = (
        watering_timestamps
    )
    df_clean["Last_Watered_Time"] = df_clean["Last_Watered_Time"].ffill()
    df_clean["Last_Watered_Time"] = df_clean["Last_Watered_Time"].fillna(
        df_clean.index[0]
    )

    # Calculate hours since that last watering timestamp
    df_clean["Hours_Since_Watered"] = (
        df_clean.index - df_clean["Last_Watered_Time"]
    ).dt.total_seconds() / 3600.0
    df_clean.drop(columns=["Last_Watered_Time"], inplace=True)

    # --- FIX 2: The Interruption-Proof Target (< 25% Moisture) ---
    dry_timestamps = df_clean[df_clean["Water moisture"] < 0.25].index
    df_clean["Hours_Until_Dry"] = np.nan

    for idx, row in df_clean.iterrows():
        # If it's already dry, time remaining is 0
        if row["Water moisture"] < 0.25:
            df_clean.loc[idx, "Hours_Until_Dry"] = 0.0
            continue

        future_dry_events = dry_timestamps[dry_timestamps > idx]
        future_waterings = watering_timestamps[watering_timestamps > idx]

        if len(future_dry_events) > 0:
            next_dry = future_dry_events[0]

            # Did a human water it BEFORE it naturally reached 25%?
            if len(future_waterings) > 0 and future_waterings[0] < next_dry:
                # CYCLE INTERRUPTED: Leave as NaN so it gets dropped.
                pass
            else:
                # PURE CYCLE: Safe to learn from
                df_clean.loc[idx, "Hours_Until_Dry"] = (
                    next_dry - idx
                ).total_seconds() / 3600.0

    # Drop rows that were interrupted or don't have a known future yet
    df_clean.dropna(subset=["Hours_Until_Dry"], inplace=True)

    return df_clean


# ==========================================
# 2. PYTORCH DATASET
# ==========================================


class TimeToDryDataset(Dataset):
    def __init__(self, features_array, target_array, sequence_length=96):
        self.sequence_length = sequence_length
        self.features = features_array
        self.targets = target_array

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x_window = self.features[idx: idx + self.sequence_length]
        y_target = self.targets[idx + self.sequence_length - 1]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(
            y_target, dtype=torch.float32
        )


# ==========================================
# 3. LSTM MODEL ARCHITECTURE
# ==========================================


class TimeToEventLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TimeToEventLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_out)
        return prediction.squeeze()


# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    df = load_and_engineer_data("sensor_data.csv")

    features_df = df.drop(columns=["Hours_Until_Dry"])
    targets_series = df["Hours_Until_Dry"].values

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features_df)

    target_scaler = MinMaxScaler()
    scaled_targets = target_scaler.fit_transform(targets_series.reshape(-1, 1))

    sequence_length = 96
    batch_size = 32

    dataset = TimeToDryDataset(
        scaled_features, scaled_targets, sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_sensor_features = len(features_df.columns)
    model = TimeToEventLSTM(
        input_size=num_sensor_features,
        hidden_size=32,
        num_layers=1,
        output_size=1,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    best_loss = float("inf")

    print(f"\nStarting training on {len(dataset)} pure sliding windows...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_y = batch_y.squeeze()
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # MODEL CHECKPOINTING
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = models_dir / "time_to_dry_weights.pth"
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            rmse_scaled = np.sqrt(avg_loss)
            rmse_hours = rmse_scaled * target_scaler.data_range_[0]
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Model is off by an average of +/- {rmse_hours:.2f} Hours"
            )

    print(
        f"\nTraining complete! The BEST version of the model was saved to '{models_dir / 'time_to_dry_weights.pth'}'."
    )

    # Save scalers for later use in predict.py
    with open(models_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(models_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print(
        f"Scalers saved to '{models_dir}'."
    )
