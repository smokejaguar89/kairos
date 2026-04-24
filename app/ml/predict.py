import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training)
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
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction.squeeze()


# ==========================================
# 2. REBUILD SCALERS & GET LATEST DATA
# ==========================================


def prep_data_and_scalers(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    if "Reading" in df.columns:
        df.drop(columns=["Reading"], inplace=True)

    df_clean = df.resample("15min").mean().interpolate(method="time")

    # --- Rebuild the 'Hours_Since_Watered' Feature using the new heuristic ---
    current_moisture = df_clean["Water moisture"]
    previous_moisture = current_moisture.shift(1)

    is_massive_jump = (current_moisture - previous_moisture) >= 0.20
    is_now_wet = current_moisture >= 0.60
    was_below_60 = previous_moisture < 0.60

    watering_timestamps = df_clean[
        is_massive_jump & is_now_wet & was_below_60
    ].index

    df_clean["Last_Watered_Time"] = np.nan
    df_clean.loc[watering_timestamps, "Last_Watered_Time"] = (
        watering_timestamps
    )
    df_clean["Last_Watered_Time"] = df_clean["Last_Watered_Time"].ffill()
    df_clean["Last_Watered_Time"] = df_clean["Last_Watered_Time"].fillna(
        df_clean.index[0]
    )

    df_clean["Hours_Since_Watered"] = (
        df_clean.index - df_clean["Last_Watered_Time"]
    ).dt.total_seconds() / 3600.0
    df_clean.drop(columns=["Last_Watered_Time"], inplace=True)

    # --- Rebuild targets purely to fit the scalers to match training ---
    df_for_scaler = df_clean.copy()
    dry_timestamps = df_for_scaler[
        df_for_scaler["Water moisture"] < 0.25
    ].index
    df_for_scaler["Hours_Until_Dry"] = np.nan

    for idx, row in df_for_scaler.iterrows():
        if row["Water moisture"] < 0.25:
            df_for_scaler.loc[idx, "Hours_Until_Dry"] = 0.0
            continue
        future_dry = dry_timestamps[dry_timestamps > idx]
        future_waters = watering_timestamps[watering_timestamps > idx]
        if len(future_dry) > 0:
            next_dry = future_dry[0]
            if len(future_waters) > 0 and future_waters[0] < next_dry:
                pass
            else:
                df_for_scaler.loc[idx, "Hours_Until_Dry"] = (
                    next_dry - idx
                ).total_seconds() / 3600.0

    df_for_scaler.dropna(subset=["Hours_Until_Dry"], inplace=True)

    features_df = df_for_scaler.drop(columns=["Hours_Until_Dry"])
    targets_series = df_for_scaler["Hours_Until_Dry"].values

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(features_df)

    target_scaler = MinMaxScaler()
    target_scaler.fit(targets_series.reshape(-1, 1))

    return feature_scaler, target_scaler, df_clean


# ==========================================
# 3. MAKE THE PREDICTION
# ==========================================
if __name__ == "__main__":
    print("Loading latest sensor data...")
    feature_scaler, target_scaler, full_df = prep_data_and_scalers(
        "sensor_data.csv"
    )

    # Grab the last 24 hours (96 intervals of 15 mins)
    last_24_hours = full_df.tail(96)

    current_moisture = last_24_hours["Water moisture"].iloc[-1]
    hours_since_water = last_24_hours["Hours_Since_Watered"].iloc[-1]

    if current_moisture < 0.25:
        print(
            "\n🌱 STATUS: The plant is ALREADY below 25% moisture. Water it now! 🌱"
        )
        exit()

    # Scale and prepare tensor
    scaled_recent_data = feature_scaler.transform(last_24_hours)
    input_tensor = torch.tensor(
        scaled_recent_data, dtype=torch.float32
    ).unsqueeze(0)

    # Load Model
    num_features = len(full_df.columns)
    model = TimeToEventLSTM(
        input_size=num_features, hidden_size=32, num_layers=1, output_size=1
    )

    try:
        model.load_state_dict(torch.load("time_to_dry_weights.pth"))
        model.eval()
    except FileNotFoundError:
        print(
            "Error: Could not find 'time_to_dry_weights.pth'. Did you run train.py?"
        )
        exit()

    # Predict
    with torch.no_grad():
        scaled_prediction = model(input_tensor)

    scaled_prediction_array = np.array([[scaled_prediction.item()]])
    real_hours = target_scaler.inverse_transform(scaled_prediction_array)[0][0]

    days = real_hours / 24.0

    print("\n" + "=" * 50)
    print("🌱 PLANT DRYING PREDICTION 🌱")
    print("=" * 50)
    print(f"Current Moisture:      {current_moisture * 100:.1f}%")
    print(f"Time Since Watered:    {hours_since_water:.1f} Hours")
    print("-" * 50)
    print(f"The LSTM predicts the soil will drop below 25% in:")
    print(f"\n>> {real_hours:.1f} Hours ({days:.1f} Days) <<")
    print("=" * 50)
