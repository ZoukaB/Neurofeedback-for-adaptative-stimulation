import pandas as pd
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

# ===== Parameters =====
sample_rate = 250  # Hz (use same as your live acquisition)
filename = "PSD_live_alpha/AlphaPower_250Hz_4_05_window_01/eyes_open.csv"  # EEG file to replay

# ===== Load EEG Data =====
df = pd.read_csv(filename)
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])
data = df.values  # shape: (n_samples, n_channels)
n_channels = data.shape[1]

print(f"✅ Loaded EEG data: {n_channels} channels, {len(data)} samples")

# ===== Create LSL Stream =====
info = StreamInfo(name="SimulatedEEGStream",
                  type="EEG",
                  channel_count=n_channels,
                  nominal_srate=sample_rate,
                  channel_format="float32",
                  source_id="sim_eeg_001")
outlet = StreamOutlet(info)

print(f"🚀 Streaming simulated EEG at {sample_rate} Hz... (Ctrl+C to stop)")

# ===== Replay Loop =====
try:
    for row in data:
        outlet.push_sample(row.tolist())
        time.sleep(1 / sample_rate)  # simulate real-time streaming
except KeyboardInterrupt:
    print("\n✅ Simulation stopped.")
