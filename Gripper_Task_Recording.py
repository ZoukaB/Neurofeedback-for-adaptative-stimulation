import numpy as np
import time
import os
import csv
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet
from psychopy import sound, core
from scipy.signal import welch, butter, lfilter

# === Setup ===
print("🔍 Resolving EEG stream...")
streams = resolve_streams()
eeg_streams = [s for s in streams if s.name() == "UnicornRecorderLSLStream"]

if not eeg_streams:
    raise RuntimeError("❌ No EEG stream found.")

print(f"Connected to: {eeg_streams[0].name()}")
inlet = StreamInlet(eeg_streams[0])

# === LSL Marker Stream ===
marker_info = StreamInfo(name='BetaMarkers', type='Markers', channel_count=1,
                         channel_format='string', source_id='beta_marker_001')
marker_outlet = StreamOutlet(marker_info)

def send_marker(marker):
    marker_outlet.push_sample([str(marker)])
    print(f"📍 Marker: {marker}")

# === Parameters ===
original_sample_rate = 250
downsample_factor = 1
sample_rate = original_sample_rate / downsample_factor

window_size = 2.0   # 2.0 s window
step_size = 0.025   # 0.025 s (25 ms) step
buffer_size = int(sample_rate * window_size)
step_samples = int(sample_rate * step_size)

selected_channels = [1]  # channel Ch2 (0-indexed)
reject_threshold = 200   # µV
beta_band = (13, 30)

beep = sound.Sound(value=440, secs=0.5)

# === Logging Setup ===
save_dir = "beta_power_task_30_trials_no_grip"
os.makedirs(save_dir, exist_ok=True)
task_beta_file = os.path.join(save_dir, "task_beta.csv")
task_eeg_file = os.path.join(save_dir, "task_eeg.csv")

task_beta_log = open(task_beta_file, 'w', newline='')
task_eeg_log = open(task_eeg_file, 'w', newline='')

task_beta_writer = csv.writer(task_beta_log)
task_eeg_writer = csv.writer(task_eeg_log)

task_beta_writer.writerow(["timestamp", "beta_power", "state"])
task_eeg_writer.writerow(["timestamp"] + [f"Ch{ch+1}" for ch in range(8)] + ["state"])

# === Filter ===
def create_bandpass_filter(fs, lowcut=0.5, highcut=40.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

b_bp, a_bp = create_bandpass_filter(sample_rate)

def apply_bandpass_filter(data):
    return lfilter(b_bp, a_bp, data)

def compute_beta_power(psd, freqs, band):
    beta_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[beta_mask])

# === Task Phase (REST → MOVEMENT) ===
print("Starting task phase")
movement_duration = 4
rest_duration = 6
cycle_duration = movement_duration + rest_duration
num_cycles = 50

eeg_buffer = {ch: [] for ch in selected_channels}
time_buffer = []

for i in range(num_cycles):
    for state, duration, marker in [("REST", rest_duration, 0), ("MOVEMENT", movement_duration, 1)]:
        send_marker(marker)
        beep.play()
        block_start = time.time()
        while time.time() - block_start < duration:
            sample, timestamp = inlet.pull_sample(timeout=0.5)
            if sample:
                for ch in selected_channels:
                    eeg_buffer[ch].append(sample[ch])
                time_buffer.append(timestamp)
                task_eeg_writer.writerow([timestamp] + [sample[ch] for ch in range(8)] + [marker])

            if len(time_buffer) >= buffer_size:
                beta_powers = []
                for ch in selected_channels:
                    data_window = np.array(eeg_buffer[ch][-buffer_size:])
                    filtered = apply_bandpass_filter(data_window)
                    if np.ptp(filtered) > reject_threshold:
                        continue
                    freqs, psd = welch(filtered, fs=sample_rate, nperseg=buffer_size)
                    beta = compute_beta_power(psd, freqs, beta_band)
                    beta_powers.append(beta)
                if beta_powers:
                    avg_beta = np.mean(beta_powers)
                    task_beta_writer.writerow([time_buffer[buffer_size-1], avg_beta, marker])
                for ch in selected_channels:
                    eeg_buffer[ch] = eeg_buffer[ch][step_samples:]
                time_buffer = time_buffer[step_samples:]

# === Cleanup ===
task_beta_log.close()
task_eeg_log.close()
print("✅ Task complete. All data saved.")
