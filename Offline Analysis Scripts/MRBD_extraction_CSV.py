import sys
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet

# ================== USER SETTINGS ==================
csv_path    = "beta_power_task_50_trials_no_grip/task_eeg.csv"
event_state_transition = (0, 1)       # detect 0 -> 1 in 'state'
line_freq   = 60.0
sfreq = 250

# Epoch timing (s)
tmin, tmax    = -1.0, 8.0
baseline_win  = (-1.0, 0.)
task_win      = (0., 4.0)

# Analysis settings (match your original logic)
beta_band  = (15., 25.)                  # Hz
freqs      = np.linspace(1, 40, 80)      # Morlet freqs (Hz)
n_cycles   = freqs / 2.0                 # cycles per freq (≈ constant time res; ~0.5 s at 1 Hz down to 0.0125 s at 40 Hz)
reject     = dict(eeg=200e-6)            # 200 µV peak-to-peak (DATA MUST BE IN VOLTS)
flat       = dict(eeg=1e-6)              # flat epoch rejection
l_freq, h_freq = 1., 40.                 # band-pass (Hz)

# Data/unit handling
EEG_CHANNELS = [f"Ch{i}" for i in range(1, 9)]
USE_CHANNEL  = "Ch2"                     # channel of interest C3 electrode
SCALE_TO_VOLTS = 1e-6                    # if CSV values are in µV
SET_MONTAGE = None                       # e.g., "standard_1020" or None

# ================== LOAD CSV ==================
df = pd.read_csv(csv_path)
required_cols = {"timestamp", "state", *EEG_CHANNELS}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {sorted(missing)}")

ts = df["timestamp"].to_numpy(float)
if np.any(np.diff(ts) <= 0):
    raise ValueError("Timestamps must be strictly increasing in 'timestamp' column.")

# data array to volts, shape (n_channels, n_samples)
data = df[EEG_CHANNELS].to_numpy(float).T * SCALE_TO_VOLTS

# ================== BUILD RAW ==================
info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info, verbose=False)
if SET_MONTAGE:
    raw.set_montage(SET_MONTAGE, on_missing="ignore")

print(f"Loaded Raw: sfreq={raw.info['sfreq']} Hz, nchan={raw.info['nchan']}")

# ================== PREPROCESS ==================
# average reference (no projection, like your code)
raw.set_eeg_reference("average", projection=False)

# notch at line/harmonics
nyq = raw.info["sfreq"] / 2.0
notch_freqs = np.arange(line_freq, nyq + 0.1, line_freq)
if len(notch_freqs) > 0:
    raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose=False)

# band-pass
raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", fir_design="firwin", verbose=False)

# ================== EVENTS FROM state (0 -> 1) ==================
state = df["state"].to_numpy(int)
a, b = event_state_transition
on_idx = np.where((state[:-1] == a) & (state[1:] == b))[0]  # index of last 'a' sample
if len(on_idx) == 0:
    raise RuntimeError(f"No {a}→{b} transitions found in 'state' column.")

# Map absolute time to sample idx in Raw (Raw starts at t=0 at the first CSV sample)
t0_abs = ts[0]
def t_to_samp(t):
    return int(np.round((t - t0_abs) * raw.info["sfreq"]))

# Use the first 'b' sample (i+1) as the event time (time 0)
events = []
print(on_idx)
for i in on_idx:
    onset_time = ts[i + 1] # first sample where state==b
    events.append([i+1, 0, 1])
events = np.asarray(events, dtype=int)
event_id = {"task": 1}

# ================== EPOCHS ==================
epochs = mne.Epochs(
    raw, events, event_id=event_id["task"],
    tmin=tmin, tmax=tmax, picks="eeg",
    baseline=None, preload=True, reject=reject, flat=flat, verbose=False
)
print(f"Epochs kept (after reject): {len(epochs)}")
if len(epochs) == 0:
    raise RuntimeError("No epochs left after rejection. Check units (µV→V scaling) or relax reject/flat.")

# ================== PICK CHANNEL (like C3 logic -> here force Ch2) ==================
if USE_CHANNEL not in epochs.ch_names:
    raise ValueError(f"{USE_CHANNEL} not found in epochs. Available: {epochs.ch_names}")
ch_name = USE_CHANNEL
print(f"Using channel: {ch_name}")

# ================== WINDOWS ==================
epochs_base = epochs.copy().crop(*baseline_win)
epochs_task = epochs.copy().crop(*task_win)

# Arrays: (n_epochs, 1, n_times)
base_data = epochs_base.get_data(picks=[ch_name])
task_data = epochs_task.get_data(picks=[ch_name])

# ================== HELPERS ==================
def mean_power_over_time(data, freqs, n_cycles, sfreq):
    """
    Returns mean spectrum across time and epochs: shape (n_freqs,)
    """
    power = tfr_array_morlet(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power"
    )  # (n_epochs, 1, n_freqs, n_times)
    return power.mean(axis=-1).mean(axis=0).squeeze()

sfreq = epochs.info["sfreq"]

# ================== BETA ERD (same logic) ==================
mean_base_spec = mean_power_over_time(base_data, freqs, n_cycles, sfreq)
mean_task_spec = mean_power_over_time(task_data, freqs, n_cycles, sfreq)

beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
beta_base = mean_base_spec[beta_mask].mean()
beta_task = mean_task_spec[beta_mask].mean()

abs_change = beta_task - beta_base
rel_change_pct = 100.0 * abs_change / beta_base
db_change = 10.0 * np.log10((beta_task + np.finfo(float).eps) / (beta_base + np.finfo(float).eps))

print("\n=== Movement-related Beta (Ch2) ===")
print(f"Baseline beta power: {beta_base:.6e} (arb. units)")
print(f"Task beta power:     {beta_task:.6e} (arb. units)")
print(f"Absolute change:     {abs_change:.6e}")
print(f"Relative change:     {rel_change_pct:.2f}%  <-- ERD if negative")
print(f"dB change:           {db_change:.2f} dB")

# ================== Time–frequency + ERD overlay (same plotting style) ==================
full_data  = epochs.get_data(picks=[ch_name])  # (n_epochs, 1, n_times)
full_power = tfr_array_morlet(full_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power")
avg_tf     = full_power.mean(axis=0).squeeze()  # (n_freqs, n_times)
times      = epochs.times

# Time-resolved beta power (avg across beta freqs)
beta_power_time = avg_tf[beta_mask, :].mean(axis=0)

# Baseline mean for ERD (%)
baseline_mask = (times >= baseline_win[0]) & (times <= baseline_win[1])
baseline_mean_beta = beta_power_time[baseline_mask].mean()
erd_timecourse = 100.0 * (beta_power_time - baseline_mean_beta) / (baseline_mean_beta + np.finfo(float).eps)

# Figure 1: Time–Frequency map
fig, ax1 = plt.subplots(figsize=(10, 5))
tf = ax1.contourf(times, freqs, 10*np.log10(avg_tf + np.finfo(float).eps), levels=40, cmap="RdBu_r")
ax1.axvline(task_win[0], color="k", linestyle="--", label="Task start")
ax1.axvline(task_win[1], color="blue", linestyle="--", label="Task end")
ax1.axhline(beta_band[0], color="y", linestyle="--", alpha=0.8)
ax1.axhline(beta_band[1], color="y", linestyle="--", alpha=0.8)
cbar = plt.colorbar(tf, ax=ax1, label="Power (dB)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
ax1.legend(loc="upper right")
plt.title(f"{ch_name}: Time–Frequency")
plt.tight_layout()
plt.show()

# ================== Per‑trial ERD (%) for beta ==================
# full_data: (n_epochs, 1, n_times) already defined earlier
full_power = tfr_array_morlet(
    full_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power"
)  # (n_epochs, 1, n_freqs, n_times)

# Average across beta freqs, but NOT across trials
beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
# -> (n_epochs, n_times)
beta_power_time_per_trial = full_power[:, 0, beta_mask, :].mean(axis=1)

# Baseline per trial (use the same baseline window you set)
baseline_mask = (times >= baseline_win[0]) & (times <= baseline_win[1])
baseline_means = beta_power_time_per_trial[:, baseline_mask].mean(axis=1, keepdims=True)

# ERD% per trial: (n_epochs, n_times)
eps = np.finfo(float).eps
erd_per_trial = 100.0 * (beta_power_time_per_trial - baseline_means) / (baseline_means + eps)
k = 1

# Figure 2: Beta ERD (%) over time
#for k in range(erd_per_trial.shape[0]):
plt.figure(figsize=(10, 3.5))
plt.plot(times, erd_timecourse, linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.axvline(task_win[0], color="k", linestyle="--", label="Task start")
plt.axvline(task_win[1], color="blue", linestyle="--", label="Task end")
plt.axvspan(baseline_win[0], baseline_win[1], color="k", alpha=0.08, label="Baseline")
plt.xlabel("Time (s)")
plt.ylabel("Beta ERD (%)")
plt.title(f"Average all trials no grip: Beta Desynchronization (ERD) over C3") #plt.title(f"Trial:{k+1}: Beta Desynchronization (ERD) over C3") 
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
