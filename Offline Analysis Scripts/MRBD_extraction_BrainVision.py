import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.io import read_raw_brainvision
from mne.time_frequency import tfr_array_morlet

# ================== USER SETTINGS ==================
vhdr_path   = "C:/Users/zecab/Desktop/Stage Montreal/Data_EEG/OneDrive_1_17-06-2025/KM0404_S3_15min_Motor.vhdr"
event_key   = "Stimulus/S 99"
line_freq   = 60.0
resample_to = 250.0
C3_idx = 7

# Epoch timing (s)
tmin, tmax    = -1.0, 8.0
baseline_win  = (-1.0, 0.)
task_win      = (0.0, 4.0)

# Analysis settings
beta_band  = (13., 30.)                  # Hz
freqs      = np.linspace(1, 40, 80)      # Morlet freqs (Hz)
n_cycles   = freqs / 2.0                 # cycles per freq (adjust if you want 5–7 in beta)
reject     = dict(eeg=200e-6)            # 200 µV p2p rejection
flat       = dict(eeg=1e-6)              # flat epoch rejection
l_freq, h_freq = 1., 40.                 # band-pass (Hz)

# ---------- Load ----------
raw = read_raw_brainvision(vhdr_path, preload=True, verbose=False)
print(f"Loaded: {raw}, sfreq={raw.info['sfreq']} Hz, {raw.info['nchan']} channels")

# ---------- Preprocess ----------
raw.set_eeg_reference("average", projection=False)

nyq = raw.info["sfreq"] / 2.0
notch_freqs = np.arange(line_freq, nyq + 0.1, line_freq)
if len(notch_freqs) > 0:
    raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose=False)

raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", fir_design="firwin", verbose=False)

if raw.info["sfreq"] != resample_to:
    raw.resample(resample_to, npad="auto", verbose=False)
print(f"After preprocessing: sfreq={raw.info['sfreq']} Hz")

# ---------- Events / Epochs ----------
events, event_id = mne.events_from_annotations(raw)
print("Available markers:", list(event_id.keys()))
if event_key not in event_id:
    raise ValueError(f"Event '{event_key}' not found. Choose from {list(event_id.keys())}")

epochs = mne.Epochs(
    raw, events, event_id=event_id[event_key],
    tmin=tmin, tmax=tmax, picks="eeg",
    baseline=None, preload=True, reject=reject, flat=flat, verbose=False
)
print(f"Epochs kept (after reject): {len(epochs)}")
if len(epochs) == 0:
    raise RuntimeError("No epochs left after rejection — loosen thresholds or check markers.")

# ---------- Choose C3 (fallback to 8th EEG channel if missing) ----------
if "C3" in epochs.ch_names:
    ch_name = "C3"
else:
    ch_name = epochs.copy().pick("eeg").ch_names[C3_idx]
print(f"Using channel: {ch_name}")

# ---------- Windows ----------
epochs_base = epochs.copy().crop(*baseline_win)
epochs_task = epochs.copy().crop(*task_win)

# Arrays: (n_epochs, 1, n_times)
base_data = epochs_base.get_data(picks=[ch_name])
task_data = epochs_task.get_data(picks=[ch_name])

# ---------- Helpers ----------
def mean_power_over_time(data, freqs, n_cycles, sfreq):
    """
    Returns mean spectrum across time and epochs: shape (n_freqs,)
    """
    power = tfr_array_morlet(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power"
    )  # (n_epochs, 1, n_freqs, n_times)
    return power.mean(axis=-1).mean(axis=0).squeeze()

sfreq = epochs.info["sfreq"]

# ---------- Beta ERD (values in beta band only) ----------
mean_base_spec = mean_power_over_time(base_data, freqs, n_cycles, sfreq)
mean_task_spec = mean_power_over_time(task_data, freqs, n_cycles, sfreq)

beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
beta_base = mean_base_spec[beta_mask].mean()
beta_task = mean_task_spec[beta_mask].mean()

abs_change = beta_task - beta_base
rel_change_pct = 100.0 * abs_change / beta_base
db_change = 10.0 * np.log10(beta_task / beta_base)

print("\n=== Movement-related Beta (C3) ===")
print(f"Baseline beta power: {beta_base:.6e} (arb. units)")
print(f"Task beta power:     {beta_task:.6e} (arb. units)")
print(f"Absolute change:     {abs_change:.6e}")
print(f"Relative change:     {rel_change_pct:.2f}%  <-- ERD if negative")
print(f"dB change:           {db_change:.2f} dB")

# ---------- Time–frequency + ERD overlay for C3 ----------
full_data  = epochs.get_data(picks=[ch_name])  # (n_epochs, 1, n_times)
full_power = tfr_array_morlet(full_data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power")
avg_tf     = full_power.mean(axis=0).squeeze()  # (n_freqs, n_times)
times      = epochs.times

# Time-resolved beta power (avg across beta freqs)
beta_power_time = avg_tf[beta_mask, :].mean(axis=0)

# Baseline mean for ERD (%)
baseline_mask = (times >= baseline_win[0]) & (times <= baseline_win[1])
baseline_mean_beta = beta_power_time[baseline_mask].mean()
erd_timecourse = 100.0 * (beta_power_time - baseline_mean_beta) / baseline_mean_beta  # ERD%

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))
tf = ax1.contourf(times, freqs, 10*np.log10(avg_tf), levels=40, cmap="RdBu_r")
ax1.axvline(0, color="k", linestyle="--", label="Task start")
ax1.axvline(4, color="blue", linestyle="--", label="Task end")
ax1.axhline(beta_band[0], color="y", linestyle="--", alpha=0.8)
ax1.axhline(beta_band[1], color="y", linestyle="--", alpha=0.8)
cbar = plt.colorbar(tf, ax=ax1, label="Power (dB)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")

# Legends
ax1.legend(loc="upper right")
plt.title(f"{ch_name}: Time–Frequency")
plt.tight_layout()
plt.show()
# --- Beta-band mask and slices ---
beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
freqs_beta = freqs[beta_mask]
tf_beta = avg_tf[beta_mask, :]  # shape: (n_beta_freqs, n_times)

# Optional: convert to dB for the TF map
tf_beta_db = 10 * np.log10(tf_beta + np.finfo(float).eps)

# --- Compute time-resolved beta power and ERD% ---
beta_power_time = tf_beta.mean(axis=0)  # average across beta freqs -> (n_times,)
baseline_mask = (times >= baseline_win[0]) & (times <= baseline_win[1])
baseline_mean_beta = beta_power_time[baseline_mask].mean()
erd_timecourse = 100.0 * (beta_power_time - baseline_mean_beta) / baseline_mean_beta

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

# --- Figure 2: Beta ERD (%) over time ---
#for k in range(erd_per_trial.shape[0]):
plt.figure(figsize=(10, 3.5))
plt.plot(times, erd_timecourse, linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.axvline(task_win[0], color="k", linestyle="--", label="Task start")
plt.axvline(task_win[1], color="blue", linestyle="--", label="Task end")
# (Optional) shade the baseline window
plt.axvspan(baseline_win[0], baseline_win[1], color="k", alpha=0.08, label="Baseline")
plt.xlabel("Time (s)")
plt.ylabel("Beta ERD (%)")
plt.title(f"Average all trials: Beta Desynchronization (ERD) over C3")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
