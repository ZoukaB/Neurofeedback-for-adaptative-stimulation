import numpy as np
import time
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet, local_clock
from psychopy import sound
from scipy.signal import welch, butter, lfilter

# ---------- Setup ----------
print("🔍 Resolving EEG stream...")
eeg_streams = [s for s in resolve_streams() if s.name() == "UnicornRecorderLSLStream"]
if not eeg_streams:
    raise RuntimeError("❌ No EEG stream found.")
print(f"✅ Connected to: {eeg_streams[0].name()}")
inlet = StreamInlet(eeg_streams[0])

# Marker outlet (we emit our own markers)
marker_info = StreamInfo(name='BetaMarkers', type='Markers', channel_count=1,
                         channel_format='string', source_id='beta_marker_001')
marker_outlet = StreamOutlet(marker_info)
def send_marker(val: int) -> float:
    ts = local_clock()
    marker_outlet.push_sample([str(val)], timestamp=ts)
    print(f"📍 Marker@{ts:.6f}: {val}")
    return ts

# ---------- Params ----------
fs = 250.0
ch_idx = 1                # 0-indexed channel to analyze
beta_band = (13.0, 30.0)
artifact_ptp_uv = 200.0   # reject if peak-to-peak after filtering exceeds this

movement_s = 4.0
rest_s     = 6.0
trials     = 50

beep = sound.Sound(value=440, secs=0.2)

# ---------- Filter ----------
def bandpass_butter(x, fs, low=0.5, high=40.0, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, x)

# ---------- Beta power helpers ----------
def beta_power_1s(seg_1s, fs):
    """Beta power from a 1 s segment using Welch with nperseg=1 s."""
    if len(seg_1s) < int(0.9*fs):
        return np.nan
    freqs, psd = welch(seg_1s, fs=fs, nperseg=int(fs), noverlap=0)
    m = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    return float(np.mean(psd[m])) if np.any(m) else np.nan

def mean_beta_nonoverlap_1s(x, fs):
    """Mean beta over non-overlapping 1 s windows fully inside x."""
    n = int(fs)
    if len(x) < n:
        return np.nan
    # use only full seconds
    full = (len(x) // n) * n
    x = x[:full]
    betas = []
    for i in range(0, full, n):
        b = beta_power_1s(x[i:i+n], fs)
        if np.isfinite(b):
            betas.append(b)
    return float(np.mean(betas)) if betas else np.nan

# ---------- Small rolling buffer ----------
# We’ll keep the last ~12 s of raw samples for one channel + timestamps.
MAX_KEEP_S = 12.0
ts_buf = []
x_buf  = []

def trim(now_ts):
    cutoff = now_ts - MAX_KEEP_S
    # drop from start while older than cutoff
    i = 0
    while i < len(ts_buf) and ts_buf[i] < cutoff:
        i += 1
    if i > 0:
        del ts_buf[:i]
        del x_buf[:i]

def slice_by_time(t0, t1):
    """Return samples with ts in [t0, t1)."""
    # simple linear pass (buffers are short)
    out = [x for (t, x) in zip(ts_buf, x_buf) if t0 <= t < t1]
    return np.asarray(out, dtype=float)

def slice_last_seconds(end_ts, seconds):
    """Return up to exactly seconds*fs samples immediately before end_ts."""
    n_needed = int(seconds * fs)
    # First try time-based
    x = [x for (t, x) in zip(ts_buf, x_buf) if end_ts - seconds <= t <= end_ts]
    if len(x) >= n_needed:
        return np.asarray(x[-n_needed:], dtype=float)
    # Fallback: index-based from tail (in case of timestamp jitter)
    # Take samples whose timestamps are < end_ts, from the end backwards.
    xs = []
    for t, x in zip(reversed(ts_buf), reversed(x_buf)):
        if t < end_ts:
            xs.append(x)
            if len(xs) == n_needed:
                break
    return np.asarray(list(reversed(xs)), dtype=float)

# ---------- Main loop (REST → MOVEMENT) ----------
print("🚀 Starting trials...")
for trial in range(1, trials + 1):

    # ----- REST (pre-movement baseline period) -----
    beep.play()
    ts = local_clock()
    rest_ts = send_marker(0)
    t_start = time.time()
    # collect during REST
    while time.time() - t_start < rest_s:
        s, t = inlet.pull_sample(timeout=0.1)
        if s is None:
            continue
        ts_buf.append(t if t is not None else local_clock())
        x_buf.append(s[ch_idx])
        trim(ts_buf[-1])

    # ----- MOVEMENT (task) -----
    beep.play()
    onset_ts = send_marker(1)  # MOVEMENT starts
    t_start = time.time()
    # collect during MOVEMENT
    while time.time() - t_start < movement_s:
        s, t = inlet.pull_sample(timeout=0.1)
        if s is None:
            continue
        ts_buf.append(t if t is not None else local_clock())
        x_buf.append(s[ch_idx])
        trim(ts_buf[-1])
    
    # Mark end of movement and the start of the next REST window
    move_end_ts = local_clock()  # End movement timestamp

    # ---------- Metrics ----------
    # Baseline: exactly the 1 s immediately before movement onset
    # (fix: correct argument order for slice_last_seconds)
    base_raw = slice_last_seconds(onset_ts, 1.0)
    base_flt = bandpass_butter(base_raw, fs) if base_raw.size else base_raw
    if base_flt.size == 0 or np.ptp(base_flt) > artifact_ptp_uv:
        baseline_beta = np.nan
    else:
        baseline_beta = beta_power_1s(base_flt, fs)

    # Task: only data fully inside [onset_ts, move_end_ts)
    task_raw = slice_by_time(onset_ts, move_end_ts)
    task_flt = bandpass_butter(task_raw, fs) if task_raw.size else task_raw
    if task_flt.size == 0 or np.ptp(task_flt) > artifact_ptp_uv:
        task_beta = np.nan
    else:
        task_beta = mean_beta_nonoverlap_1s(task_flt, fs)

    # ERD%
    if np.isfinite(baseline_beta) and np.isfinite(task_beta) and baseline_beta != 0:
        erd = ((task_beta - baseline_beta) / baseline_beta) * 100.0
        print(f"🏁 Trial {trial:02d} | Baselineβ {baseline_beta:.4f} | Taskβ {task_beta:.4f} | ERD% {erd:.2f}")
    else:
        print(f"🏁 Trial {trial:02d} | ERD% NaN "
              f"(baselineβ={baseline_beta}, taskβ={task_beta}, n_task={task_flt.size})")

print("✅ Done.")
