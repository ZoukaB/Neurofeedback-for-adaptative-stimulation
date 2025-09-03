from pylsl import StreamInlet, resolve_streams
import csv

STREAM_NAME = "UnicornRecorderLSLStream"  # name of the EEG stream to look for

# --- Resolve the EEG stream ---
print(f"🔍 Looking for EEG stream '{STREAM_NAME}'...")
streams = resolve_streams()
eeg_streams = [s for s in streams if s.name() == STREAM_NAME]

if not eeg_streams:
    raise RuntimeError(f"❌ No EEG stream named '{STREAM_NAME}' found.")

info = eeg_streams[0]
print(f"Found stream: {info.name()} | type={info.type()} | ch={info.channel_count()} | fs={info.nominal_srate()} Hz")

# --- Try to get channel labels from metadata ---
def get_channel_labels(stream_info):
    labels = []
    try:
        chs = stream_info.desc().child("channels")
        ch = chs.child("channel")
        while ch.name():
            lbl = ch.child_value("label")
            labels.append(lbl if lbl else None)
            ch = ch.next_sibling()
    except Exception:
        pass
    if not labels or len(labels) != stream_info.channel_count():
        labels = [f"ch{i+1}" for i in range(stream_info.channel_count())]
    return labels

ch_labels = get_channel_labels(info)
print(f"📋 Channel labels: {ch_labels}")

# --- Create inlet and CSV ---
inlet = StreamInlet(info)
with open("eeg_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp"] + ch_labels)
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            writer.writerow([timestamp] + sample)
    except KeyboardInterrupt:
        print("\n✅ Recording stopped.")
