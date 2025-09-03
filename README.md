# Neurofeedback-for-adaptative-stimulation
This repository recenses my work as a research intern over the summer 2025 in a neuroscience laboratory focusing on brain reorganization due to age and stroke at McGill University. 
The goal of the internship was to work on the design, implementation, and evaluation of a prototype for a closed-loop neurofeedback system aiming to detect EEG biomarkers in real time in order to trigger non-invasive brain stimulation (tES/tACS). 
The main objective was to demonstrate the feasibility of a closed-loop system modulated by spectral EEG features, starting with alpha power and extending to movement-related beta desynchronization (MRBD).

A detailed report on my work is available in this repository.
I included a notebook briefly explaining how to work with LSL streams in Python. I also included an annotated notebook of my MRBD_live_detection script 

## 1. Test and Helper Scripts

This document briefly explains the utility of each Python script present in the repo.
These scripts are designed to make testing easier.
### **unicorn_test.py**
●	Minimal LSL logger to verify Unicorn streaming.

●	Resolves the EEG stream, pulls samples continuously, infers channel labels from stream metadata (when possible).

●	Writes a timestamped CSV (eeg_log.csv) until interrupted.

### **EEG_replay_script.py**
●	Replays a saved multichannel EEG CSV in real time over LSL.

●	Useful for testing real-time codes without reacquiring data.

●	Creates an LSL outlet named SimulatedEEGStream at 250 Hz (modifiable).

●	Pushes each row as a sample with the original channel count, effectively simulating a live device for downstream pipelines.

## 2. Data Acquisition Script
Recording script for offline analysis and parameter testing (window size, overlapping, step size).

### **Gripper_Task_Recording.py**
●	Runs an alternating REST → MOVEMENT task while recording from UnicornRecorderLSLStream.

●	Beeps on state changes, sends LSL markers (0 = rest, 1 = movement).

●	Band-pass filters (0.5–40 Hz).

●	Computes beta power (13–30 Hz) in sliding 2 s windows every 25 ms with artifact rejection.

●	Logs both raw EEG (8 channels) and per-window beta power to CSVs.

## 3. Offline Analysis Scripts
### **MRBD_extraction_BrainVision.py**

●	Offline MRBD analysis for BrainVision recordings.

●	Loads a .vhdr, applies notch/band-pass filtering, and optional resampling (to 250 Hz).

●	Epochs around chosen events (e.g., Stimulus/S 99).

●	Computes beta-band power using Morlet TFRs for baseline (−1–0 s) vs task (0–4 s).

●	Reports ERD metrics and plots time–frequency maps and ERD time courses.

### **MRBD_extraction_CSV.py**

●	Offline MRBD analysis from CSV exported by the task recorder.

●	Reconstructs an MNE Raw object from the multichannel CSV.

●	Filters data, detects event onsets from state transitions (0 → 1).

●	Epochs (−1–8 s).

●	Computes beta power via Morlet TFRs for baseline vs task windows.

●	Outputs ERD% summaries and produces time–frequency/ERD plots.

## 4. Live Detection Script
This script was originally tested with Alpha power synchronization on the Unicorn headset.
It did not detect MRBD directly. The logic works but should be tested on the BrainVision System.
Additional filters need to be added. See MRBD_detection_live_annotated notebook for more details.

### **MRBD_detection_live.py**
●	Online MRBD/ERD demo for the Unicorn LSL stream with REST/MOVEMENT trials.

●	Maintains a rolling buffer, band-pass filters (0.5–40 Hz).

●	Estimates beta power via Welch method.

●	Derives baseline from 1 s pre-movement interval.

●	Computes task beta during movement.

●	Prints MRBD% per trial.

## Contact
If you have any questions regarding these scripts or the closed-loop neurofeedback system, feel free to contact me at:
 📧 zecabuclet@gmail.com
