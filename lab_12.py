#Write a MATLAB/Python program to separation of voiced/un voiced/silence regions 
#from a speech signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- 1. Load Audio Signal ---
fs, signal = wavfile.read('simple.wav')

# Convert stereo to mono if needed
if len(signal.shape) > 1:
    signal = signal[:, 0]

# Normalize signal between -1 and 1
signal = signal / np.max(np.abs(signal))

# --- 2. Framing Parameters ---
frame_duration = 0.025  # 25 ms frame
frame_length = int(frame_duration * fs)
hop_length = int(frame_length / 2)  # 50% overlap

# Calculate total number of frames
num_frames = int(np.floor((len(signal) - frame_length) / hop_length)) + 1

energy = np.zeros(num_frames)
zcr = np.zeros(num_frames)

# --- 3. Compute Short-Time Energy (STE) and Zero-Crossing Rate (ZCR) ---
for i in range(num_frames):
    start = i * hop_length
    end = start + frame_length
    frame = signal[start:end]
    
    # Short-Time Energy
    energy[i] = np.sum(frame ** 2)
    
    # Zero Crossing Rate
    zcr[i] = 0.5 * np.sum(np.abs(np.diff(np.sign(frame)))) / frame_length

# Normalize Energy and ZCR values between 0 and 1
energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)

# --- 4. Define Thresholds & Classify Frames ---
energy_thresh = 0.15
zcr_thresh = 0.25

classification = []  # 0: Silence, 1: Voiced, 2: Unvoiced

for e, zbit in zip(energy, zcr):
    if e < energy_thresh and zbit < zcr_thresh:
        classification.append(0)  # Silence
    elif e >= energy_thresh and zbit < zcr_thresh:
        classification.append(1)  # Voiced
    else:
        classification.append(2)  # Unvoiced

classification = np.array(classification)

# --- 5. Visualization ---
time_signal = np.linspace(0, len(signal) / fs, len(signal))
time_frames = np.linspace(0, len(signal) / fs, num_frames)

plt.figure(figsize=(12, 8))

# Original Audio Waveform
plt.subplot(4, 1, 1)
plt.plot(time_signal, signal, color='gray')
plt.title('Speech Signal Waveform')
plt.ylabel('Amplitude')

# Normalized Short-Time Energy
plt.subplot(4, 1, 2)
plt.plot(time_frames, energy, color='blue')
plt.axhline(y=energy_thresh, color='r', linestyle='--', label='Energy Threshold')
plt.title('Short-Time Energy (STE)')
plt.ylabel('Normalized Energy')
plt.legend()

# Normalized Zero-Crossing Rate
plt.subplot(4, 1, 3)
plt.plot(time_frames, zcr, color='green')
plt.axhline(y=zcr_thresh, color='r', linestyle='--', label='ZCR Threshold')
plt.title('Zero-Crossing Rate (ZCR)')
plt.ylabel('Normalized ZCR')
plt.legend()

# Classification Output
plt.subplot(4, 1, 4)
plt.plot(time_frames, classification, color='black', drawstyle='steps-pre')
plt.yticks([0, 1, 2], ['Silence', 'Voiced', 'Unvoiced'])
plt.title('Voiced / Unvoiced / Silence Separation')
plt.xlabel('Time (seconds)')
plt.ylabel('Region')

plt.tight_layout()
plt.show()
