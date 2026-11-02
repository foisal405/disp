
import numpy as np
import matplotlib.pyplot as plt
import librosa

# 1. Load MP3 Audio Signal
file_path = "drop_water.mp3"

signal, fs = librosa.load(file_path, sr=None, mono=True)

# Normalize the signal safely
max_amplitude = np.max(np.abs(signal))

if max_amplitude > 0:
    signal = signal / max_amplitude

# 2. Framing Parameters
frame_duration = 0.025  # 25 ms
frame_length = int(frame_duration * fs)
hop_length = frame_length // 2  # 50% overlap

if len(signal) < frame_length:
    raise ValueError("Audio file is too short for 25 ms framing.")

# Calculate total number of frames
num_frames = 1 + (len(signal) - frame_length) // hop_length

energy = np.zeros(num_frames)
zcr = np.zeros(num_frames)

# 3. Calculate Short-Time Energy and ZCR
for i in range(num_frames):
    start = i * hop_length
    frame = signal[start:start + frame_length]

    # Short-Time Mean-Square Energy
    energy[i] = np.mean(frame ** 2)

    # Zero-Crossing Rate
    zcr[i] = np.mean(
        np.abs(np.diff(np.signbit(frame)))
    )

# 4. Normalize Energy for Classification
# Use maximum energy so silence is not artificially raised to zero-to-one range
if np.max(energy) > 0:
    energy_norm = energy / np.max(energy)
else:
    energy_norm = energy.copy()

# 5. Define Thresholds
energy_thresh = 0.02
zcr_thresh = 0.10

# Classification:
# 0 = Silence, 1 = Voiced, 2 = Unvoiced
classification = np.zeros(num_frames, dtype=int)

for i in range(num_frames):
    if energy_norm[i] < energy_thresh:
        classification[i] = 0  # Silence
    elif zcr[i] < zcr_thresh:
        classification[i] = 1  # Voiced
    else:
        classification[i] = 2  # Unvoiced

# 6. Time Axes
time_signal = np.arange(len(signal)) / fs
time_frames = np.arange(num_frames) * hop_length / fs

# 7. Display Results
plt.figure(figsize=(12, 9))

# Original Audio Waveform
plt.subplot(4, 1, 1)
plt.plot(time_signal, signal, color="gray")
plt.title("MP3 Speech Signal Waveform")
plt.ylabel("Amplitude")
plt.xlim(0, len(signal) / fs)

# Short-Time Energy
plt.subplot(4, 1, 2)
plt.plot(time_frames, energy_norm, color="blue")
plt.axhline(
    y=energy_thresh, color="red",
    linestyle="--", label="Energy Threshold"
)
plt.title("Normalized Short-Time Energy")
plt.ylabel("Energy")
plt.legend()

# Zero-Crossing Rate
plt.subplot(4, 1, 3)
plt.plot(time_frames, zcr, color="green")
plt.axhline(
    y=zcr_thresh, color="red",
    linestyle="--", label="ZCR Threshold"
)
plt.title("Zero-Crossing Rate")
plt.ylabel("ZCR")
plt.legend()

# Voiced / Unvoiced / Silence Classification
plt.subplot(4, 1, 4)
plt.step(time_frames, classification, where="post", color="black")
plt.yticks([0, 1, 2], ["Silence", "Voiced", "Unvoiced"])
plt.title("Voiced / Unvoiced / Silence Separation")
plt.xlabel("Time (seconds)")
plt.ylabel("Region")
plt.ylim(-0.5, 2.5)

plt.tight_layout()
plt.show()
