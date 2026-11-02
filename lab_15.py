#Write a MATLAB/Python program for text-to-speech conversion and record speech signal.

import pyttsx3
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from matplotlib.widgets import Button

# Create TTS engine
engine = pyttsx3.init()

# Text input
text = " Hello, how are you? Welcome to our speech processing laboratory."

# Convert text to speech and save
engine.save_to_file(text, "output.wav")
engine.runAndWait()

print("Speech recorded and saved as output.wav")

# Read the recorded speech
fs, speech = wavfile.read("output.wav")

# Convert stereo to mono
if len(speech.shape) > 1:
    speech = speech.mean(axis=1)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Plot speech signal
ax.plot(speech)
ax.set_title("Text-to-Speech Recorded Signal")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.grid()

# Play button function
def play_speech(event):
    sd.stop()
    sd.play(speech, fs)

# Stop button function
def stop_speech(event):
    sd.stop()

# Create buttons
ax_play = plt.axes([0.30, 0.07, 0.20, 0.08])
ax_stop = plt.axes([0.55, 0.07, 0.20, 0.08])

button_play = Button(ax_play, "Play Speech")
button_stop = Button(ax_stop, "Stop")

button_play.on_clicked(play_speech)
button_stop.on_clicked(stop_speech)

plt.show()
