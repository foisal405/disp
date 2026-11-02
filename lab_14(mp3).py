
import speech_recognition as sr
from pydub import AudioSegment

def recognize_speech_from_file(file_path: str):

    """Transcribes speech from an MP3 audio file using Google Web Speech API."""

    recognizer = sr.Recognizer()

    try:
        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(file_path)
        audio.export("temp.wav", format="wav")

        # Load the audio file
        with sr.AudioFile("temp.wav") as source:

            print(f"Processing audio file: '{file_path}'...")

            # Read the entire audio file
            audio_data = recognizer.record(source)

        # Recognize speech using Google Speech Recognition
        print("Translating speech to text...")

        text = recognizer.recognize_google(audio_data)

        print("\n--- Recognition Result ---")
        print(f"Transcript: {text}")

    except sr.UnknownValueError:
        print("Error: Google Speech Recognition could not understand the audio.")

    except sr.RequestError as e:
        print(f"Error: Could not request results from Google Speech Recognition service; {e}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found in your directory.")

if __name__ == "__main__":

    audio_file = "audio.mp3"
    recognize_speech_from_file(audio_file)
