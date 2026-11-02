#Write a MATLAB/Python program to recognize speech signal 

import speech_recognition as sr
def recognize_speech_from_file(file_path: str):
    """Transcribes speech from a .wav audio file using Google Web Speech API."""
    recognizer = sr.Recognizer()
    try:
        # Load the audio file
        with sr.AudioFile(file_path) as source:
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
    audio_file = "Recording.wav" 
    recognize_speech_from_file(audio_file)
