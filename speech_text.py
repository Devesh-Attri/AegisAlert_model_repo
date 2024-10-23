import requests
import json

AUDIO_FILE = "temp_audio.wav"
DEEPGRAM_API_KEY = "ff2cdb5878677800ae178708dc0fe07b4edafc1f"

def transcribe_audio_deepgram(audio_file):
    try:
        with open(audio_file, "rb") as file:
            buffer_data = file.read()

        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/mp3"
        }
        response = requests.post(url, headers=headers, data=buffer_data)

        if response.status_code == 200:
            transcription_result = response.json()
            transcript = transcription_result["results"]["channels"][0]["alternatives"][0]["transcript"]
            print("Transcription result: ")
            print(transcript)
            return transcript
        else:
            print(f"Failed to transcribe audio. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")

transcript = transcribe_audio_deepgram(AUDIO_FILE)