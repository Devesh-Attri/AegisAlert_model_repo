# **Women's Safety Alert System** 🚨👩‍💻

This AI-powered Women's Safety Alert System continuously monitors real-time voice input to detect situations of distress or danger and generates alerts when necessary. The system leverages multiple machine learning models to analyze voice gender, emotion, speech content, and speaker presence, combined with location data, to ensure enhanced safety.

## Features
- **Voice Gender Analysis (model.h5):** Detects the gender of the speaker in real time to determine if the speaker is a woman.
- **Speech Emotion Recognition (ser_model.h5):** Identifies emotional states (Sad, Happy, Disgust, Angry, Pleasant Surprise) to detect distress signals.
- **Speech-to-Text (speech_text.py):** Converts spoken words to text using Deepgram API for further analysis.
- **Speaker Diarization (speaker_diarization.py):** Determines the number of people in the scene and detects overlapping speech.
- **Threat Detection:** Monitors keywords spoken by the user (from keywords.txt) in situations of distress and correlates it with a crime-prone location (using crimes.csv).
- **Location-Based Alerting:** Uses the device's real-time location to compare with high-crime areas from crimes.csv. If the location is identified as threat-prone, and distress is detected, an alert is generated.

## How It Works
- The system records voice input continuously and processes it every 20 seconds.
- It reduces background noise and analyzes the voice for gender and emotion.
- The speech-to-text module checks for distress keywords like "help," "danger," or screams.
- The current location is compared against crime records. If the location is dangerous and distress signals are detected (e.g., "disgust" emotion or distress keywords), the user is prompted to confirm if they are safe.
- If the user confirms danger, an alert is generated immediately.

## Requirements

- Python 3.9+
- TensorFlow for deep learning models
- Pyannote.audio for speaker diarization
- Deepgram API for speech-to-text
- Pyaudio, Noisereduce, Torchaudio for audio processing

## Installation

- Clone the repository:
```
git clone https://github.com/Devesh-Attri/AegisAlert_model_repo.git
```
- Install dependencies:
```
pip install -r requirements.txt
```
- Add your Deepgram API key and Hugging Face authentication token.

## Usage:
- Run the main program:
```
python main.py
```
The system will continuously monitor voice input and process it in real-time to detect distress and generate alerts.

## Contributions
Feel free to fork this repository, submit issues, and make pull requests to improve the system!
