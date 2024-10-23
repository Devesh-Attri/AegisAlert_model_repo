import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import queue
from speech_text import transcribe_audio_deepgram
from speaker_diarization import process_audio_file, print_diarization_results
import soundfile as sf
import random

# Load models
gender_model = load_model('voice_gender_analysis_model/results/model.h5')
emotion_model = load_model('models/ser_model.h5')

# Load crime data and keywords
crime_data = pd.read_csv('crime.csv')
with open('keywords.txt', 'r') as f:
    keywords = set(f.read().splitlines())

# Global variables
audio_queue = queue.Queue()
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 10
TEMP_AUDIO_FILE = "temp_audio.wav"

# List of locations from the crime.csv file
LOCATIONS = crime_data['nm_pol'].tolist()

def extract_features(audio_data, sr=RATE, n_mfcc=40):
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=1)
    mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=n_mfcc)
    return mfccs  # Shape will be (n_mfcc, time_steps)

def analyze_audio(audio_data):
    try:
        # Extract features for gender model
        gender_features = extract_feature(audio_data, mel=True).reshape(1, -1)
        
        # Gender analysis
        gender_prob = gender_model.predict(gender_features)
        gender = "male" if gender_prob[0][0] > 0.5 else "female"

        # Extract features for emotion model
        emotion_features = extract_features_for_emotion(audio_data)

        # Emotion analysis
        emotion_probs = emotion_model.predict(emotion_features)[0]
        emotions = ['Angry', 'Fear', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Pleasant Surprise']
        emotion = emotions[np.argmax(emotion_probs)]

        # Speech to text
        try:
            transcript = transcribe_audio_deepgram(TEMP_AUDIO_FILE)
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            transcript = ""

        return gender, emotion, transcript, emotion_probs
    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        return "Unknown", "Unknown", "", []

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X = np.frombuffer(file_name, dtype=np.float32)
    sample_rate = RATE
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

def extract_features_for_emotion(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=audio_np, sr=RATE, n_mfcc=40)
    target_length = 1024
    if mfccs.shape[1] < target_length:
        pad_width = ((0, 0), (0, target_length - mfccs.shape[1]))
        mfccs = np.pad(mfccs, pad_width, mode='constant')
    elif mfccs.shape[1] > target_length:
        mfccs = mfccs[:, :target_length]
    mfccs_flattened = mfccs.flatten()[:1024]
    if len(mfccs_flattened) < 1024:
        mfccs_flattened = np.pad(mfccs_flattened, (0, 1024 - len(mfccs_flattened)), mode='constant')
    mfccs_reshaped = mfccs_flattened.reshape(1, -1)
    return mfccs_reshaped

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def is_threat_prone_area(location):
    if location in crime_data['nm_pol'].values:
        total_crime = crime_data.loc[crime_data['nm_pol'] == location, 'totalcrime'].values[0]
        return total_crime > 10, total_crime
    return False, 0

def save_temp_audio(audio_data, filename=TEMP_AUDIO_FILE):
    sf.write(filename, audio_data, RATE)

def process_audio():
    print("Recording audio for 10 seconds...")
    buffer = np.array([], dtype=np.float32)
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        chunk = audio_queue.get()
        buffer = np.concatenate((buffer, chunk.flatten()))

    print("Processing audio...")
    save_temp_audio(buffer)
    gender, emotion, transcript, emotion_probs = analyze_audio(buffer)

    # Perform speaker diarization
    num_speakers, speakers = process_audio_file(TEMP_AUDIO_FILE)

    print("\n--- Audio Analysis Results ---")
    print(f"Gender: {gender}")
    print(f"Emotion: {emotion}")
    print(f"Emotion probabilities:")
    emotions = ['Angry', 'Fear', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Pleasant Surprise']
    for emotion, prob in zip(emotions, emotion_probs):
        print(f"  {emotion}: {prob:.4f}")
    print(f"Transcript: {transcript}")
    
    print("\n--- Speaker Diarization Results ---")
    print_diarization_results(num_speakers, speakers)

    # Simulate a location name (replace with actual location data in production)
    current_location = random.choice(LOCATIONS)

    # Check for threat conditions
    is_threat_prone, total_crime = is_threat_prone_area(current_location)
    is_threat = (emotion in ["Angry", "Fear", "Disgust"]) and is_threat_prone

    print(f"\nLocation: {current_location}")
    print(f"Total crime in this area: {total_crime}")
    print(f"Is threat-prone area: {'Yes' if is_threat_prone else 'No'}")

    if is_threat:
        print("\nALERT: Potential threat detected!")
    else:
        print("\nNo immediate threat detected.")

def main():
    print("Starting audio stream...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=RATE, blocksize=CHUNK):
        process_audio()

    print("Audio processing completed.")

if __name__ == "__main__":
    main()
