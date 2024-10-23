import numpy as np
import pyaudio
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import noisereduce as nr
from pyannote.audio.pipelines import OverlappedSpeechDetection
from tensorflow.keras.models import load_model
import librosa
import wave
import datetime
import math
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# Function to record audio from microphone
def record_audio(duration=10, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
    return audio_data, sample_rate

# Function to reduce background noise
def reduce_noise(audio_data, sample_rate):
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
    return reduced_noise

# Function to test Hugging Face token authentication
def test_authentication(auth_token):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
        overlap_detector = OverlappedSpeechDetection.from_pretrained("pyannote/overlapped-speech-detection", use_auth_token=auth_token)
        print("Pipeline loaded successfully.")
        return pipeline, overlap_detector
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None, None

# Function for speaker diarization using pyannote.audio
def speaker_diarization(file_path, pipeline):
    try:
        diarization = pipeline(file_path)
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
        return len(speakers), diarization
    except Exception as e:
        print(f"Error performing diarization: {e}")
        return None, None

# Function to detect and process overlapping speech
def detect_overlap(file_path, overlap_detector):
    overlap_segments = overlap_detector(file_path)
    return overlap_segments

# Saving the recorded audio to a temporary file for pyannote.audio
def save_audio_to_file(audio_data, sample_rate, file_path="temp_audio.wav"):
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    audio_segment.export(file_path, format="wav")

# Function to split audio into segments
def split_audio_into_segments(audio_file, segment_length_ms=1000):
    audio = AudioSegment.from_wav(audio_file)
    num_segments = math.ceil(len(audio) / segment_length_ms)
    segments = [audio[i * segment_length_ms:(i + 1) * segment_length_ms] for i in range(num_segments)]
    return segments

# Function to extract features for both models
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, sr=16000)
    mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
    return mel_features.reshape(1, -1)

# Function to predict gender and emotion
def predict_gender_and_emotion(file_path, gender_model, emotion_model):
    features = extract_features(file_path)
    gender_pred = gender_model.predict(features)
    gender = 'Male' if gender_pred > 0.5 else 'Female'
    emotion_pred = emotion_model.predict(features)
    emotion_index = np.argmax(emotion_pred, axis=1)[0]
    emotions = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise', 'disgust']
    emotion = emotions[emotion_index]
    return gender, emotion

# Function to process full audio for gender and emotion using embeddings
def process_full_audio(audio_file, gender_model, emotion_model):
    encoder = VoiceEncoder()
    wav = preprocess_wav(Path(audio_file))
    if len(wav) == 0:
        print("Error: Audio file is empty or could not be processed.")
        return 0, 0
    embedding = encoder.embed_utterance(wav)
    embedding = np.expand_dims(embedding, axis=0)
    gender_pred = gender_model.predict(embedding)
    gender = 'Male' if gender_pred[0] == 0 else 'Female'
    emotion_pred = emotion_model.predict(embedding)
    emotion = np.argmax(emotion_pred, axis=1)
    return gender, emotion

def analyze_audio(audio_data, sample_rate, pipeline, overlap_detector, gender_model, emotion_model):
    file_path = "temp_audio.wav"
    save_audio_to_file(audio_data, sample_rate, file_path)
    num_speakers, diarization = speaker_diarization(file_path, pipeline)
    overlap_segments = detect_overlap(file_path, overlap_detector)
    if num_speakers is None:
        print("Diarization failed.")
        return 0, None, None

    num_men, num_women = 0, 0
    segments = split_audio_into_segments(file_path, segment_length_ms=1000)
    for segment in segments:
        segment_file = "segment_temp.wav"
        segment.export(segment_file, format="wav")
        gender, _ = predict_gender_and_emotion(segment_file, gender_model, emotion_model)
        if gender == 'Male':
            num_men += 1
        else:
            num_women += 1

    print(f"Number of unique speakers: {num_speakers}")
    print(f"Total Men: {num_men}")
    print(f"Total Women: {num_women}")

    current_time = datetime.datetime.now().time()
    if num_women == 1 and num_men == 0 and current_time.hour >= 20:
        print("Warning: Alone woman detected at night")
    if num_women > 0 and num_men > num_women:
        print("Warning: Woman surrounded by multiple men")
    
    return num_speakers, num_men, num_women

if __name__ == "__main__":
    duration = 10  # Duration in seconds for recording
    sample_rate = 16000  # Sample rate for recording
    auth_token = "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"  # Replace with your Hugging Face token

    pipeline, overlap_detector = test_authentication(auth_token)
    emotion_model = load_model('models/ser_model.h5')
    gender_model = load_model('models/model.h5')

    if pipeline and overlap_detector and emotion_model and gender_model:
        audio_data, sample_rate = record_audio(duration, sample_rate)
        reduced_audio = reduce_noise(audio_data, sample_rate)

        num_speakers, num_men, num_women = analyze_audio(reduced_audio, sample_rate, pipeline, overlap_detector, gender_model, emotion_model)
    else:
        print("Authentication failed or model loading failed. Please check your token and model paths.")
