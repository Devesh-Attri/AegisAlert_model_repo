import sounddevice as sd
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import queue
import soundfile as sf

# Initialize the pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)
pipeline.to(torch.device("mps"))  # Use MPS for macOS (Apple Silicon)

# Queue to hold audio chunks
audio_queue = queue.Queue()
SAMPLING_RATE = 16000  # Required sampling rate for pyannote.audio

# Set to keep track of unique speakers
unique_speakers = set()

# Callback function to capture audio chunks
def audio_callback(indata, frames, time, status):
    """Called whenever new audio data is available."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Start audio stream (continuous capture)
def start_stream():
    """Starts the audio stream to capture real-time audio."""
    return sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,  # Mono audio required
        dtype='float32',
        callback=audio_callback
    )

# Process real-time audio chunks
def process_audio_stream():
    """Processes audio from the queue using the diarization pipeline."""
    global unique_speakers

    with start_stream() as stream:
        print("Streaming audio... Press Ctrl+C to stop.")
        
        buffer = np.zeros((0,))  # Buffer to accumulate audio
        try:
            while True:
                # Retrieve audio chunk from the queue
                chunk = audio_queue.get()
                buffer = np.concatenate((buffer, chunk.flatten()))

                # If buffer exceeds 2 seconds of audio, process it
                if len(buffer) >= SAMPLING_RATE * 2:
                    # Convert buffer to a PyTorch tensor (float32)
                    waveform = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0)

                    # Run the diarization pipeline on the chunk
                    diarization = pipeline({"waveform": waveform, "sample_rate": SAMPLING_RATE})

                    # Track and count unique speakers
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        unique_speakers.add(speaker)
                        print(f"Speaker {speaker} from {turn.start:.1f}s to {turn.end:.1f}s")

                    # Display the total count of unique speakers
                    print(f"Number of people in the scene: {len(unique_speakers)}")

                    # Clear buffer after processing
                    buffer = np.zeros((0,))

        except KeyboardInterrupt:
            print("Audio streaming stopped.")

def process_audio_file(audio_file_path):
    """
    Process a single audio file and return the number of speakers and diarization results.
    """
    # Load the audio file
    waveform, sample_rate = sf.read(audio_file_path)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Convert to PyTorch tensor
    waveform = torch.tensor(waveform).unsqueeze(0).float()

    # Run the diarization pipeline
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Extract unique speakers and their speaking times
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((turn.start, turn.end))

    return len(speakers), speakers

def print_diarization_results(num_speakers, speakers):
    """
    Print the diarization results in a formatted way.
    """
    print(f"Number of speakers detected: {num_speakers}")
    for speaker, times in speakers.items():
        total_time = sum(end - start for start, end in times)
        print(f"\n{speaker}:")
        print(f"  Total speaking time: {total_time:.2f} seconds")
        print("  Speaking intervals:")
        for start, end in times:
            print(f"    {start:.2f}s to {end:.2f}s")

if __name__ == "__main__":
    audio_file_path = "temp_audio.wav"  # Make sure this matches the path in realtime_models.py
    num_speakers, speakers = process_audio_file(audio_file_path)
    print_diarization_results(num_speakers, speakers)
