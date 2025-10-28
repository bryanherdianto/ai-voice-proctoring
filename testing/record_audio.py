import sounddevice as sd
import numpy as np
import wave
import threading
import os
from datetime import datetime


class AudioRecorder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []

        # Create assets folder if it doesn't exist
        if not os.path.exists("assets"):
            os.makedirs("assets")

    def audio_callback(self, indata, frames, time, status):
        """Callback function to capture audio"""
        if self.recording:
            self.audio_data.append(indata.copy())

    def input_listener(self):
        """Listen for Enter to stop recording"""
        while self.recording:
            try:
                input()  # Just wait for Enter press
                self.recording = False
                break
            except:
                break

    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.audio_data = []

        print("Recording started...")
        print("Press Enter to stop")

        # Start input listener thread
        input_thread = threading.Thread(target=self.input_listener)
        input_thread.daemon = True
        input_thread.start()

        # Start audio stream
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
        ):
            try:
                while self.recording:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.recording = False

    def save_recording(self, filename=None):
        """Save the recorded audio to file"""

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

        if not filename.endswith(".wav"):
            filename += ".wav"

        filepath = os.path.join("assets", filename)

        # Combine all audio chunks
        audio_array = np.concatenate(self.audio_data, axis=0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        print(f"Recording saved: {filepath}")
        print(f"Duration: {len(audio_array) / self.sample_rate:.2f} seconds")
        return filepath


def main():
    print("=== Audio Recorder ===")

    custom_name = input("Enter filename (or press Enter for auto): ").strip()
    if not custom_name:
        custom_name = None

    recorder = AudioRecorder()
    recorder.start_recording()
    recorder.save_recording(custom_name)


if __name__ == "__main__":
    main()

"""
This is a short audio sample for testing microphone clarity, tone, and background noise levels. 
The quick brown fox jumps over the lazy dog.
"""

"""
This is a longer audio sample intended for testing voice recording quality, microphone positioning, and environmental acoustics. 
When recording audio, it's important to maintain a consistent distance from the microphone and speak at a natural volume.

Background noise, echo, and distortion can greatly affect the final sound quality. To achieve the best results, record in a quiet room 
with minimal reflections. The quick brown fox jumps over the lazy dog several times, testing rhythm, pacing, and articulation across 
various tones and pitches. This concludes the long-form sample recording.
"""