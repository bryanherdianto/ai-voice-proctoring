import webrtcvad
import sounddevice as sd
import numpy as np
import queue
import sys

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
CHANNELS = 1

vad = webrtcvad.Vad(2)  # aggressiveness 0-3 (higher = more aggressive)

q = queue.Queue()


def callback(indata, frames, time, status):
    """sounddevice callback — put raw int16 bytes into queue"""
    if status:
        print(status, file=sys.stderr)
    # indata is float32 by default; convert to int16
    audio16 = (indata[:, 0] * 32767).astype(np.int16)
    q.put(audio16.tobytes())


def main():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        print("Listening (press Ctrl+C to stop)...")
        try:
            while True:
                frame_bytes = q.get()
                if len(frame_bytes) != FRAME_SAMPLES * 2:
                    continue
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
                print("SPEECH" if is_speech else "silence")
        except KeyboardInterrupt:
            print("Stopped")


if __name__ == "__main__":
    main()
