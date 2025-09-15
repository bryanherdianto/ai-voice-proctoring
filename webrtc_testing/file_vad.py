import webrtcvad
import soundfile as sf


def frame_generator(frame_ms, audio, sample_rate):
    """Yield consecutive frames of frame_ms length (bytes)."""
    n = int(sample_rate * (frame_ms / 1000.0))
    start = 0
    while start + n <= len(audio):
        frame = audio[start : start + n]
        yield frame.tobytes()
        start += n


def simple_vad(path, aggressiveness=2, frame_ms=30):
    vad = webrtcvad.Vad(aggressiveness)

    audio, sr = sf.read(path, dtype="int16")
    assert sr == 16000, "Audio must be 16 kHz"

    frames = frame_generator(frame_ms, audio, sr)

    speech_frames = 0
    total_frames = 0
    for frame in frames:
        is_speech = vad.is_speech(frame, sr)
        total_frames += 1
        if is_speech:
            speech_frames += 1
    speech_ratio = speech_frames / total_frames if total_frames else 0
    print(f"Aggressiveness: {aggressiveness}, frame_ms: {frame_ms}")
    print(f"Speech frames: {speech_frames}/{total_frames} ({speech_ratio:.2%})")


if __name__ == "__main__":
    simple_vad("assets/talking_16k.wav", aggressiveness=2, frame_ms=30)
