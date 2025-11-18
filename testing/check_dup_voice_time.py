import argparse
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav


def main():
    parser = argparse.ArgumentParser(
        description="Compare voice similarity between two audio files (with optional time slice)."
    )
    parser.add_argument(
        "file1", type=str, help="Path to first audio file (e.g., your short voice)"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to second audio file (e.g., your longer recording)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds for file2 (default: 0.0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds for file2 (default: full duration)",
    )

    args = parser.parse_args()

    enc = VoiceEncoder()

    # --- Embedding for file1 (full file) ---
    wav1 = preprocess_wav(args.file1)
    emb1 = enc.embed_utterance(wav1)

    # --- Load file2 and slice if needed ---
    y2, sr = librosa.load(args.file2, sr=None)

    start_sample = int(args.start * sr)
    if args.end is None:
        end_sample = len(y2)
    else:
        end_sample = int(args.end * sr)

    if start_sample >= len(y2):
        raise ValueError(
            f"Start time {args.start}s is beyond audio duration ({len(y2)/sr:.2f}s)."
        )
    if end_sample > len(y2):
        print(
            f"Warning: end time {args.end}s exceeds duration ({len(y2)/sr:.2f}s). Clipping to end."
        )
        end_sample = len(y2)
    if start_sample >= end_sample:
        raise ValueError(
            "Start time must be less than end time, and segment must be non-empty."
        )

    segment = y2[start_sample:end_sample]

    # Preprocess sliced segment
    wav2 = preprocess_wav(segment, source_sr=sr)
    emb2 = enc.embed_utterance(wav2)

    # Cosine similarity
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
