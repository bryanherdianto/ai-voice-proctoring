"""
Double Voice Detection Module
"""

import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict
import warnings


def _process_segment_frames(
    audio_data: np.ndarray,
    reference_embedding: np.ndarray,
    sample_rate: int = 16000,
    window_size: float = 1.0,
    hop_size: float = 0.5,
    threshold: float = 0.6,
) -> Dict:
    enc = VoiceEncoder()

    window_samples = int(sample_rate * window_size)
    hop_samples = int(sample_rate * hop_size)
    min_frame_length = int(sample_rate * 0.3)

    frame_similarities = []

    for i in range(0, len(audio_data) - window_samples, hop_samples):
        frame = audio_data[i : i + window_samples]

        if len(frame) < min_frame_length:
            continue

        try:
            frame_emb = enc.embed_utterance(frame)
            frame_sim = np.dot(reference_embedding, frame_emb)
            frame_similarities.append(frame_sim)
        except Exception as e:
            warnings.warn(f"Error processing frame at index {i}: {e}")
            continue

    if not frame_similarities:
        return {
            "has_multiple_speakers": False,
            "overall_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "std_similarity": 0.0,
            "different_frames_percentage": 0.0,
            "total_frames": 0,
        }

    frame_sims = np.array(frame_similarities)
    overall_sim = np.mean(frame_sims)
    min_sim = np.min(frame_sims)
    max_sim = np.max(frame_sims)
    std_sim = np.std(frame_sims)

    different_frames = np.sum(frame_sims < threshold)
    total_frames = len(frame_sims)
    different_percentage = (different_frames / total_frames) * 100

    has_multiple_speakers = different_percentage > 20

    return {
        "has_multiple_speakers": has_multiple_speakers,
        "overall_similarity": float(overall_sim),
        "min_similarity": float(min_sim),
        "max_similarity": float(max_sim),
        "std_similarity": float(std_sim),
        "different_frames_percentage": float(different_percentage),
        "total_frames": total_frames,
    }


def _process_single_timestamp(args: Tuple) -> Tuple[int, float, float, Dict]:
    index, start_time, end_time, audio_path, reference_embedding, sample_rate = args

    try:
        y, sr = librosa.load(audio_path, sr=None)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if start_sample >= len(y):
            raise ValueError(f"Start time {start_time}s is beyond audio duration")
        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= end_sample:
            raise ValueError("Start time must be less than end time")

        segment = y[start_sample:end_sample]
        processed_segment = preprocess_wav(segment, source_sr=sr)

        results = _process_segment_frames(
            processed_segment, reference_embedding, sample_rate=16000
        )

        return (index, start_time, end_time, results)

    except Exception as e:
        warnings.warn(f"Error processing timestamp [{start_time}, {end_time}]: {e}")
        return (
            index,
            start_time,
            end_time,
            {
                "has_multiple_speakers": False,
                "overall_similarity": 0.0,
                "error": str(e),
            },
        )


def detect_double_voice(
    timestamps: List[List[float]],
    audio: str,
    parallel: bool = True,
    threshold: float = 0.6,
    different_speaker_threshold: float = 20.0,
) -> Dict:
    """
    Detect if multiple speakers are present in specified audio segments.

    Args:
        timestamps: 2D array of [start, end] time pairs in seconds
        audio: Path to audio file to analyze
        parallel: Whether to use parallel processing (default: True)
        threshold: Similarity threshold for frame-level detection (default: 0.6)
        different_speaker_threshold: Percentage threshold for multiple speaker detection (default: 20.0)

    Returns:
        Dictionary containing detection result and suspicious segments
    """

    if not timestamps:
        return {"multiple_speakers_detected": "NO", "suspicious_segments": []}

    if not isinstance(timestamps, (list, np.ndarray)):
        raise TypeError("timestamps must be a list or numpy array")

    timestamps = np.array(timestamps)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError("timestamps must be a 2D array with shape (n, 2)")

    enc = VoiceEncoder()

    # Extract reference from first timestamp
    first_start, first_end = timestamps[0]
    y_ref, sr_ref = librosa.load(audio, sr=None)
    start_sample = int(first_start * sr_ref)
    end_sample = int(first_end * sr_ref)
    reference_segment = y_ref[start_sample:end_sample]
    reference_wav = preprocess_wav(reference_segment, source_sr=sr_ref)
    reference_embedding = enc.embed_utterance(reference_wav)

    results = []

    # Process remaining timestamps (skip first since it's the reference)
    timestamps_to_check = timestamps[1:] if len(timestamps) > 1 else []

    if parallel and len(timestamps_to_check) > 1:
        with ProcessPoolExecutor() as executor:
            futures = []
            for i, (start_time, end_time) in enumerate(timestamps_to_check, start=1):
                args = (
                    i,
                    float(start_time),
                    float(end_time),
                    audio,
                    reference_embedding,
                    16000,
                )
                future = executor.submit(_process_single_timestamp, args)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        results.sort(key=lambda x: x[0])
    else:
        for i, (start_time, end_time) in enumerate(timestamps_to_check, start=1):
            args = (
                i,
                float(start_time),
                float(end_time),
                audio,
                reference_embedding,
                16000,
            )
            result = _process_single_timestamp(args)
            results.append(result)

    suspicious_segments = []
    for index, start_time, end_time, analysis in results:
        if analysis.get("has_multiple_speakers", False):
            suspicious_segments.append([start_time, end_time])

    detection_result = "YES" if suspicious_segments else "NO"

    print(f"Multiple speakers detected: {detection_result}")
    if detection_result == "YES":
        for seg in suspicious_segments:
            print(f"  [{seg[0]:.1f}s - {seg[1]:.1f}s]")

    return {
        "multiple_speakers_detected": detection_result,
        "suspicious_segments": suspicious_segments,
    }
