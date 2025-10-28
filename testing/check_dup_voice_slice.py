from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import librosa
import soundfile as sf
import os


def process_slice_with_frames(
    slice_data, reference_embedding, slice_num, start_time, end_time
):
    """Process a single audio slice with frame-by-frame analysis"""
    enc = VoiceEncoder()
    sample_rate = 16000

    wav = slice_data

    try:
        window_size = int(sample_rate * 1.0)
        hop_size = int(sample_rate * 0.5)

        frame_similarities = []
        frame_times = []

        for i in range(0, len(wav) - window_size, hop_size):
            frame = wav[i : i + window_size]
            if len(frame) < sample_rate * 0.3:
                continue

            frame_emb = enc.embed_utterance(frame)
            frame_sim = np.dot(reference_embedding, frame_emb)
            frame_similarities.append(frame_sim)

            frame_time = start_time + (i / sample_rate)
            frame_times.append(frame_time)

        if not frame_similarities:
            return None

        frame_sims = np.array(frame_similarities)
        overall_sim = np.mean(frame_sims)
        min_sim = np.min(frame_sims)
        max_sim = np.max(frame_sims)
        std_sim = np.std(frame_sims)

        different_frames = np.sum(frame_sims < 0.6)
        total_frames = len(frame_sims)
        different_percentage = (different_frames / total_frames) * 100

        multiple_speakers = different_percentage > 30

        return (
            slice_num,
            overall_sim,
            start_time,
            end_time,
            min_sim,
            max_sim,
            std_sim,
            different_percentage,
            multiple_speakers,
            frame_similarities,
            frame_times,
        )

    except Exception as e:
        print(f"Error processing slice {slice_num}: {e}")
        return None


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 10)
    return f"{minutes:02d}:{secs:02d}.{millis}"


def analyze_audio_slices(
    reference_file, long_file, slice_duration=10, parallel=True, save_processed=True
):
    """Analyze long audio in slices with frame-level analysis"""

    enc = VoiceEncoder()
    me = preprocess_wav(reference_file)

    original_audio, original_sr = librosa.load(long_file, sr=None)
    original_duration = len(original_audio) / original_sr

    friend = preprocess_wav(long_file)
    processed_duration = len(friend) / 16000

    print(f"Original: {original_duration:.1f}s at {original_sr}Hz")
    print(f"Processed: {processed_duration:.1f}s at 16000Hz")

    if abs(original_duration - processed_duration) > 1:
        print(f"Duration changed by {abs(original_duration - processed_duration):.1f}s")

    if save_processed:
        processed_filename = os.path.join("assets", "processed_for_validation.wav")
        sf.write(processed_filename, friend, 16000)
        print(f"Saved: {processed_filename}")

    emb_me = enc.embed_utterance(me)

    sample_rate = 16000
    slice_samples = slice_duration * sample_rate
    total_slices = int(np.ceil(len(friend) / slice_samples))
    total_duration = len(friend) / sample_rate

    print(f"\nSlices: {total_slices} x {slice_duration}s")
    print(f"Duration: {format_timestamp(total_duration)} ({total_duration:.1f}s)")
    print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
    print("-" * 90)

    start_time_proc = time.time()
    results = []

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = []

            for i in range(total_slices):
                start_idx = i * slice_samples
                end_idx = min((i + 1) * slice_samples, len(friend))
                slice_data = friend[start_idx:end_idx]

                start_sec = start_idx / sample_rate
                end_sec = end_idx / sample_rate

                if len(slice_data) < sample_rate:
                    continue

                future = executor.submit(
                    process_slice_with_frames, slice_data, emb_me, i, start_sec, end_sec
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                results.append(result)

                (
                    slice_num,
                    overall_sim,
                    start_sec,
                    end_sec,
                    min_sim,
                    max_sim,
                    std_sim,
                    diff_pct,
                    multiple_speakers,
                    frame_sims,
                    frame_times,
                ) = result

                timestamp_range = (
                    f"[{format_timestamp(start_sec)}-{format_timestamp(end_sec)}]"
                )
                status = "ALERT" if multiple_speakers else "OK"

                print(
                    f"Slice {slice_num + 1:2d} {timestamp_range} {status:5s}: "
                    f"Avg={overall_sim:.3f} Min={min_sim:.3f} Max={max_sim:.3f} "
                    f"Diff={diff_pct:.1f}% Std={std_sim:.3f}"
                )

        results.sort(key=lambda x: x[0])

    else:
        for i in range(total_slices):
            start_idx = i * slice_samples
            end_idx = min((i + 1) * slice_samples, len(friend))
            slice_data = friend[start_idx:end_idx]

            start_sec = start_idx / sample_rate
            end_sec = end_idx / sample_rate

            if len(slice_data) < sample_rate:
                continue

            result = process_slice_with_frames(
                slice_data, emb_me, i, start_sec, end_sec
            )
            if result is None:
                continue
            results.append(result)

            (
                slice_num,
                overall_sim,
                start_sec,
                end_sec,
                min_sim,
                max_sim,
                std_sim,
                diff_pct,
                multiple_speakers,
                frame_sims,
                frame_times,
            ) = result

            timestamp_range = (
                f"[{format_timestamp(start_sec)}-{format_timestamp(end_sec)}]"
            )
            status = "ALERT" if multiple_speakers else "OK"

            print(
                f"Slice {slice_num + 1:2d} {timestamp_range} {status:5s}: "
                f"Avg={overall_sim:.3f} Min={min_sim:.3f} Max={max_sim:.3f} "
                f"Diff={diff_pct:.1f}% Std={std_sim:.3f}"
            )

    elapsed_time = time.time() - start_time_proc

    print("-" * 90)
    print(f"Processing time: {elapsed_time:.2f}s")

    suspicious_slices = [r for r in results if r[8]]

    print(f"\nTotal slices: {len(results)}")
    print(f"Suspicious slices: {len(suspicious_slices)}/{len(results)}")

    if suspicious_slices:
        print(f"\nMultiple speakers detected:")
        for result in suspicious_slices:
            (
                slice_num,
                overall_sim,
                start_sec,
                end_sec,
                min_sim,
                max_sim,
                std_sim,
                diff_pct,
                multiple_speakers,
                frame_sims,
                frame_times,
            ) = result

            timestamp_range = (
                f"[{format_timestamp(start_sec)}-{format_timestamp(end_sec)}]"
            )
            print(
                f"  Slice {slice_num + 1}: {timestamp_range} - {diff_pct:.1f}% different frames"
            )
            print(f"    Similarity: {min_sim:.3f} to {max_sim:.3f}")

    if len(suspicious_slices) > 0:
        print("\nResult: MULTIPLE SPEAKERS DETECTED")
    else:
        print("\nResult: Single speaker")

    return results


if __name__ == "__main__":
    print("Voice Similarity Analysis\n")

    results = analyze_audio_slices(
        "assets/my_voice_short.wav",
        "assets/my_voice_long.wav",
        slice_duration=10,
        parallel=True,
        save_processed=True,
    )
