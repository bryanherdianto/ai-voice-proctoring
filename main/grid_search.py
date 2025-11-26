import time
import os
import librosa
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, f1_score
from double_voice import detect_double_voice

DATASET_FOLDER = "./assets"


def build_ground_truth(folder: str) -> dict:
    """
    Build ground truth from file naming convention:
    - Files with '_no_' in name -> label 0 (single speaker)
    - Files with '_yes_' in name -> label 1 (multiple speakers)
    """
    ground_truth = {}
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            if "_no_" in filename:
                ground_truth[filename] = 0
            elif "_yes_" in filename:
                ground_truth[filename] = 1
    return ground_truth


GROUND_TRUTH = build_ground_truth(DATASET_FOLDER)

# Added overlap_ratio to make sure hop_size is calculated correctly
param_grid = {
    "window_size": [0.5, 1.0, 1.5],
    "threshold": [0.55, 0.60, 0.65],
    "diff_threshold": [15.0, 20.0, 25.0],
    "overlap_ratio": [0.5],
}


def get_audio_duration(path):
    """Get the duration of an audio file in seconds."""
    return librosa.get_duration(path=path)


# Generate timestamps for windowed segments
def generate_timestamps(duration):
    timestamps = []
    start = 0.0
    window = 10.0
    while start < duration:
        end = min(start + window, duration)
        timestamps.append([start, end])
        start += window
    return timestamps


def run_benchmark():
    results = []

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(combinations)} parameter combinations...")

    for i, params in enumerate(combinations):
        win_size = params["window_size"]
        # Ensure hop_size is calculated if your detector needs it
        hop_size = win_size * (1 - params["overlap_ratio"])

        y_true = []
        y_pred = []
        rtf_scores = []

        for filename, actual_label in GROUND_TRUTH.items():
            filepath = os.path.join(DATASET_FOLDER, filename)

            try:
                duration = get_audio_duration(filepath)

                # Segment audio into windows for detection
                timestamps = generate_timestamps(duration)

                start_time = time.time()

                # --- CALL YOUR MODULE ---
                output = detect_double_voice(
                    timestamps=timestamps,
                    audio=filepath,
                    parallel=False,
                    threshold=params["threshold"],
                    different_speaker_threshold=params["diff_threshold"],
                    window_size=win_size,
                    hop_size=hop_size,
                )

                process_time = time.time() - start_time

                # Calculate Real-Time Factor (Lower is better)
                rtf = process_time / duration if duration > 0 else 0
                rtf_scores.append(rtf)

                detected = 1 if output["multiple_speakers_detected"] == "YES" else 0
                y_true.append(actual_label)
                y_pred.append(detected)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # METRICS
        avg_rtf = sum(rtf_scores) / len(rtf_scores) if rtf_scores else 0
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append(
            {
                "window_size": win_size,
                "threshold": params["threshold"],
                "diff_thresh": params["diff_threshold"],
                "avg_rtf": round(avg_rtf, 4),
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1, 4),
            }
        )

    # --- DATAFRAME ANALYSIS (Moved inside function or return df) ---
    df = pd.DataFrame(results)

    if df.empty:
        print("No results found.")
        return

    # 1. Normalize F1 (Higher is better -> map to 0..1)
    f1_min, f1_max = df["f1_score"].min(), df["f1_score"].max()
    if f1_max - f1_min == 0:
        df["f1_norm"] = 0  # Or 1, depending on if it's all good or all bad
    else:
        df["f1_norm"] = (df["f1_score"] - f1_min) / (f1_max - f1_min)

    # 2. Normalize RTF (Lower is better -> We still normalize 0..1 based on value)
    rtf_min, rtf_max = df["avg_rtf"].min(), df["avg_rtf"].max()
    if rtf_max - rtf_min == 0:
        df["rtf_norm"] = 0
    else:
        df["rtf_norm"] = (df["avg_rtf"] - rtf_min) / (rtf_max - rtf_min)

    # 3. Calculate Euclidean Distance to Best
    df["distance_to_ideal"] = ((1 - df["f1_norm"]) ** 2 + (df["rtf_norm"]) ** 2) ** 0.5

    df.to_csv("benchmark_results.csv", index=False)

    print("\n--- TOP 3 BALANCED CONFIGS (Best Trade-off) ---")

    # Sort by smallest distance
    print(df.sort_values(by="distance_to_ideal", ascending=True).head(3))

    best = df.sort_values(by="distance_to_ideal", ascending=True).iloc[0]
    print(
        f"\nWINNER: Window={best['window_size']}, Thresh={best['threshold']}, Diff={best['diff_thresh']}"
    )


if __name__ == "__main__":
    run_benchmark()
