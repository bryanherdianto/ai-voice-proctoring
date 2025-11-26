"""
Example usage of the double_voice detection module.
"""

from double_voice import detect_double_voice


def example_basic_usage():
    timestamps = [[1.0, 10.0], [10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]]

    result = detect_double_voice(
        timestamps,
        audio="assets/a_yes_10_2.wav",
        window_size=1.0,
        hop_size=0.5,
        different_speaker_threshold=15.0,
        threshold=0.65,
    )

    print(result)


if __name__ == "__main__":
    try:
        example_basic_usage()
    except FileNotFoundError as e:
        print(f"\nSkipping basic example: {e}")
        print("Update audio file path in the example to run this demonstration.")
