"""
Example usage of the double_voice detection module.
"""

from double_voice import detect_double_voice


def example_basic_usage():
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    timestamps = [[1.0, 7.0], [10.0, 15.0], [20.0, 25.0]]

    result = detect_double_voice(timestamps, audio="assets/dob_voice_long.wav")

    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(result)


def example_exam_proctoring():
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Exam Proctoring")
    print("=" * 70)

    exam_timestamps = [
        [0, 30],
        [30, 300],
        [300, 600],
        [600, 900],
    ]

    print("\nAnalyzing exam recording...")
    print("Note: Update audio path to match your actual file")

    # result = detect_double_voice(
    #     exam_timestamps,
    #     audio="path/to/exam_recording.wav"
    # )


def example_continuous_monitoring():
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Continuous Monitoring")
    print("=" * 70)

    duration = 120
    segment_length = 10
    timestamps = [[i, i + segment_length] for i in range(0, duration, segment_length)]

    print(f"\nMonitoring {duration}s recording in {len(timestamps)} segments...")
    print("Note: Update audio path to match your actual file")

    # result = detect_double_voice(
    #     timestamps,
    #     audio="path/to/recording.wav",
    #     parallel=True
    # )


def example_custom_parameters():
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Custom Parameters")
    print("=" * 70)

    timestamps = [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]]

    print("\nUsing custom thresholds...")
    print("Note: Update audio path to match your actual file")

    # result = detect_double_voice(
    #     timestamps,
    #     audio="path/to/recording.wav",
    #     threshold=0.7,
    #     different_speaker_threshold=15.0
    # )


if __name__ == "__main__":
    print("Double Voice Detection - Usage Examples")
    print("=" * 70)

    try:
        example_basic_usage()
    except FileNotFoundError as e:
        print(f"\nSkipping basic example: {e}")
        print("Update audio file path in the example to run this demonstration.")

    example_exam_proctoring()
    example_continuous_monitoring()
    example_custom_parameters()

    print("\n\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
