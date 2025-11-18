# Double Voice Detection Module

A simple and easy-to-use module for detecting multiple speakers in audio recordings using voice embeddings and frame-level analysis.

## Installation

1. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Import the module in your Python script:

    ```python
    from double_voice import detect_double_voice
    ```

## Quick Start

```python
from double_voice import detect_double_voice

# Define timestamp ranges (first segment is used as reference)
timestamps = [
    [1.0, 7.0],    # Reference segment
    [10.0, 15.0],  # Check this segment
    [20.0, 25.0]   # Check this segment
]

# Run detection
result = detect_double_voice(timestamps, audio="path/to/recording.wav")

# Output:
# Multiple speakers detected: YES
#   [10.0s - 15.0s]
```

## API Reference

### `detect_double_voice(timestamps, audio, **kwargs)`

Detect if multiple speakers are present in specified audio segments.

**Parameters:**

- `timestamps` (List[List[float]]): 2D array of [start, end] time pairs in seconds. First segment is used as reference.
- `audio` (str): Path to audio file to analyze
- `parallel` (bool, optional): Whether to use parallel processing. Default: `True`
- `threshold` (float, optional): Similarity threshold for frame-level detection. Default: `0.6`
- `different_speaker_threshold` (float, optional): Percentage threshold for multiple speaker detection. Default: `20.0`

**Returns:**

Dictionary containing:

- `multiple_speakers_detected` (str): "YES" or "NO"
- `suspicious_segments` (List): List of [start, end] timestamps where multiple speakers detected

## How It Works

1. **Reference Embedding**: Uses the first timestamp segment as reference voice
2. **Segment Analysis**: For each remaining timestamp range:
   - Extracts the audio segment
   - Analyzes it frame-by-frame (1-second windows with 0.5s overlap)
   - Compares each frame against the reference embedding
3. **Detection Logic**:
   - If more than 20% of frames have similarity < 0.6, multiple speakers are detected
   - Returns YES/NO with suspicious segment timestamps

## Example Use Cases

### 1. Exam Proctoring

```python
exam_timestamps = [
    [0, 30],       # Reference (first 30 seconds)
    [30, 300],     # Question 1
    [300, 600],    # Question 2
    [600, 900],    # Question 3
]

result = detect_double_voice(exam_timestamps, audio="exam_recording.wav")
```

### 2. Interview Verification

```python
interview_segments = [
    [0, 10.0],      # Reference (intro)
    [10.5, 45.2],   # Question 1 answer
    [60.0, 120.5],  # Question 2 answer
    [150.0, 200.0]  # Question 3 answer
]

result = detect_double_voice(interview_segments, audio="interview.wav")
```

### 3. Continuous Monitoring

```python
duration = 300
segment_length = 10
timestamps = [[i, i+segment_length] for i in range(0, duration, segment_length)]

result = detect_double_voice(
    timestamps,
    audio="recording.wav",
    parallel=True
)
```

## Performance Tips

- **Parallel Processing**: Enable `parallel=True` (default) for faster processing
- **Segment Length**: Optimal segment length is 5-15 seconds
- **Reference Segment**: Use first segment with clean audio of target speaker
- **Audio Quality**: Higher quality audio yields better results

## Requirements

Use Python 3.11.9 and install the packages listed in requirements.txt.
