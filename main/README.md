# Double Voice Detection Module

A simple and easy-to-use module designed to detect voice spoofing and double voices in audio recordings. By utilizing voice embeddings and frame-level analysis, this tool identifies the presence of multiple speakers to ensure audio authenticity and prevent unauthorized external assistance.

## Flowchart

![picture 0](https://i.imgur.com/wfkWoPz.png)  

## Models & Approaches

We evaluated three primary baselines for the voice verification system, weighing performance against computational efficiency.

1. Pyannote.Audio  
    A robust industry standard for speaker diarization and verification.

    Pros: Highly accurate for separating speakers and offline processing.

    Cons: High latency makes it unsuitable for instant real-time feedback (processing often completes after the speech ends). It lacks phoneme-level verification, making it susceptible to visual spoofing (random lip movements).

2. SyncNet  
    A deep learning model for audio-visual synchronization that verifies identity based on lip movements and voice.

    Pros: Performs phoneme-level verification, making it robust against random lip movements and harder to spoof visually.

    Cons: Computationally expensive; requires a GPU for effective performance. As a "black box" deep learning model, it is difficult to quantify validation metrics compared to standard embedding distances.

3. Resemblyzer (Selected Approach)  
    A lightweight alternative focusing on high-speed voice embedding and verification.

    Pros: Low computational overhead with support for parallel processing. Significantly faster than Pyannote.Audio, offering near real-time performance on standard CPUs.

    Cons: Similar to Pyannote, it lacks phoneme-level verification.

We have chosen Resemblyzer as the core engine for this project. While SyncNet offers superior anti-spoofing via visual cues, the hardware requirements (GPU) are too high for accessible deployment. Resemblyzer offers the best balance, providing high-speed, parallelizable detection that fits within the computational constraints of standard proctoring/monitoring environments while maintaining reliable speaker differentiation.

## Installation

1. Install required dependencies using the provided script:

    **Windows:**

    ```bat
    install.bat
    ```

    **Linux/macOS:**

    ```bash
    bash install.sh
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
- `window_size` (float, optional): Size of the analysis window in seconds. Default: `1.0`
- `hop_size` (float, optional): Step size between windows in seconds. Default: `0.5`

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

## Performance Tips

- **Parallel Processing**: Enable `parallel=True` (default) for faster processing
- **Segment Length**: Optimal segment length is 5-15 seconds
- **Reference Segment**: Use first segment with clean audio of target speaker
- **Audio Quality**: Higher quality audio yields better results

## Requirements

Use Python 3.11.9 and install the packages using the provided installation scripts.
