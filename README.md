# AI Voice Proctoring System

An AI-powered voice proctoring system designed to monitor and verify the identity of individuals during online examinations or assessments. This system utilizes advanced voice recognition technology to ensure the integrity of the examination process by detecting any unauthorized assistance or impersonation.

This code uses Python version 3.11.9. To replicate the environment, you can use the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install --no-deps resemblyzer==0.1.4
```

or you can use conda:

```bash
conda create -n ai-voice-proctoring python=3.11.9
conda activate ai-voice-proctoring
pip install -r requirements.txt
pip install --no-deps resemblyzer==0.1.4
```

## Features

- Real-time voice activity detection
- Speaker diarization and verification
- Configurable sensitivity and thresholds for detection

## Approaches

1. **MediaPipe + WebRTC VAD + Pyannote.Audio**: This approach combines MediaPipe for face detection, WebRTC VAD for voice activity detection, and Pyannote.Audio for speaker diarization and verification. It provides a robust solution for real-time monitoring and identity verification. MediaPipe and WebRTC VAD are used for real-time processing, while Pyannote.Audio offers offline processing capabilities. The downside of this is that Pyannote.Audio is not real-time, hence might give response only after the person has finished speaking. This system doesn't do phoneme-level verification, hence might be fooled by random movements of lips in accordance with the audio.
2. **MediaPipe + WebRTC VAD + Resemblyzer**: This approach uses MediaPipe for face detection, WebRTC VAD for voice activity detection, and Resemblyzer for speaker verification. It is a simpler alternative to the first approach, focusing on real-time processing with less computational overhead. This system also doesn't do phoneme-level verification, hence might be fooled by random movements of lips in accordance with the audio.
3. **SyncNet**: This approach employs [SyncNet](https://github.com/joonson/syncnet_python), a deep learning model for audio-visual synchronization, to verify the identity of the speaker based on lip movements and voice. It is particularly effective in scenarios where visual cues are essential for verification. The only downside is that it requires a GPU to run effectively or it's very slow on CPU. However, it does **phoneme-level verification**, hence is more robust to random lip movements.
