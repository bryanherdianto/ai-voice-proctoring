import ffmpeg

ffmpeg.input("assets/video_sample.mp4").output("assets/audio_sample.wav", ac=1, ar=16000).run(overwrite_output=True)