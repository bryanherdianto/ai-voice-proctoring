import ffmpeg

ffmpeg.input("assets/talking.mp3").output("assets/talking_16k.wav", ac=1, ar=16000).run(overwrite_output=True)