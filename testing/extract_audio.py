import ffmpeg
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from a video file and save as a separate audio file."
    )
    parser.add_argument("src", type=str, help="Path to specify source video file")
    parser.add_argument("dest", type=str, help="Path to specify destination audio file")
    args = parser.parse_args()

    ffmpeg.input(args.src).output(args.dest, ac=1, ar=16000).run(overwrite_output=True)


if __name__ == "__main__":
    main()
