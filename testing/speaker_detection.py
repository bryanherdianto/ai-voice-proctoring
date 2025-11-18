import os
from pyannote.audio import Pipeline
import warnings
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load environment variables from .env file

# Ignore all warnings
warnings.filterwarnings("ignore")

# Pretrained pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
)


def main():
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization on an audio file."
    )
    parser.add_argument("src", type=str, help="Path to specify source audio file")
    args = parser.parse_args()

    # Run diarization on a file
    diarization = pipeline(args.src)

    # Count speakers
    speakers = len(
        set(label for _, _, label in diarization.itertracks(yield_label=True))
    )
    print("Speakers detected:", speakers)


if __name__ == "__main__":
    main()
