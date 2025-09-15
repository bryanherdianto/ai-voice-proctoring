import os
from pyannote.audio import Pipeline
import warnings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Ignore all warnings
warnings.filterwarnings("ignore")

# Pretrained pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
)

# Run diarization on a file
diarization = pipeline("assets/talking.mp3")

# Count speakers
speakers = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
print("Speakers detected:", speakers)
