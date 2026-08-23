import os
from pathlib import Path

from dotenv import load_dotenv


def load_hf_token(dotenv_path: Path = Path(".env")) -> str:
    load_dotenv(dotenv_path=dotenv_path)
    return os.environ["HF_TOKEN"]
