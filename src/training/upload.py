import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs/training"


def main() -> None:
    # ---------------------------------------------------------
    # Load credentials and upload the selected training artifacts.
    # ---------------------------------------------------------
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        repo_id=os.environ["HF_MODEL_REPO"],
        folder_path=OUTPUT_DIR,
        repo_type="model",
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
        commit_message="Upload training artifacts",
    )


if __name__ == "__main__":
    main()
