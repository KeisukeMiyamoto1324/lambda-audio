import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs/training"


def main() -> None:
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    snapshot_download(
        repo_id=os.environ["HF_MODEL_REPO"],
        repo_type="model",
        local_dir=OUTPUT_DIR,
        token=os.environ["HF_TOKEN"],
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
    )


if __name__ == "__main__":
    main()
