import os
from pathlib import Path


def configure_environment() -> None:
    cache_path = Path(__file__).resolve().parents[3] / "outputs" / ".cache" / "matplotlib"
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_path)


configure_environment()
