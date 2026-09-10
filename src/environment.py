import os
from pathlib import Path

from dotenv import load_dotenv

CUDA_ALLOCATOR_CONFIGURATION = "expandable_segments:True,garbage_collection_threshold:0.8"
TRAINING_CPU_THREADS = "1"


def configure_training_cpu_threads() -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = TRAINING_CPU_THREADS
    os.environ["OMP_NUM_THREADS"] = TRAINING_CPU_THREADS


def configure_cuda_allocator() -> None:
    os.environ["PYTORCH_ALLOC_CONF"] = CUDA_ALLOCATOR_CONFIGURATION


def load_hf_token(dotenv_path: Path = Path(".env")) -> str:
    load_dotenv(dotenv_path=dotenv_path)
    return os.environ["HF_TOKEN"]
