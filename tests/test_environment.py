import os
from pathlib import Path

from pytest import MonkeyPatch

from src.environment import (
    CUDA_ALLOCATOR_CONFIGURATION,
    TRAINING_CPU_THREADS,
    configure_cuda_allocator,
    configure_training_cpu_threads,
    load_hf_token,
)


def test_configure_cuda_allocator(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    configure_cuda_allocator()

    assert os.environ["PYTORCH_ALLOC_CONF"] == CUDA_ALLOCATOR_CONFIGURATION


def test_configure_training_cpu_threads(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    configure_training_cpu_threads()

    assert os.environ["OPENBLAS_NUM_THREADS"] == TRAINING_CPU_THREADS
    assert os.environ["OMP_NUM_THREADS"] == TRAINING_CPU_THREADS


def test_load_hf_token_from_dotenv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=test-token\n")

    assert load_hf_token(dotenv_path) == "test-token"
