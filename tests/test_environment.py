from pathlib import Path

from pytest import MonkeyPatch

from src.shared.environment import load_hf_token


def test_load_hf_token_from_dotenv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=test-token\n")

    assert load_hf_token(dotenv_path) == "test-token"
