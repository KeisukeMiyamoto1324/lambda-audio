from pathlib import Path
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from src.training import download as download_module


def test_download_uses_training_environment_and_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=test-token\nHF_MODEL_REPO=owner/model\n")
    output_dir = tmp_path / "output"
    download = MagicMock()

    monkeypatch.setattr(download_module, "DOTENV_PATH", dotenv_path)
    monkeypatch.setattr(download_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(download_module, "snapshot_download", download)

    download_module.main()  # type: ignore[attr-defined]

    download.assert_called_once_with(
        repo_id="owner/model",
        repo_type="model",
        local_dir=output_dir,
        token="test-token",
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
    )
