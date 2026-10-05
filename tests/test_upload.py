from pathlib import Path
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from src.training import upload as upload_module


def test_upload_uses_training_environment_and_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=test-token\nHF_MODEL_REPO=owner/model\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    api = MagicMock()
    api_class = MagicMock(return_value=api)

    monkeypatch.setattr(upload_module, "DOTENV_PATH", dotenv_path)
    monkeypatch.setattr(upload_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(upload_module, "HfApi", api_class)

    upload_module.main()  # type: ignore[attr-defined]

    api_class.assert_called_once_with(token="test-token")
    api.upload_folder.assert_called_once_with(
        repo_id="owner/model",
        folder_path=output_dir,
        repo_type="model",
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
        commit_message="Upload training artifacts",
    )
