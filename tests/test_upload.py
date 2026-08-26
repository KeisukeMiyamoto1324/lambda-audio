from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from src.stage1 import upload as stage1_upload
from src.stage2 import upload as stage2_upload


@pytest.mark.parametrize(
    ("upload_module", "repository_variable", "repository", "stage"),
    [
        (stage1_upload, "HF_STAGE1_REPO", "owner/stage1", "Stage 1"),
        (stage2_upload, "HF_STAGE2_REPO", "owner/stage2", "Stage 2"),
    ],
)
def test_upload_uses_stage_environment_and_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    upload_module: ModuleType,
    repository_variable: str,
    repository: str,
    stage: str,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"HF_TOKEN=test-token\n{repository_variable}={repository}\n")
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
        repo_id=repository,
        folder_path=output_dir,
        repo_type="model",
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
        commit_message=f"Upload {stage} training artifacts",
    )
