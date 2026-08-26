from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from src.stage1 import download as stage1_download
from src.stage2 import download as stage2_download


@pytest.mark.parametrize(
    ("download_module", "repository_variable", "repository"),
    [
        (stage1_download, "HF_STAGE1_REPO", "owner/stage1"),
        (stage2_download, "HF_STAGE2_REPO", "owner/stage2"),
    ],
)
def test_download_uses_stage_environment_and_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    download_module: ModuleType,
    repository_variable: str,
    repository: str,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        f"HF_TOKEN=test-token\n{repository_variable}={repository}\n"
    )
    output_dir = tmp_path / "output"
    download = MagicMock()

    monkeypatch.setattr(download_module, "DOTENV_PATH", dotenv_path)
    monkeypatch.setattr(download_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(download_module, "snapshot_download", download)

    download_module.main()  # type: ignore[attr-defined]

    download.assert_called_once_with(
        repo_id=repository,
        repo_type="model",
        local_dir=output_dir,
        token="test-token",
        allow_patterns=["model.ckpt", "logs/**/metrics.csv"],
    )
