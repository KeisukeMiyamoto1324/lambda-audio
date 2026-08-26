from pathlib import Path
from typing import cast

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

DATASET_ID = "litagin/reazon-speech-v2-clone"


def dataset_revision(token: str) -> str:
    revision = HfApi(token=token).dataset_info(DATASET_ID).sha
    if revision is None:
        raise RuntimeError("The dataset revision is unavailable.")
    return revision


def list_reazon_speech_shards(token: str, revision: str) -> list[str]:
    files = HfApi(token=token).list_repo_files(
        DATASET_ID,
        repo_type="dataset",
        revision=revision,
    )
    shards = sorted(
        file_name
        for file_name in files
        if file_name.startswith("data/") and file_name.endswith(".tar")
    )
    if not shards:
        raise RuntimeError("The ReazonSpeech dataset has no audio shards.")
    return shards


def download_reazon_speech_file(
    token: str,
    revision: str,
    filename: str,
    local_dir: Path | None = None,
) -> Path:
    with disable_progress_bars():
        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=local_dir,
        )
    return Path(cast(str, path))


def prepare_reazon_speech_validation_shard(
    token: str,
    cache_dir: Path,
) -> Path:
    revision = dataset_revision(token)
    shard_file = list_reazon_speech_shards(token, revision)[0]
    validation_cache = cache_dir / "validation"
    validation_cache.mkdir(parents=True, exist_ok=True)
    return download_reazon_speech_file(
        token,
        revision,
        shard_file,
        validation_cache,
    )
