from pathlib import Path

import pytest

from src.data.metadata import (
    MetadataReader,
    create_metadata_index,
    metadata_index_revision,
)


def test_metadata_index_stores_revision_and_transcriptions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.sqlite3"
    create_metadata_index(
        [
            {
                "file_name": "data/000/audio.flac",
                "transcription": " 音声 ",
            }
        ],
        path,
        revision="revision",
    )

    reader = MetadataReader(path)
    transcription = reader.transcription("000/audio")
    reader.close()

    assert transcription == "音声"
    assert metadata_index_revision(path) == "revision"


def test_metadata_index_keeps_first_duplicate_transcription(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.sqlite3"
    rows = [
        {"file_name": "data/000/audio.flac", "transcription": "一"},
        {"file_name": "data/000/audio.flac", "transcription": "二"},
    ]
    create_metadata_index(rows, path, "revision")
    reader = MetadataReader(path)

    transcription = reader.transcription("000/audio")
    reader.close()

    assert transcription == "一"


def test_metadata_reader_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "metadata.sqlite3"
    create_metadata_index([], path, revision="revision")
    reader = MetadataReader(path)

    with pytest.raises(ValueError, match="Transcription not found"):
        reader.transcription("000/missing")

    reader.close()
