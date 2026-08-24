import csv
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DATASET_ID = "litagin/reazon-speech-v2-clone"
INSERT_BATCH_SIZE = 10_000


def metadata_key(example: dict[str, Any]) -> str:
    file_name = cast(str, example["file_name"])
    prefix = "data/"
    suffix = ".flac"
    if not file_name.startswith(prefix) or not file_name.endswith(suffix):
        raise ValueError(f"Invalid metadata file name: {file_name}")
    return file_name[len(prefix) : -len(suffix)]


def dataset_revision(token: str) -> str:
    revision = HfApi(token=token).dataset_info(DATASET_ID).sha
    if revision is None:
        raise RuntimeError("The dataset revision is unavailable.")
    return revision


def metadata_index_revision(path: Path) -> str | None:
    if not path.exists():
        return None

    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT revision FROM dataset_info").fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("The metadata index has no dataset revision.")
    return cast(str, row[0])


def create_metadata_index(
    rows: Iterable[dict[str, Any]],
    path: Path,
    revision: str,
) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(
            "CREATE TABLE transcriptions ("
            "audio_key TEXT PRIMARY KEY, "
            "transcription TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE dataset_info (revision TEXT NOT NULL)"
        )

        values: list[tuple[str, str]] = []
        for row in rows:
            values.append(
                (metadata_key(row), cast(str, row["transcription"]).strip())
            )
            if len(values) < INSERT_BATCH_SIZE:
                continue
            connection.executemany(
                "INSERT OR IGNORE INTO transcriptions VALUES (?, ?)",
                values,
            )
            values.clear()

        if values:
            connection.executemany(
                "INSERT OR IGNORE INTO transcriptions VALUES (?, ?)",
                values,
            )
        connection.execute("INSERT INTO dataset_info VALUES (?)", (revision,))
        connection.commit()
    finally:
        connection.close()

    temporary_path.replace(path)


def read_metadata_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as metadata_file:
        rows = csv.DictReader(metadata_file)
        progress = tqdm(
            rows,
            desc="Indexing metadata",
            unit=" rows",
            dynamic_ncols=True,
        )
        yield from cast(Iterator[dict[str, Any]], progress)


def prepare_metadata_index(token: str, path: Path) -> None:
    revision = dataset_revision(token)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(f"{path}.lock")
    with lock:
        if metadata_index_revision(path) == revision:
            return

        metadata_path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                filename="metadata.csv",
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        )
        create_metadata_index(
            read_metadata_rows(metadata_path),
            path,
            revision,
        )


class MetadataReader:
    def __init__(self, path: Path) -> None:
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)

    def transcription(self, audio_key: str) -> str:
        row = self.connection.execute(
            "SELECT transcription FROM transcriptions WHERE audio_key = ?",
            (audio_key,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Transcription not found for audio key: {audio_key}")
        return cast(str, row[0])

    def close(self) -> None:
        self.connection.close()
