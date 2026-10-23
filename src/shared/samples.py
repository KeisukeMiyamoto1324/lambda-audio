from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from src.shared.languages import OutputLanguage


class InferenceSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_name: str
    audio: bytes
    reference: str
    lang: OutputLanguage = "ja"


def load_inference_samples(input_path: Path) -> list[InferenceSample]:
    if input_path.suffix.lower() == ".wav":
        return [
            InferenceSample(
                audio_name=input_path.name,
                audio=input_path.read_bytes(),
                reference="",
            )
        ]

    if input_path.suffix.lower() != ".parquet":
        raise ValueError("The input must be a WAV or Parquet file.")

    table = pq.read_table(input_path, columns=["audio", "text"])
    samples: list[InferenceSample] = []
    for index, row in enumerate(table.to_pylist(), start=1):
        samples.append(
            InferenceSample(
                audio_name=f"{input_path.name}:{index}",
                audio=row["audio"]["bytes"],
                reference=row["text"],
            )
        )
    if not samples:
        raise ValueError("The input parquet contains no samples.")
    return samples


def load_multilingual_inference_samples(input_path: Path) -> list[InferenceSample]:
    # ---------------------------------------------------------
    # Read language-specific references for training inference.
    # ---------------------------------------------------------
    table = pq.read_table(input_path, columns=["id", "audio", "lang", "text"])
    if table.num_rows == 0:
        raise ValueError("The input parquet contains no samples.")
    return [InferenceSample(audio_name=row["id"], audio=row["audio"]["bytes"], lang=row["lang"], reference=row["text"]) for row in table.to_pylist()]
