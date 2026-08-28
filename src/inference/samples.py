from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict


class InferenceSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_name: str
    audio: bytes
    reference: str


def load_inference_samples(input_path: Path) -> list[InferenceSample]:
    table = pq.read_table(input_path, columns=["audio", "text"])
    samples: list[InferenceSample] = []
    for row in table.to_pylist():
        samples.append(
            InferenceSample(
                audio_name=row["audio"]["path"],
                audio=row["audio"]["bytes"],
                reference=row["text"],
            )
        )
    if not samples:
        raise ValueError("The input parquet contains no samples.")
    return samples
