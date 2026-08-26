from pathlib import Path

from pydantic import BaseModel, ConfigDict, FilePath


class InferenceSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_path: FilePath
    reference: str


def load_inference_samples(input_dir: Path) -> list[InferenceSample]:
    audio_paths = sorted(input_dir.glob("*.flac"))
    if not audio_paths:
        raise ValueError("The input directory contains no FLAC files.")

    samples: list[InferenceSample] = []
    for audio_path in audio_paths:
        reference_path = audio_path.with_suffix(".txt")
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference transcription not found: {reference_path}"
            )
        samples.append(
            InferenceSample(
                audio_path=audio_path,
                reference=reference_path.read_text(encoding="utf-8").strip(),
            )
        )
    return samples
