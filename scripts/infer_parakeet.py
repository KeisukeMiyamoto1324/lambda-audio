import argparse
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol, cast

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

MODEL_ID = "nvidia/parakeet-tdt_ctc-0.6b-ja"
SAMPLE_RATE = 16_000
DEFAULT_PARQUET = Path(__file__).resolve().parents[1] / "data" / "data.parquet"


class Transcription(Protocol):
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Japanese ASR with NVIDIA Parakeet TDT-CTC 0.6B.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_PARQUET,
        help="A WAV file or data/data.parquet.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    return parser.parse_args()


def write_wav(source: Path | bytes, output_path: Path) -> None:
    audio_source = io.BytesIO(source) if isinstance(source, bytes) else source
    audio, _ = librosa.load(audio_source, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")


def transcribe_files(model: object, audio_paths: list[Path], batch_size: int) -> list[str]:
    transcribe = getattr(model, "transcribe")
    with torch.inference_mode():
        results = cast(list[Transcription], transcribe([str(path) for path in audio_paths], batch_size=batch_size))
    return [result.text for result in results]

def run_wav(model: object, input_path: Path, batch_size: int) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        audio_path = Path(temporary_directory) / "input.wav"
        write_wav(input_path, audio_path)
        prediction = transcribe_files(model, [audio_path], batch_size)[0]
    print(prediction)


def run_parquet(model: object, input_path: Path, batch_size: int) -> None:
    rows = pq.read_table(input_path, columns=["id", "audio", "text"]).to_pylist()
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        audio_paths: list[Path] = []
        for index, row in enumerate(rows):
            audio_bytes = row["audio"]["bytes"]
            if audio_bytes is None:
                raise ValueError(f"Row {index} does not contain audio bytes.")
            audio_path = temporary_path / f"{index:08d}.wav"
            write_wav(audio_bytes, audio_path)
            audio_paths.append(audio_path)

        predictions = transcribe_files(model, audio_paths, batch_size)
        for row, prediction in zip(rows, predictions, strict=True):
            result = {"id": row["id"], "reference": row["text"], "prediction": prediction}
            print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file was not found: {input_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if input_path.suffix.lower() not in {".wav", ".parquet"}:
        raise ValueError("Input must be a WAV or parquet file.")

    # ---------------------------------------------------------
    # Import NeMo after configuring the MPS fallback setting.
    # ---------------------------------------------------------
    import nemo.collections.asr as nemo_asr

    device = torch.device(args.device)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)
    model = model.to(device)
    model.eval()

    if input_path.suffix.lower() == ".wav":
        run_wav(model, input_path, args.batch_size)
        return
    run_parquet(model, input_path, args.batch_size)


if __name__ == "__main__":
    main()
