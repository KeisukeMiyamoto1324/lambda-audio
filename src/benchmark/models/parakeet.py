from time import perf_counter
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch

if TYPE_CHECKING:
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel


class ParakeetTranscriber:
    def __init__(self, model: "EncDecHybridRNNTCTCBPEModel", config: TranscriptionConfig) -> None:
        self.model = model
        self.config = config

    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]:
        start = perf_counter()
        arrays = decode_audio_batch(audio, self.config)
        with torch.inference_mode():
            hypotheses = self.model.transcribe(audio=arrays, batch_size=len(arrays), return_hypotheses=True, num_workers=0, verbose=False)
        seconds = (perf_counter() - start) / len(audio)
        return [
            TranscriptionResult(text=hypothesis.text.strip(), audio_seconds=array.size / self.config.sampling_rate, inference_seconds=seconds)
            for hypothesis, array in zip(hypotheses, arrays, strict=True)
        ]


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    checkpoint = hf_hub_download("nvidia/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja.nemo", token=token)
    model = EncDecHybridRNNTCTCBPEModel.restore_from(checkpoint, map_location=device)
    model.to(device=device, dtype=torch.float32).eval()
    decoding = OmegaConf.merge(model.cfg.decoding, {"strategy": "greedy_batch"})
    model.change_decoding_strategy(decoding, decoder_type="rnnt", verbose=False)
    return ParakeetTranscriber(model, config)
