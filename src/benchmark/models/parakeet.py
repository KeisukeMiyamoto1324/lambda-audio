from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models import parakeet_environment  # noqa: F401
from src.benchmark.models.base import BenchmarkModel
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch

# isort: split
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils import logging as nemo_logging


class ParakeetTranscriber:
    def __init__(self, model: EncDecHybridRNNTCTCBPEModel, config: TranscriptionConfig) -> None:
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
    nemo_logging.set_verbosity(nemo_logging.ERROR)
    checkpoint = hf_hub_download("nvidia/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja.nemo", token=token)
    model = EncDecHybridRNNTCTCBPEModel.restore_from(checkpoint, map_location=device)
    model.to(device=device, dtype=torch.float32).eval()
    # ---------------------------------------------------------
    # NeMo 2.4 conditional CUDA graphs fail with the current PyTorch.
    # Use regular GPU batch decoding without conditional graphs.
    # ---------------------------------------------------------
    decoding = OmegaConf.merge(OmegaConf.to_container(model.cfg.decoding, resolve=True), {"strategy": "greedy_batch", "greedy": {"use_cuda_graph_decoder": False}})
    model.change_decoding_strategy(decoding, decoder_type="rnnt", verbose=False)
    return ParakeetTranscriber(model, config)
