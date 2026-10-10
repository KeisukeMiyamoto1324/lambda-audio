from time import perf_counter

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration, CohereAsrProcessor

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch


class CohereTranscriber:
    def __init__(self, model: CohereAsrForConditionalGeneration, processor: CohereAsrProcessor, config: TranscriptionConfig, device: torch.device) -> None:
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device

    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]:
        start = perf_counter()
        arrays = decode_audio_batch(audio, self.config)
        inputs = self.processor(arrays, sampling_rate=self.config.sampling_rate, language="ja", punctuation=True, return_tensors="pt")
        inputs.to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, num_beams=1, do_sample=False, use_cache=True)
        texts = self.processor.decode(outputs, skip_special_tokens=True, audio_chunk_index=inputs["audio_chunk_index"], language="ja")
        seconds = (perf_counter() - start) / len(audio)
        return [
            TranscriptionResult(text=text.strip(), audio_seconds=array.size / self.config.sampling_rate, inference_seconds=seconds)
            for text, array in zip(texts, arrays, strict=True)
        ]


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    model_id = "CohereLabs/cohere-transcribe-03-2026"
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = CohereAsrForConditionalGeneration.from_pretrained(model_id, token=token, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device).eval()
    return CohereTranscriber(model, processor, config, device)
