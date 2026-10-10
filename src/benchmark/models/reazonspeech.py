from time import perf_counter

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch


class ReazonSpeechTranscriber:
    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, config: TranscriptionConfig, device: torch.device) -> None:
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device

    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]:
        start = perf_counter()
        arrays = decode_audio_batch(audio, self.config)
        inputs = self.processor(arrays, sampling_rate=self.config.sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt")
        lengths = inputs["attention_mask"].sum(dim=1)
        with torch.inference_mode():
            logits = self.model(input_values=inputs["input_values"].to(self.device, dtype=self.model.dtype)).logits
            predicted_ids = logits.argmax(dim=-1).cpu()
        output_lengths = self.model._get_feat_extract_output_lengths(lengths)
        texts = self.processor.batch_decode([ids[: int(length)] for ids, length in zip(predicted_ids, output_lengths, strict=True)], skip_special_tokens=True)
        seconds = (perf_counter() - start) / len(audio)
        return [
            TranscriptionResult(text=text.strip(), audio_seconds=array.size / self.config.sampling_rate, inference_seconds=seconds)
            for text, array in zip(texts, arrays, strict=True)
        ]


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    model_id = "reazon-research/japanese-wav2vec2-base-rs35kh"
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, token=token, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device).eval()
    return ReazonSpeechTranscriber(model, processor, config, device)
