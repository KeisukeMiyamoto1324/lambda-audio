from time import perf_counter

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch


class WhisperTranscriber:
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        config: TranscriptionConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device

    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]:
        inference_start = perf_counter()
        arrays = decode_audio_batch(audio, self.config)
        inputs = self.processor.feature_extractor(
            arrays,
            sampling_rate=self.config.sampling_rate,
            padding="max_length",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # Decode short Japanese clips greedily without retry heuristics.
        # ---------------------------------------------------------
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_features=inputs["input_features"].to(self.device, dtype=self.model.dtype),
                attention_mask=inputs["attention_mask"].to(self.device),
                language="ja",
                task="transcribe",
                max_new_tokens=self.config.max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                return_timestamps=False,
                condition_on_prev_tokens=False,
                compression_ratio_threshold=None,
                logprob_threshold=None,
                no_speech_threshold=None,
                use_cache=True,
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        inference_seconds = (perf_counter() - inference_start) / len(audio)
        return [
            TranscriptionResult(text=text.strip(), audio_seconds=array.size / self.config.sampling_rate, inference_seconds=inference_seconds)
            for text, array in zip(texts, arrays, strict=True)
        ]


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    model_id = "openai/whisper-large-v3-turbo"
    processor = WhisperProcessor.from_pretrained(model_id, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, token=token, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return WhisperTranscriber(model, processor, config, device)
