from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_audio.stt.models.whisper import Model
from mlx_audio.stt.models.whisper.tokenizer import get_tokenizer

from whisper.audio import TranscriptionResult, decode_audio_batch
from whisper.config import MODEL_ID, InferenceConfig


@dataclass(frozen=True)
class WhisperEngine:
    model: Model
    config: InferenceConfig

    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]:
        if len(audio) != 1:
            raise ValueError("Whisper benchmarking supports batch size 1 only.")
        mx.synchronize()
        start = perf_counter()
        array = decode_audio_batch(audio, self.config)[0]
        # ---------------------------------------------------------
        # Use Japanese greedy decoding with no retries or silence skips.
        # The backend expects sample_len for the generation token limit.
        # ---------------------------------------------------------
        output = self.model.generate(
            array,
            language="ja",
            task="transcribe",
            sample_len=self.config.max_new_tokens,
            temperature=0.0,
            without_timestamps=True,
            word_timestamps=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=None,
            logprob_threshold=None,
            no_speech_threshold=None,
            verbose=None,
        )
        mx.synchronize()
        return [TranscriptionResult(text=output.text.strip(), audio_seconds=array.size / self.config.sampling_rate, inference_seconds=perf_counter() - start)]


def load_engine(config: InferenceConfig, token: str) -> WhisperEngine:
    if not mx.metal.is_available():
        raise RuntimeError("Whisper MLX requires an Apple Silicon Mac with Metal.")
    if config.sampling_rate != 16000:
        raise ValueError("Whisper requires a sampling rate of 16000 Hz.")
    # ---------------------------------------------------------
    # Match LambdaAudio's buffer cache policy and exclude loading from timing.
    # Pass a local snapshot so the backend makes no unauthenticated Hub calls.
    # ---------------------------------------------------------
    mx.set_cache_limit(0)
    mx.clear_cache()
    model_path = snapshot_download(MODEL_ID, allow_patterns=["config.json", "model.safetensors", "multilingual.tiktoken"], token=token)
    if not (Path(model_path) / "multilingual.tiktoken").is_file():
        raise FileNotFoundError("The model snapshot must contain multilingual.tiktoken.")
    model = Model.from_pretrained(model_path, dtype=mx.float16)
    model.eval()
    get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="ja", task="transcribe", model_path=Path(model_path))
    mx.eval(model.parameters())
    mx.synchronize()
    return WhisperEngine(model, config)
