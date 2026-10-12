from time import perf_counter

import mlx.core as mx
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from mlx.model import LambdaAudioModel
from mlx.model.lfm2 import LayerCache
from src.shared.transcription import TranscriptionConfig, TranscriptionResult, decode_audio_batch


def generate(model: LambdaAudioModel, input_values: mx.array, padding_mask: mx.array, bos_token_id: int, eos_token_id: int, max_new_tokens: int) -> list[list[int]]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    # ---------------------------------------------------------
    # Encode audio and project cross-attention keys and values once.
    # ---------------------------------------------------------
    features, padding_mask = model.encode(input_values, padding_mask)
    audio_key_values = model.decoder.project_audio_key_values(features)
    mx.eval(audio_key_values, padding_mask)
    cache: list[LayerCache] = [None] * len(model.decoder.model.layers)
    tokens = mx.full((input_values.shape[0], 1), bos_token_id, dtype=mx.int32)
    finished = mx.zeros((input_values.shape[0],), dtype=mx.bool_)
    generated = []

    # ---------------------------------------------------------
    # Greedy batch decoding reuses attention and convolution caches.
    # Synchronize each step so lazy graphs do not grow across tokens.
    # ---------------------------------------------------------
    for _ in range(max_new_tokens):
        logits = model.decoder(tokens, audio_key_values, padding_mask, cache)
        next_tokens = mx.where(finished, eos_token_id, mx.argmax(logits[:, -1], axis=-1)).astype(mx.int32)
        finished = finished | (next_tokens == eos_token_id)
        all_finished = mx.all(finished)
        mx.eval(next_tokens, all_finished, cache)
        generated.append(next_tokens)
        if bool(all_finished.item()):
            break
        tokens = next_tokens[:, None]
    return mx.stack(generated, axis=1).tolist()


def transcribe(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    audio: list[bytes],
) -> list[TranscriptionResult]:
    mx.synchronize()
    start = perf_counter()
    arrays = decode_audio_batch(audio, config)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")
    batch = feature_extractor(arrays, sampling_rate=config.sampling_rate, padding=True, return_attention_mask=True, return_tensors="np")
    dtype = model.encoder.feature_extractor.conv_layers[0].conv.weight.dtype
    input_values = mx.array(batch["input_values"], dtype=dtype)
    padding_mask = mx.array(batch["attention_mask"] == 0)
    token_ids = generate(model, input_values, padding_mask, tokenizer.bos_token_id, tokenizer.eos_token_id, config.max_new_tokens)
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    seconds = (perf_counter() - start) / len(arrays)
    return [
        TranscriptionResult(text=text.strip(), audio_seconds=array.size / config.sampling_rate, inference_seconds=seconds) for text, array in zip(texts, arrays, strict=True)
    ]
