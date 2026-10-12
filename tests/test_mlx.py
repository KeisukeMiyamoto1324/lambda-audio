from functools import partial
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from pydantic import ValidationError
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from mlx.inference import generate, transcribe  # noqa: E402
from mlx.model import LambdaAudioModel  # noqa: E402
from mlx.model.lfm2 import LayerCache  # noqa: E402
from src.benchmark.config import BenchmarkConfig  # noqa: E402
from src.modeling.lambda_audio import LambdaAudioModel as TorchLambdaAudioModel  # noqa: E402
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM  # noqa: E402
from src.modeling.zipformer import ZipformerConfig, ZipformerModel  # noqa: E402


@pytest.fixture(params=["default"])
def models(request: pytest.FixtureRequest) -> tuple[TorchLambdaAudioModel, LambdaAudioModel]:
    mode = request.param
    torch.manual_seed(7)
    encoder_config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2), (8, 3, 2)),
        extractor_mode=mode,
        encoder_dim=(8, 16, 8),
        downsampling_factor=(1, 2, 2),
        num_encoder_layers=(1, 1, 1),
        encoder_unmasked_dim=(8, 8, 8),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2, 2, 2),
        feedforward_dim=(16, 32, 16),
        cnn_module_kernel=(3, 3, 3),
    )
    decoder_config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
    )
    reference = TorchLambdaAudioModel(ZipformerModel(encoder_config), AudioLfm2ForCausalLM(decoder_config), encoder_hidden_size=16).eval()
    with torch.no_grad():
        reference.decoder.model.layers[1].cross_attention.gate.fill_(0.6)
    model = LambdaAudioModel(encoder_config, decoder_config)
    weights = {"model." + name: mx.array(tensor.numpy()) for name, tensor in reference.state_dict().items()}
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    return reference, model


@pytest.mark.parametrize("models", ["default", "layer_norm"], indirect=True)
def test_mlx_encoder_and_decoder_match_pytorch(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    reference, model = models
    values = torch.randn(2, 81)
    padding = torch.arange(81)[None, :] >= torch.tensor([57, 81])[:, None]
    values.masked_fill_(padding, 0)
    tokens = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
    with torch.inference_mode():
        expected_features, expected_mask = reference.encode(values, padding)
        expected = reference.decoder(
            input_ids=tokens,
            audio_key_values=reference.decoder.project_audio_key_values(expected_features),
            audio_padding_mask=expected_mask,
        ).logits
    features, mask = model.encode(mx.array(values.numpy()), mx.array(padding.numpy()))
    logits = model.decoder(mx.array(tokens.numpy()), model.decoder.project_audio_key_values(features), mask)
    np.testing.assert_array_equal(np.array(mask), expected_mask.numpy())
    np.testing.assert_allclose(np.array(features), expected_features.numpy(), rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(np.array(logits), expected.numpy(), rtol=3e-4, atol=3e-5)


def test_mlx_decoder_cache_matches_full_sequence(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    _, model = models
    features = mx.random.normal((2, 5, 16))
    mask = mx.arange(5)[None, :] >= mx.array([3, 5])[:, None]
    audio_key_values = model.decoder.project_audio_key_values(features)
    tokens = mx.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]])
    expected = model.decoder(tokens, audio_key_values, mask)
    cache: list[LayerCache] = [None] * 3
    prefill = model.decoder(tokens[:, :2], audio_key_values, mask, cache)
    outputs = [prefill]
    for index in range(2, tokens.shape[1]):
        outputs.append(model.decoder(tokens[:, index : index + 1], audio_key_values, mask, cache))
    np.testing.assert_allclose(np.array(mx.concatenate(outputs, axis=1)), np.array(expected), rtol=2e-4, atol=2e-5)


def test_mlx_compiled_decoder_handles_variable_audio_and_batches(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    _, model = models
    for batch, length in [(1, 5), (3, 8), (1, 11)]:
        features = mx.random.normal((batch, length, 16))
        mask = mx.broadcast_to(mx.arange(length) >= length - 1, (batch, length))
        audio = model.decoder.project_audio_key_values(features)
        tokens = mx.random.randint(0, 32, shape=(batch, 5))
        expected = model.decoder(tokens, audio, mask)
        cache: list[LayerCache] = [None] * 3
        outputs = [model.decoder(tokens[:, index : index + 1], audio, mask, cache) for index in range(tokens.shape[1])]
        np.testing.assert_allclose(np.array(mx.concatenate(outputs, axis=1)), np.array(expected), rtol=2e-4, atol=2e-5)


def test_mlx_generate_matches_pytorch_greedy(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    reference, model = models
    values = torch.randn(2, 65)
    padding = torch.arange(65)[None, :] >= torch.tensor([49, 65])[:, None]
    values.masked_fill_(padding, 0)
    with torch.inference_mode():
        expected = reference.generate(
            input_values=values,
            audio_padding_mask=padding,
            input_ids=torch.ones((2, 1), dtype=torch.long),
            attention_mask=torch.ones((2, 1), dtype=torch.long),
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=31,
            pad_token_id=31,
            use_cache=True,
        )
    actual = generate(model, mx.array(values.numpy()), mx.array(padding.numpy()), 1, 31, 6)
    assert actual == expected[:, 1:].tolist()


def test_mlx_generate_stops_each_item_at_eos() -> None:
    class FakeDecoder:
        def __init__(self) -> None:
            self.model = SimpleNamespace(layers=[None])
            self.step = 0

        def project_audio_key_values(self, features: mx.array) -> list[None]:
            return [None]

        def __call__(self, tokens: mx.array, audio_keys: list[None], mask: mx.array, cache: list[LayerCache]) -> mx.array:
            self.step += 1
            predicted = [3, 2] if self.step == 1 else [1, 3]
            return mx.eye(4)[mx.array(predicted)][:, None]

    class FakeModel:
        def __init__(self) -> None:
            self.decoder = FakeDecoder()

        def encode(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
            return values, mask

    model = FakeModel()
    tokens = generate(model, mx.ones((2, 4)), mx.zeros((2, 4), dtype=mx.bool_), 1, 3, 8)  # type: ignore[arg-type]
    assert tokens == [[3, 3], [2, 3]]
    assert model.decoder.step == 2


def test_mlx_encoder_matches_unpadded_item_in_fp16(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    _, model = models
    from mlx.utils import tree_map

    model.update(tree_map(lambda value: value.astype(mx.float16), model.parameters()))
    short = mx.random.normal((1, 57)).astype(mx.float16)
    batch = mx.concatenate((mx.pad(short, [(0, 0), (0, 24)]), mx.random.normal((1, 81)).astype(mx.float16)), axis=0)
    expected, expected_mask = model.encode(short, mx.zeros(short.shape, dtype=mx.bool_))
    actual, actual_mask = model.encode(batch, mx.arange(81)[None, :] >= mx.array([57, 81])[:, None])
    np.testing.assert_array_equal(np.array(actual_mask[0, : expected.shape[1]]), np.array(expected_mask[0]))
    np.testing.assert_allclose(np.array(actual[0, : expected.shape[1]]), np.array(expected[0]), rtol=0.02, atol=0.01)
    assert bool(mx.all(mx.isfinite(actual)).item())


def test_mlx_rejects_incompatible_weights_and_short_audio(models: tuple[TorchLambdaAudioModel, LambdaAudioModel]) -> None:
    _, model = models
    with pytest.raises(ValueError):
        model.load_weights([("adapter.projection.weight", mx.zeros((2, 2)))], strict=True)
    with pytest.raises(ValueError, match="at least"):
        model.encode(mx.ones((1, 3)), mx.zeros((1, 3), dtype=mx.bool_))
    with pytest.raises(ValueError, match="positive"):
        generate(model, mx.ones((1, 65)), mx.zeros((1, 65), dtype=mx.bool_), 1, 31, 0)


def test_mlx_transcriber_uses_shared_benchmark_results(models: tuple[TorchLambdaAudioModel, LambdaAudioModel], tmp_path: Path) -> None:
    _, model = models
    input_path = tmp_path / "input.parquet"
    input_path.touch()
    config = BenchmarkConfig(model="whisper", input_path=input_path, max_new_tokens=2)
    encoded_audio = []
    for size in [65, 81]:
        buffer = BytesIO()
        sf.write(buffer, np.linspace(-0.5, 0.5, size, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
        encoded_audio.append(buffer.getvalue())

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 31

        def batch_decode(self, tokens: list[list[int]], skip_special_tokens: bool) -> list[str]:
            assert skip_special_tokens
            assert len(tokens) == 2 and all(1 <= len(item) <= 2 for item in tokens)
            return [" 一 ", " 二 "]

    results = transcribe(model, Wav2Vec2FeatureExtractor(return_attention_mask=True), Tokenizer(), config, encoded_audio)  # type: ignore[arg-type]
    assert [result.text for result in results] == ["一", "二"]
    assert [result.audio_seconds for result in results] == [65 / 16000, 81 / 16000]
    assert all(result.inference_seconds > 0 for result in results)


def test_mlx_backend_cli_requires_safetensors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.benchmark.cli import parse_arguments

    data = tmp_path / "input.parquet"
    data.touch()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", "lambda-audio-mlx", "--input", str(data), "--checkpoint", str(checkpoint)])
    config = parse_arguments()
    assert config.checkpoint_path == checkpoint
    with pytest.raises(ValidationError, match="requires --checkpoint"):
        BenchmarkConfig(model="lambda-audio-mlx", input_path=data)
    invalid = tmp_path / "model.ckpt"
    invalid.touch()
    with pytest.raises(ValidationError, match=".safetensors"):
        BenchmarkConfig(model="lambda-audio-mlx", checkpoint_path=invalid, input_path=data)


def test_mlx_backend_passes_token_and_returns_shared_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.inference
    from src.benchmark.registry import MODEL_LOADERS

    data = tmp_path / "input.parquet"
    data.touch()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    config = BenchmarkConfig(model="lambda-audio-mlx", input_path=data, checkpoint_path=checkpoint)
    components = SimpleNamespace(model=object(), feature_extractor=object(), tokenizer=object())

    def load(received_config: BenchmarkConfig, path: Path, token: str) -> Any:
        assert received_config is config and path == checkpoint and token == "explicit-token"
        return components

    monkeypatch.setattr(mlx.inference, "load_model_components", load)
    engine = MODEL_LOADERS["lambda-audio-mlx"](config, torch.device("mps"), "explicit-token")
    assert isinstance(engine, partial)
    assert engine.func is transcribe
    assert engine.args == (components.model, components.feature_extractor, components.tokenizer, config)
