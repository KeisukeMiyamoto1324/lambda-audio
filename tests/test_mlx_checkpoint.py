from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mlx.config import Lfm2Config, ZipformerConfig  # noqa: E402
from mlx.inference.checkpoint import CHECKPOINT_METADATA, load_checkpoint, save_checkpoint  # noqa: E402
from mlx.inference.transcription import generate  # noqa: E402
from mlx.model import LambdaAudioModel  # noqa: E402


@pytest.fixture
def checkpoint_models() -> tuple[LambdaAudioModel, LambdaAudioModel]:
    mx.random.seed(19)
    encoder = ZipformerConfig(
        conv_feature_layers=((8, 3, 2), (8, 3, 2)),
        encoder_dim=(8,),
        downsampling_factor=(1,),
        num_encoder_layers=(1,),
        encoder_unmasked_dim=(8,),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2,),
        feedforward_dim=(16,),
        cnn_module_kernel=(3,),
    )
    decoder = Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
    )
    models = LambdaAudioModel(encoder, decoder), LambdaAudioModel(encoder, decoder)
    for model in models:
        model.eval()
    return models


def test_mlx_checkpoint_round_trip_preserves_weights_and_generation(checkpoint_models: tuple[LambdaAudioModel, LambdaAudioModel], tmp_path: Path) -> None:
    original, restored = checkpoint_models
    original.decoder.model.layers[1].cross_attention.gate = mx.array(0.6)
    weights = {
        "model." + name: (value.transpose(0, 2, 1) if value.ndim == 3 and name.endswith(".weight") else value).astype(mx.float16)
        for name, value in tree_flatten(original.parameters())
    }
    source = tmp_path / "source.safetensors"
    output = tmp_path / "quantized.safetensors"
    mx.save_safetensors(str(source), weights, {"format": "pt"})
    load_checkpoint(original, source)
    save_checkpoint(original, output)
    load_checkpoint(restored, output)

    expected_weights = dict(tree_flatten(original.parameters()))
    actual_weights = dict(tree_flatten(restored.parameters()))
    assert expected_weights.keys() == actual_weights.keys()
    assert any(value.dtype == mx.uint32 for value in actual_weights.values())
    for name, expected in expected_weights.items():
        np.testing.assert_array_equal(np.array(actual_weights[name]), np.array(expected))
    for batch, length in [(1, 65), (2, 81)]:
        audio = mx.random.normal((batch, length)).astype(mx.float16)
        padding = mx.broadcast_to(mx.arange(length) >= length - 8, audio.shape)
        audio = mx.where(padding, 0, audio)
        assert generate(restored, audio, padding, 1, 63, 6) == generate(original, audio, padding, 1, 63, 6)
    with pytest.raises(FileExistsError):
        save_checkpoint(original, output)


@pytest.mark.parametrize("metadata", [{"format": "unknown"}, {**CHECKPOINT_METADATA, "decoder_bits": "8"}, {"format": "mlx"}])
def test_mlx_checkpoint_rejects_unsupported_formats(checkpoint_models: tuple[LambdaAudioModel, LambdaAudioModel], tmp_path: Path, metadata: dict[str, str]) -> None:
    path = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(path), {"weight": mx.zeros((64, 64))}, metadata)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(checkpoint_models[0], path)


def test_mlx_checkpoint_rejects_unquantized_export(checkpoint_models: tuple[LambdaAudioModel, LambdaAudioModel], tmp_path: Path) -> None:
    output = tmp_path / "invalid.safetensors"
    with pytest.raises(ValueError, match="Quantize"):
        save_checkpoint(checkpoint_models[0], output)
    assert not output.exists()
