import torch
from pytest import MonkeyPatch
from torch import nn
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

from src.modeling.lfm2_decoder import (
    AudioLfm2DecoderLayer,
    AudioLfm2ForCausalLM,
    add_cross_attention,
    enable_causal_convolution_kernels,
)


def test_causal_convolution_kernels_are_enabled(monkeypatch: MonkeyPatch) -> None:
    loaded_names: list[str] = []

    class FakeKernel(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    class FakeRepository:
        def __init__(
            self,
            repo_id: str,
            *,
            layer_name: str,
            version: int,
        ) -> None:
            assert repo_id == "kernels-community/mamba-ssm"
            assert version == 3
            loaded_names.append(layer_name)

        def load(self) -> type[FakeKernel]:
            return FakeKernel

    monkeypatch.setattr(
        "src.modeling.lfm2_decoder.LayerRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        modeling_lfm2,
        "causal_conv1d_fn",
        modeling_lfm2.causal_conv1d_fn,
    )
    monkeypatch.setattr(
        modeling_lfm2,
        "causal_conv1d_update",
        modeling_lfm2.causal_conv1d_update,
    )

    enable_causal_convolution_kernels()

    assert loaded_names == ["causal_conv1d_fn", "causal_conv1d_update"]
    assert isinstance(modeling_lfm2.causal_conv1d_fn, FakeKernel)
    assert isinstance(modeling_lfm2.causal_conv1d_update, FakeKernel)


def test_only_full_attention_layers_are_replaced() -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
    )
    model = Lfm2ForCausalLM(config)

    add_cross_attention(model)

    assert not isinstance(model.model.layers[0], AudioLfm2DecoderLayer)
    assert isinstance(model.model.layers[1], AudioLfm2DecoderLayer)
    assert not isinstance(model.model.layers[2], AudioLfm2DecoderLayer)


def test_decoder_returns_logits_without_labels() -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["full_attention"],
    )
    model = AudioLfm2ForCausalLM(config)
    add_cross_attention(model)
    audio_features = torch.randn(2, 4, config.hidden_size)

    output = model(
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        audio_key_values=model.project_audio_key_values(audio_features),
        audio_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
    )

    assert output.loss is None
    assert output.logits is not None
    assert output.logits.shape == (2, 3, config.vocab_size)
