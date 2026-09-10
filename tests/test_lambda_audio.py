from types import SimpleNamespace
from typing import Any

import torch
from pytest import MonkeyPatch
from torch import nn
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from src import LambdaAudioModel
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 8)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> Any:
        hidden_states = self.projection(input_values.unsqueeze(-1))
        return SimpleNamespace(last_hidden_state=hidden_states)

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        return padding_mask


def test_lambda_audio_forward(monkeypatch: MonkeyPatch) -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["full_attention"],
        pad_token_id=0,
    )
    decoder = AudioLfm2ForCausalLM(config)
    model = LambdaAudioModel(DummyEncoder(), decoder, encoder_hidden_size=8)
    projection_calls = 0
    compiled_forward_calls = 0
    cross_attention = decoder.model.layers[0].cross_attention
    project_key_value = cross_attention.project_key_value

    def compile_decoder_forward(
        function: Any,
        *,
        dynamic: bool,
    ) -> Any:
        assert dynamic is True

        def compiled_forward(**kwargs: Any) -> Any:
            nonlocal compiled_forward_calls
            compiled_forward_calls += 1
            return function(**kwargs)

        return compiled_forward

    def count_projection(audio_features: torch.Tensor) -> torch.Tensor:
        nonlocal projection_calls
        projection_calls += 1
        return project_key_value(audio_features)

    monkeypatch.setattr(cross_attention, "project_key_value", count_projection)
    monkeypatch.setattr(torch, "compile", compile_decoder_forward)
    model.compile_decoder()

    output = model(
        input_values=torch.randn(2, 8),
        audio_padding_mask=torch.zeros(2, 8, dtype=torch.bool),
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3], [1, 4, 5]]),
    )

    assert output.logits.shape == (2, 3, 32)
    assert output.loss is not None
    assert compiled_forward_calls == 1

    # ---------------------------------------------------------
    # Check generation and the first training stage settings.
    # ---------------------------------------------------------
    model.set_backbones_trainable(False)
    generated_ids = model.generate(
        input_values=torch.randn(1, 8),
        audio_padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        max_new_tokens=2,
    )
    generation_projection_calls = projection_calls - 1
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert generated_ids.shape[0] == 1
    assert compiled_forward_calls == 1
    assert generation_projection_calls == 1
    assert any(name.startswith("adapter.") for name in trainable_names)
    assert any("cross_attention" in name for name in trainable_names)
    assert not any(
        name.startswith("decoder.") and "cross_attention" not in name
        for name in trainable_names
    )
