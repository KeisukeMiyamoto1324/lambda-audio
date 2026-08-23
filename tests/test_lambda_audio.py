from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from src.shared.lambda_audio import LambdaAudioModel
from src.shared.lfm2_decoder import AudioLfm2ForCausalLM


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


def test_lambda_audio_forward() -> None:
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

    output = model(
        input_values=torch.randn(2, 8),
        audio_padding_mask=torch.zeros(2, 8, dtype=torch.bool),
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3], [1, 4, 5]]),
    )

    assert output.logits.shape == (2, 3, 32)
    assert output.loss is not None

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
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert generated_ids.shape[0] == 1
    assert any(name.startswith("adapter.") for name in trainable_names)
    assert any("cross_attention" in name for name in trainable_names)
    assert not any(
        name.startswith("decoder.") and "cross_attention" not in name
        for name in trainable_names
    )
