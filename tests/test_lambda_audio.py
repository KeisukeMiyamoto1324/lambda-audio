from types import SimpleNamespace
from typing import Any

import torch
from pytest import MonkeyPatch
from torch import nn
from torch.nn import functional as F
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from src import LambdaAudioModel
from src.modeling import lfm2_decoder as lfm2_decoder_module
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


class ReferenceFusedCrossEntropy(nn.Module):
    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(F.linear(hidden_states, weight).float(), labels)


def test_lambda_audio_forward(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        lfm2_decoder_module,
        "_get_fused_cross_entropy",
        lambda: ReferenceFusedCrossEntropy(),
    )
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
    model = LambdaAudioModel(
        DummyEncoder(),
        decoder,
        ctc_vocab_size=8,
        ctc_blank_token_id=0,
        ctc_loss_weight=0.3,
        encoder_hidden_size=8,
    )
    projection_calls = 0
    cross_attention = decoder.model.layers[0].cross_attention
    project_key_value = cross_attention.project_key_value

    def count_projection(audio_features: torch.Tensor) -> torch.Tensor:
        nonlocal projection_calls
        projection_calls += 1
        return project_key_value(audio_features)

    monkeypatch.setattr(cross_attention, "project_key_value", count_projection)

    output = model(
        input_values=torch.randn(2, 8),
        audio_padding_mask=torch.zeros(2, 8, dtype=torch.bool),
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        ctc_labels=torch.tensor([[1, 2], [3, 4]]),
        ctc_label_lengths=torch.tensor([2, 2]),
    )

    assert output.logits.numel() == 0
    assert output.loss is not None
    output.loss.backward()
    assert model.ctc_output.weight.grad is not None
    assert model.adapter.projection.weight.grad is not None
    assert decoder.model.layers[0].cross_attention.gate.grad is not None

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
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    model.cast_frozen_linear_embedding_weights(torch.bfloat16)

    assert generated_ids.shape[0] == 1
    assert generation_projection_calls == 1
    assert any(name.startswith("adapter.") for name in trainable_names)
    assert any("cross_attention" in name for name in trainable_names)
    assert not any(name.startswith("decoder.") and "cross_attention" not in name for name in trainable_names)
    assert model.decoder.model.embed_tokens.weight.dtype == torch.bfloat16
    assert cross_attention.attention.out_proj.weight.dtype == torch.float32
