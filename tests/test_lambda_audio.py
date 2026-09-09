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
        self.config = SimpleNamespace(hidden_size=8)
        self.projection = nn.Linear(1, 8)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(24))

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any:
        hidden_states = self.projection(input_features)
        return SimpleNamespace(last_hidden_state=hidden_states)


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
        input_features=torch.randn(2, 8, 1),
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
        input_features=torch.randn(1, 8, 1),
        audio_padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        max_new_tokens=2,
    )
    generation_projection_calls = projection_calls - 1
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    model.cast_frozen_weights(torch.bfloat16)

    assert generated_ids.shape[0] == 1
    assert generation_projection_calls == 1
    assert any(name.startswith("adapter.") for name in trainable_names)
    assert any("cross_attention" in name for name in trainable_names)
    assert not any(name.startswith("decoder.") and "cross_attention" not in name for name in trainable_names)
    assert model.decoder.model.embed_tokens.weight.dtype == torch.bfloat16
    assert cross_attention.attention.out_proj.weight.dtype == torch.float32


def test_ctc_loss_ignores_unalignable_examples(monkeypatch: MonkeyPatch) -> None:
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
    model = LambdaAudioModel(
        DummyEncoder(),
        AudioLfm2ForCausalLM(config),
        ctc_vocab_size=8,
        ctc_blank_token_id=0,
        ctc_loss_weight=0.3,
    )

    output = model(
        input_features=torch.randn(2, 2, 1),
        audio_padding_mask=torch.zeros(2, 2, dtype=torch.bool),
        input_ids=torch.tensor([[1, 2], [1, 3]]),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
        labels=torch.tensor([[1, 2], [1, 3]]),
        ctc_labels=torch.tensor([[1, 1], [2, 3]]),
        ctc_label_lengths=torch.tensor([2, 2]),
    )

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.ctc_output.weight.grad is not None


def test_encoder_trains_only_selected_later_layers() -> None:
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
    model = LambdaAudioModel(
        DummyEncoder(),
        AudioLfm2ForCausalLM(config),
        ctc_vocab_size=8,
        ctc_blank_token_id=0,
        ctc_loss_weight=0.3,
    )

    model.set_encoder_trainable_layers(12)

    assert not any(parameter.requires_grad for parameter in model.encoder.projection.parameters())
    assert not any(parameter.requires_grad for layer in model.encoder.encoder.layers[:12] for parameter in layer.parameters())
    assert all(parameter.requires_grad for layer in model.encoder.encoder.layers[12:] for parameter in layer.parameters())
