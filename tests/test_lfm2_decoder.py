import pytest
import torch
from pytest import MonkeyPatch
from torch import nn
from torch.nn import functional as F
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

from src.modeling.lfm2_decoder import (
    AudioLfm2DecoderLayer,
    AudioLfm2ForCausalLM,
    _get_fused_cross_entropy,
    add_cross_attention,
)


class ReferenceFusedCrossEntropy(nn.Module):
    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(F.linear(hidden_states, weight).float(), labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Liger Kernel")
def test_fused_cross_entropy_matches_torch() -> None:
    fused_hidden_states = torch.randn(12, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fused_weight = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 32, (12,), device="cuda")
    reference_hidden_states = fused_hidden_states.detach().clone().requires_grad_(True)
    reference_weight = fused_weight.detach().clone().requires_grad_(True)

    fused_loss = _get_fused_cross_entropy()(fused_weight, fused_hidden_states, labels)
    reference_loss = F.cross_entropy(F.linear(reference_hidden_states, reference_weight).float(), labels)
    fused_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(fused_loss, reference_loss, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fused_hidden_states.grad, reference_hidden_states.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_weight.grad, reference_weight.grad, rtol=2e-2, atol=2e-2)


def test_supervised_forward_supports_tuple_output(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.modeling.lfm2_decoder._get_fused_cross_entropy",
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
    )
    model = AudioLfm2ForCausalLM(config)
    add_cross_attention(model)
    audio_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
    audio_key_values = model.project_audio_key_values(torch.randn(2, 3, 16))

    output = model(
        input_ids=torch.tensor([[1, 2, 3], [1, 3, 4]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3], [1, 3, 4]]),
        use_cache=False,
        return_dict=False,
        audio_key_values=audio_key_values,
        audio_padding_mask=audio_padding_mask,
    )

    assert isinstance(output, tuple)
    assert output[0].ndim == 0
    assert output[1].numel() == 0


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


@pytest.mark.parametrize("num_beams", [1, 3])
def test_audio_cache_expansion_and_batched_generation(num_beams: int) -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AudioLfm2ForCausalLM(config).eval()
    add_cross_attention(model)
    input_ids = torch.tensor([[1, 2], [1, 3]])
    attention_mask = torch.ones_like(input_ids)
    audio_padding_mask = torch.tensor([[False, False, False], [False, False, True]])

    with torch.no_grad():
        audio_key_values = model.project_audio_key_values(torch.randn(2, 3, 16))
        expanded_ids, expanded_kwargs = model._expand_inputs_for_generation(
            expand_size=num_beams,
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_key_values=audio_key_values,
            audio_padding_mask=audio_padding_mask,
        )
        indices = torch.tensor([0] * num_beams + [1] * num_beams)
        torch.testing.assert_close(expanded_ids, input_ids[indices])
        torch.testing.assert_close(expanded_kwargs["attention_mask"], attention_mask[indices])
        torch.testing.assert_close(expanded_kwargs["audio_padding_mask"], audio_padding_mask[indices])
        assert isinstance(expanded_kwargs["audio_key_values"], tuple)
        for original, expanded in zip(audio_key_values, expanded_kwargs["audio_key_values"], strict=True):
            torch.testing.assert_close(expanded[0], original[0][indices])
            torch.testing.assert_close(expanded[1], original[1][indices])
        if num_beams == 1:
            assert expanded_kwargs["audio_key_values"] is audio_key_values

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_key_values=audio_key_values,
            audio_padding_mask=audio_padding_mask,
            num_beams=num_beams,
            max_new_tokens=3,
            do_sample=False,
        )

    assert generated_ids.shape == (2, 5)
    torch.testing.assert_close(generated_ids[:, :2], input_ids)
