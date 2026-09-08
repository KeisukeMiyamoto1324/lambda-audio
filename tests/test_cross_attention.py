import torch

from src.modeling.cross_attention import GatedCrossAttention


def test_zero_gate_preserves_hidden_states() -> None:
    module = GatedCrossAttention(hidden_size=16, num_heads=4, norm_eps=1e-5)
    hidden_states = torch.randn(2, 3, 16)
    audio_features = torch.randn(2, 5, 16)
    audio_padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    audio_key_values = module.project_key_value(audio_features)

    output = module(hidden_states, audio_key_values, audio_padding_mask)

    torch.testing.assert_close(output, hidden_states)


def test_cached_attention_matches_multihead_attention() -> None:
    module = GatedCrossAttention(hidden_size=16, num_heads=4, norm_eps=1e-5)
    module.gate.data.fill_(1.0)
    hidden_states = torch.randn(2, 3, 16)
    audio_features = torch.randn(2, 5, 16)
    audio_padding_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])

    expected_attention, _ = module.attention(
        query=module.norm(hidden_states),
        key=audio_features,
        value=audio_features,
        key_padding_mask=audio_padding_mask,
        need_weights=False,
    )
    audio_key_values = module.project_key_value(audio_features)
    assert audio_key_values[0].untyped_storage().data_ptr() == audio_key_values[1].untyped_storage().data_ptr()
    output = module(hidden_states, audio_key_values, audio_padding_mask)
    expected = hidden_states + torch.tanh(module.gate) * expected_attention

    torch.testing.assert_close(output, expected)
