import torch

from str.cross_attention import GatedCrossAttention


def test_zero_gate_preserves_hidden_states() -> None:
    module = GatedCrossAttention(hidden_size=16, num_heads=4, norm_eps=1e-5)
    hidden_states = torch.randn(2, 3, 16)
    audio_features = torch.randn(2, 5, 16)
    audio_padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    output = module(hidden_states, audio_features, audio_padding_mask)

    torch.testing.assert_close(output, hidden_states)
