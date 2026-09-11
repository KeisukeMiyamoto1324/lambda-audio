import torch

from src.modeling.adapter import AudioAdapter


def test_adapter_output_shape() -> None:
    adapter = AudioAdapter(input_size=8, hidden_size=16)
    hidden_states = torch.randn(2, 5, 8)
    padding_mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])

    output, output_mask = adapter(hidden_states, padding_mask)

    assert output.shape == (2, 3, 16)
    assert output_mask.shape == (2, 3)


def test_adapter_ignores_padded_features() -> None:
    adapter = AudioAdapter(input_size=8, hidden_size=16).eval()
    padding_mask = torch.tensor([[False, False, False, True, True]])
    hidden_states = torch.randn(1, 5, 8)
    changed_hidden_states = hidden_states.clone()
    changed_hidden_states[:, 3:] = torch.randn(1, 2, 8)

    with torch.inference_mode():
        output, output_mask = adapter(hidden_states, padding_mask)
        changed_output, changed_output_mask = adapter(
            changed_hidden_states,
            padding_mask,
        )

    torch.testing.assert_close(output[:, ~output_mask[0]], changed_output[:, ~changed_output_mask[0]])
