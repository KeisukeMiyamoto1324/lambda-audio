import torch

from src.modeling.adapter import AudioAdapter


def test_adapter_output_shape() -> None:
    adapter = AudioAdapter(input_size=8, hidden_size=16)
    hidden_states = torch.randn(2, 5, 8)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )

    output, output_mask = adapter(hidden_states, padding_mask)

    assert output.shape == (2, 3, 16)
    assert output_mask.shape == (2, 3)
