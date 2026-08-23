import torch

from src.encoders.zipformer.zipformer import CompactRelPositionalEncoding


def test_positional_encoding_recovers_from_meta_device() -> None:
    encoding = CompactRelPositionalEncoding(embed_dim=8, dropout_rate=0.0)
    encoding.pe = encoding.pe.to(device="meta")

    output = encoding(torch.zeros(4, 1, 8))

    assert output.device.type == "cpu"
    assert output.shape == (1, 7, 8)
