import torch
from torch import nn

from src.shared.zipformer.modeling_zipformer import ZipformerModel
from src.shared.zipformer.zipformer import CompactRelPositionalEncoding


def test_freeze_feature_extractor_disables_gradients() -> None:
    model = object.__new__(ZipformerModel)
    nn.Module.__init__(model)
    model.feature_extractor = nn.Sequential(nn.Conv1d(1, 2, 3))
    model.feature_grad_mult = 0.1

    model.freeze_feature_extractor()

    assert model.feature_grad_mult == 0.0
    assert not any(
        parameter.requires_grad for parameter in model.feature_extractor.parameters()
    )


def test_positional_encoding_recovers_from_meta_device() -> None:
    encoding = CompactRelPositionalEncoding(embed_dim=8, dropout_rate=0.0)
    encoding.pe = encoding.pe.to(device="meta")

    output = encoding(torch.zeros(4, 1, 8))

    assert output.device.type == "cpu"
    assert output.shape == (1, 7, 8)
