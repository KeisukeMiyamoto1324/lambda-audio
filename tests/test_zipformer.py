import torch

from src.modeling.zipformer import ZipformerConfig, ZipformerForCTC
from src.modeling.zipformer.zipformer import CompactRelPositionalEncoding


def test_positional_encoding_recovers_from_meta_device() -> None:
    encoding = CompactRelPositionalEncoding(embed_dim=8, dropout_rate=0.0)
    encoding.pe = encoding.pe.to(device="meta")

    output = encoding(torch.zeros(4, 1, 8))

    assert output.device.type == "cpu"
    assert output.shape == (1, 7, 8)


def test_ctc_model_accepts_missing_padding_mask() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2), (8, 3, 2)),
        encoder_dim=(8,),
        downsampling_factor=(1,),
        num_encoder_layers=(1,),
        encoder_unmasked_dim=(8,),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2,),
        feedforward_dim=(16,),
        cnn_module_kernel=(3,),
        vocab_size=12,
    )
    model = ZipformerForCTC(config).eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 64))

    assert output.logits.shape == (1, 15, 12)
    assert torch.isfinite(output.logits).all()
