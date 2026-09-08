import torch

from src.modeling.zipformer import ZipformerConfig, ZipformerForCTC, ZipformerModel
from src.modeling.zipformer.scaling import SwooshLForward, SwooshRForward
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


def test_swoosh_forward_has_finite_gradients_for_large_inputs() -> None:
    for swoosh in (SwooshLForward, SwooshRForward):
        input_tensor = torch.tensor([100.0], requires_grad=True)

        swoosh(input_tensor).sum().backward()

        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()


def test_zipformer_trains_only_selected_later_stacks() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
        encoder_dim=(8, 8, 8, 8, 8, 8),
        downsampling_factor=(1, 1, 1, 1, 1, 1),
        num_encoder_layers=(1, 1, 1, 1, 1, 1),
        encoder_unmasked_dim=(8, 8, 8, 8, 8, 8),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2, 2, 2, 2, 2, 2),
        feedforward_dim=(16, 16, 16, 16, 16, 16),
        cnn_module_kernel=(3, 3, 3, 3, 3, 3),
    )
    model = ZipformerModel(config)

    model.set_trainable_stacks(3)

    assert not any(parameter.requires_grad for parameter in model.feature_extractor.parameters())
    for index, stack in enumerate(model.encoder.encoders):
        assert all(parameter.requires_grad == (index >= 3) for parameter in stack.parameters())


def test_padding_mask_uses_convolution_output_lengths() -> None:
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
    )
    model = ZipformerModel(config)
    features = torch.zeros(2, 4, 8)
    input_padding_mask = torch.tensor(
        [
            [False] * 19,
            [False] * 11 + [True] * 8,
        ]
    )

    output_padding_mask = model.forward_padding_mask(features, input_padding_mask)

    torch.testing.assert_close(
        output_padding_mask,
        torch.tensor(
            [
                [False, False, False, False],
                [False, False, True, True],
            ]
        ),
    )


def test_time_mask_changes_only_valid_training_frames() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
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
        mask_time_probability=1.0,
        mask_time_length=1,
    )
    model = ZipformerModel(config).train()
    features = torch.randn(2, 4, 8)
    padding_mask = torch.tensor([[False, False, False, False], [False, False, True, True]])

    masked = model.apply_time_mask(features, padding_mask)

    torch.testing.assert_close(masked[0], model.mask_emb.expand(4, -1))
    torch.testing.assert_close(masked[1, :2], model.mask_emb.expand(2, -1))
    torch.testing.assert_close(masked[1, 2:], features[1, 2:])

    model.eval()
    torch.testing.assert_close(model.apply_time_mask(features, padding_mask), features)
