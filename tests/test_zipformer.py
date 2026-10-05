import pytest
import torch

from src.modeling.zipformer import ZipformerConfig, ZipformerForCTC, ZipformerModel
from src.modeling.zipformer.scaling import softmax
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
    features = torch.empty(2, 15, 8)
    padding_mask = torch.arange(64).unsqueeze(0) >= torch.tensor([[64], [40]])

    output_mask = model.forward_padding_mask(features, padding_mask)

    assert (~output_mask).sum(dim=-1).tolist() == [15, 9]


def test_zipformer_output_is_independent_of_padded_audio() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2), (8, 3, 2)),
        extractor_mode="layer_norm",
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
    model = ZipformerModel(config).eval()
    padding_mask = torch.arange(64).unsqueeze(0) >= 40
    input_values = torch.randn(1, 64)
    changed_input_values = input_values.clone()
    changed_input_values[:, 40:] = torch.randn(1, 24)

    with torch.inference_mode():
        output = model(input_values, padding_mask).last_hidden_state
        changed_output = model(changed_input_values, padding_mask).last_hidden_state

    torch.testing.assert_close(output[:, :9], changed_output[:, :9])


def test_zipformer_output_matches_unpadded_batch_item() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
        extractor_mode="layer_norm",
        encoder_dim=(8, 8),
        downsampling_factor=(1, 2),
        num_encoder_layers=(1, 1),
        encoder_unmasked_dim=(8, 8),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2, 2),
        feedforward_dim=(16, 16),
        cnn_module_kernel=(3, 3),
    )
    model = ZipformerModel(config).eval()
    input_values = torch.randn(1, 31)
    batch_input_values = torch.cat(
        (
            torch.nn.functional.pad(input_values, (0, 10)),
            torch.randn(1, 41),
        )
    )
    padding_mask = torch.arange(41).unsqueeze(0) >= torch.tensor([[31], [41]])

    with torch.inference_mode():
        expected = model(
            input_values,
            torch.zeros_like(input_values, dtype=torch.bool),
        ).last_hidden_state
        actual = model(batch_input_values, padding_mask).last_hidden_state[:1]

    torch.testing.assert_close(actual[:, : expected.shape[1]], expected)


def test_group_norm_ignores_batch_padding() -> None:
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
        extractor_mode="default",
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
    model = ZipformerModel(config).eval()
    input_values = torch.randn(1, 33)
    padded_input_values = torch.nn.functional.pad(input_values, (0, 15))
    padding_mask = torch.arange(48).unsqueeze(0) >= 33

    with torch.inference_mode():
        expected = model.feature_extractor(input_values)
        actual = model.feature_extractor(padded_input_values, padding_mask)

    torch.testing.assert_close(actual[:, :, : expected.shape[-1]], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softmax_backward_is_finite_and_matches_float32(dtype: torch.dtype) -> None:
    values = torch.tensor([[1000.0, 1000.0, -1000.0], [0.0, 0.5, -0.5]], dtype=dtype, requires_grad=True)
    gradient = torch.tensor([[10000.0, -10000.0, 0.0], [10000.0, 5000.0, -10000.0]], dtype=dtype)
    reference = values.detach().float().requires_grad_()
    output = softmax(values, dim=-1)
    output.backward(gradient)
    reference.softmax(dim=-1).backward(gradient.float())
    assert torch.isfinite(output).all()
    assert values.grad is not None
    assert reference.grad is not None
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad.float(), reference.grad, rtol=0.015, atol=2.0)


def test_trainable_zipformer_backward_is_finite() -> None:
    torch.manual_seed(42)
    config = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
        extractor_mode="layer_norm",
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
    model = ZipformerModel(config).train()
    output = model(torch.randn(2, 64), torch.zeros(2, 64, dtype=torch.bool)).last_hidden_state
    loss = output.square().mean()
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert torch.isfinite(loss)
    assert any(name.startswith("feature_extractor.") for name in gradients)
    assert any(name.startswith("encoder.") for name in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
