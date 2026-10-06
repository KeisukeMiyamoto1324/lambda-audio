import copy

import pytest
import torch

from src.modeling.kernels.bias_norm import bias_norm
from src.modeling.kernels.group_norm import masked_group_norm
from src.modeling.kernels.softmax import softmax as fused_softmax
from src.modeling.kernels.swoosh import swoosh
from src.modeling.zipformer.scaling import BiasNormFunction, softmax
from src.modeling.zipformer.utils import Fp32GroupNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels require a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [37, 4099])
def test_group_norm_outputs_and_gradients(dtype: torch.dtype, length: int) -> None:
    torch.manual_seed(18)
    reference = Fp32GroupNorm(4, 4).cuda()
    reference.weight.data.uniform_(0.3, 1.7)
    reference.bias.data.normal_()
    actual = copy.deepcopy(reference)
    actual.use_triton = True
    x = torch.randn(3, 4, length, device="cuda", dtype=dtype, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    mask = torch.arange(length, device="cuda")[None, :] >= torch.tensor([length, length // 2, 1], device="cuda")[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(x, mask)
        output = actual(z, mask)
    grad = torch.randn_like(output)
    expected.backward(grad)
    output.backward(grad)
    torch.testing.assert_close(output, expected, atol=0.032 if dtype == torch.bfloat16 else 0.001, rtol=0.008 if dtype == torch.bfloat16 else 0.001)
    torch.testing.assert_close(z.grad, x.grad, atol=0.08, rtol=0.02)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad, atol=0.02, rtol=0.002)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad, atol=0.02, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("store_output", [False, True])
@pytest.mark.parametrize("channel_dim", [-1, 1])
def test_bias_norm_outputs_and_gradients(dtype: torch.dtype, store_output: bool, channel_dim: int) -> None:
    torch.manual_seed(19)
    x = torch.randn(3, 7, 7, device="cuda", dtype=dtype, requires_grad=True)
    b = (torch.randn(7, device="cuda") * 0.1).requires_grad_()
    s = torch.tensor(0.3, device="cuda", requires_grad=True)
    originals = (x, b, s)
    copies = tuple(v.detach().clone().requires_grad_() for v in originals)
    expected = BiasNormFunction.apply(x, b, s, channel_dim, store_output)
    actual = bias_norm(*copies, channel_dim, store_output)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for original, cloned in zip(originals, copies, strict=True):
        torch.testing.assert_close(cloned.grad, original.grad, atol=0.016 if dtype == torch.bfloat16 else 3e-5, rtol=0.01 if dtype == torch.bfloat16 else 3e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [7, 499, 1499])
def test_softmax_outputs_and_gradients(dtype: torch.dtype, length: int) -> None:
    torch.manual_seed(20)
    x = torch.randn(2, 4, 5, length, device="cuda", dtype=dtype, requires_grad=True)
    x.data[..., -2:] = -1000
    z = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = softmax(x, -1)
        actual = fused_softmax(z, -1)
    grad = torch.randn(2, 4, length, 5, device="cuda", dtype=actual.dtype).transpose(-1, -2)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(z.grad, x.grad, atol=2e-5, rtol=0.008)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("activation,offset,constant", [("SwooshL", 4.0, 0.035), ("SwooshR", 1.0, 0.313261687)])
def test_swoosh_outputs_and_gradients(dtype: torch.dtype, autocast: bool, activation: str, offset: float, constant: float) -> None:
    x = torch.linspace(-20, 20, 4003, device="cuda", dtype=dtype).reshape(1, -1).expand(3, -1).detach().requires_grad_()
    z = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        # ---------------------------------------------------------
        # Use an independent FP64 logaddexp reference for stable softplus.
        # Keep the same input/output rounding as the mixed-precision kernel.
        # ---------------------------------------------------------
        shifted = (x - offset).double()
        expected = torch.logaddexp(torch.zeros_like(shifted), shifted).to(dtype) - 0.08 * x - constant
        actual = swoosh(z, activation)
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=0.001 if dtype == torch.bfloat16 else 3e-6, rtol=0.008 if dtype == torch.bfloat16 else 3e-6)
    torch.testing.assert_close(z.grad, x.grad, atol=0.002 if dtype == torch.bfloat16 else 3e-6, rtol=0.01 if dtype == torch.bfloat16 else 3e-6)


def test_group_norm_without_padding_and_frozen_parameters() -> None:
    x = torch.randn(2, 3, 29, device="cuda", requires_grad=True)
    weight = torch.ones(3, device="cuda")
    bias = torch.zeros(3, device="cuda")
    y = masked_group_norm(x, None, weight, bias, 3, 1e-5)
    z = torch.nn.functional.group_norm(x, 3, weight, bias)
    torch.testing.assert_close(y, z)
    expected = torch.autograd.grad(z.square().sum(), x)[0]
    actual = torch.autograd.grad(y.square().sum(), x)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0.05)


def test_group_norm_large_batch_outputs_and_gradients() -> None:
    if torch.cuda.mem_get_info()[0] < 32 * 1024**3:
        pytest.skip("The int32 overflow regression requires 32 GiB of free CUDA memory")

    # ---------------------------------------------------------
    # Match the first convolution for 53 thirty-second recordings.
    # Repeated samples allow a small independent PyTorch reference.
    # ---------------------------------------------------------
    batch, channels, length = 53, 512, 95_999
    reference = Fp32GroupNorm(channels, channels).cuda()
    reference.weight.data.copy_(torch.linspace(0.5, 1.5, channels, device="cuda"))
    reference.bias.data.copy_(torch.linspace(-0.5, 0.5, channels, device="cuda"))
    actual = copy.deepcopy(reference)
    actual.use_triton = True
    waveform = ((torch.arange(length, device="cuda") % 127).float() / 64 - 1).to(torch.bfloat16)
    x = waveform.expand(1, channels, length).contiguous().requires_grad_()
    z = waveform.expand(batch, channels, length).contiguous().requires_grad_()
    mask = (torch.arange(length, device="cuda") >= length * 3 // 4)[None, :]
    expected = reference(x, mask)
    output = actual(z, mask.expand(batch, -1))
    expected.sum().backward()
    output.sum().backward()

    assert x.grad is not None and z.grad is not None
    for index in (0, 43, batch - 1):
        torch.testing.assert_close(output[index : index + 1], expected, atol=0.016, rtol=0.008)
        torch.testing.assert_close(z.grad[index : index + 1], x.grad, atol=0.016, rtol=0.008)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad * batch, atol=0.1, rtol=0.002)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad * batch, atol=0.1, rtol=0.002)


@pytest.mark.parametrize("frozen", [False, True])
def test_encoder_integration_preserves_checkpoints_and_gradients(frozen: bool) -> None:
    import random

    from src.modeling.zipformer import ZipformerConfig, ZipformerModel
    from src.modeling.zipformer.scaling import ScheduledFloat
    from src.training.kernels import enable_training_kernels

    torch.manual_seed(22)
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
    )
    reference = ZipformerModel(config).cuda().train(not frozen).requires_grad_(not frozen)
    for module in reference.modules():
        if isinstance(module, ScheduledFloat):
            module.batch_count = 5000
    actual = copy.deepcopy(reference)
    enable_training_kernels(actual)
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    x = torch.randn(2, 81, device="cuda")
    mask = torch.arange(81, device="cuda")[None, :] >= torch.tensor([81, 57], device="cuda")[:, None]
    outputs = []
    for model in (reference, actual):
        random.seed(22)
        torch.manual_seed(22)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x, mask).last_hidden_state
        if not frozen:
            y.square().mean().backward()
        outputs.append(y)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.005, rtol=0.015)
    if frozen:
        assert not outputs[1].requires_grad
        return
    for (name, expected), (_, result) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
        assert (expected.grad is None) == (result.grad is None), name
        if expected.grad is not None:
            assert torch.isfinite(result.grad).all(), name
            torch.testing.assert_close(result.grad, expected.grad, atol=0.002, rtol=0.04, msg=name)


@pytest.mark.parametrize("operator", ["swoosh", "softmax", "group_norm"])
def test_variable_lengths_reuse_compiled_kernels(operator: str) -> None:
    import importlib

    module = importlib.import_module(f"src.modeling.kernels.{operator}")
    names = {
        "swoosh": ("_forward", "_backward"),
        "softmax": ("_backward",),
        "group_norm": ("_statistics", "_merge", "_forward", "_backward_statistics", "_backward"),
    }[operator]
    device = torch.cuda.current_device()
    cache_sizes: list[list[int]] = []
    for batch, length in ((2, 2051), (3, 2112), (1, 3001), (2, 4096)):
        if operator == "group_norm":
            length += 2050
        x = torch.randn(batch, 4, length, device="cuda", requires_grad=True)
        if operator == "swoosh":
            y = swoosh(x, "SwooshL")
        elif operator == "softmax":
            y = fused_softmax(x, -1)
        else:
            weight = torch.ones(4, device="cuda", requires_grad=True)
            bias = torch.zeros(4, device="cuda", requires_grad=True)
            mask = torch.arange(length, device="cuda")[None, :].expand(batch, -1) >= length - 7
            y = masked_group_norm(x, mask, weight, bias, 4, 1e-5)
        y.backward(torch.randn_like(y))
        assert torch.isfinite(y).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        cache_sizes.append([len(getattr(module, name).device_caches[device][0]) for name in names])
    assert all(sizes == cache_sizes[0] for sizes in cache_sizes[1:])


@pytest.mark.parametrize("activation,offset,constant", [("SwooshL", 4.0, 0.035), ("SwooshR", 1.0, 0.313261687)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_swoosh_extreme_inputs_have_stable_gradients(activation: str, offset: float, constant: float, dtype: torch.dtype, autocast: bool) -> None:
    x = torch.tensor([-1000, -100, -30, 19.5, 20, 20.5, 80, 88, 89, 100, 1000], device="cuda", dtype=dtype)
    x = (x + offset).requires_grad_()
    grad = torch.tensor([1, -1, 0, 0.125, -0.5, 1, -2, 0, 1e-20, 1, -1], device="cuda", dtype=dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = swoosh(x, activation)
    y.backward(grad)
    shifted = (x.detach() - offset).float()
    expected_y = torch.logaddexp(torch.zeros_like(shifted, dtype=torch.float64), shifted.double()).to(dtype) - (0.08 * x.detach()).to(dtype)
    expected_y = (expected_y - constant).to(dtype)
    expected_grad = ((grad.float() * shifted.sigmoid()).to(dtype) + (-0.08 * grad.float()).to(dtype)).to(dtype)
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    torch.testing.assert_close(y, expected_y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, expected_grad, atol=1e-7, rtol=1e-5)
