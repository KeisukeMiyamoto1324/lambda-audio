import pytest
import torch

from src.modeling.kernels.balancer import balance
from src.modeling.kernels.whitening import whitening_backward
from src.modeling.zipformer.scaling import BalancerFunction, _whitening_metric

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mean,scale", [(0.0, 1.0), (2.0, 0.1), (-2.0, 5.0), (0.0, 0.01), (0.0, 0.0), (3.0, 0.0)])
@pytest.mark.parametrize("dim", [0, -1])
def test_balancer_matches_autograd(dtype: torch.dtype, mean: float, scale: float, dim: int) -> None:
    torch.manual_seed(51)
    x = (torch.randn(4, 3073, device="cuda") * scale + mean).to(dtype)
    if dim == -1:
        x = x.T
    x.requires_grad_()
    z = x.detach().clone().requires_grad_()
    config = (-0.2, 0.2, 0.2, 2.0, 0.03, dim)
    g = torch.randn_like(x)
    BalancerFunction.apply(x, *config).backward(g)
    balance(z, *config).backward(g)
    torch.testing.assert_close(z.grad, x.grad, atol=0.016 if dtype == torch.bfloat16 else 2e-5, rtol=0.008 if dtype == torch.bfloat16 else 2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("selected,limit", [(True, 1.01), (True, 100.0), (False, 1.01)])
def test_whitening_matches_autograd(dtype: torch.dtype, groups: int, selected: bool, limit: float) -> None:
    torch.manual_seed(61)
    x = torch.randn(7, 19, 66, device="cuda", dtype=dtype)
    x[..., 0] *= 4
    grad = torch.randn_like(x)
    reference = x.float().detach().requires_grad_()
    metric = _whitening_metric(reference, groups)
    expected = grad
    probability = 0.5
    if selected:
        probability = 0.1
        if metric >= limit:
            probability = 0.8
            penalty = torch.autograd.grad(metric, reference)[0]
            factor = 0.03 * (grad.float().norm() / (penalty.norm() + 1e-20))
            expected = grad + (penalty * factor).to(dtype)
    state = torch.tensor(0.5, device="cuda")
    actual = whitening_backward(x, grad, state, torch.tensor(selected, device="cuda"), groups, limit, 0.1, 0.8, 0.03)
    torch.testing.assert_close(actual, expected, atol=0.016 if dtype == torch.bfloat16 else 2e-5, rtol=0.008 if dtype == torch.bfloat16 else 2e-5)
    torch.testing.assert_close(state, torch.tensor(probability, device="cuda"))


@pytest.mark.parametrize("scale", [0.0, 1e-8])
def test_whitening_degenerate_inputs(scale: float) -> None:
    x = torch.randn(21, 16, device="cuda") * scale
    grad = torch.randn_like(x)
    probability = torch.tensor(0.5, device="cuda")
    result = whitening_backward(x, grad, probability, torch.tensor(True, device="cuda"), 1, 1.1, 0.1, 0.8, 0.03)
    torch.testing.assert_close(result, grad, rtol=0, atol=0)
    assert torch.isfinite(result).all()


def test_whitening_keeps_probability_and_selection_on_device() -> None:
    from src.modeling.kernels.whitening import whiten
    from src.modeling.zipformer.scaling import Whiten

    module = Whiten(1, 100.0, (0.1, 0.8), 0.03)
    module.register_buffer("kernel_probability", torch.tensor(0.8, device="cuda"), persistent=False)
    x = torch.randn(53, 32, device="cuda", requires_grad=True)
    grad = torch.randn_like(x)
    whiten(x, module, 0.5).backward(grad)
    torch.testing.assert_close(x.grad, grad)
    torch.testing.assert_close(module.kernel_probability, torch.tensor(0.1, device="cuda"))
    module.whitening_limit = 1.0
    x.grad = None
    whiten(x, module, 0.5).backward(grad)
    torch.testing.assert_close(x.grad, grad)
    x.grad = None
    whiten(x, module, 0.05).backward(grad)
    assert not torch.equal(x.grad, grad)
    torch.testing.assert_close(module.kernel_probability, torch.tensor(0.8, device="cuda"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        whiten(x, module, 0.5).backward(grad)
    assert not any(event.key in ("aten::item", "aten::_local_scalar_dense") for event in profile.key_averages())
