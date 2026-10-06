import torch
from torch import nn

from src.modeling.zipformer.attention import RelPositionMultiheadAttentionWeights
from src.modeling.zipformer.scaling import ActivationDropoutAndLinear, Balancer, BiasNorm, Whiten
from src.modeling.zipformer.utils import Fp32GroupNorm


def enable_training_kernels(encoder: nn.Module) -> None:
    for module in encoder.modules():
        if isinstance(module, Fp32GroupNorm):
            if module.num_groups != module.num_channels or not module.affine:
                raise ValueError("Training kernels require affine GroupNorm with one group per channel.")
            module.use_triton = True
        if isinstance(module, (BiasNorm, ActivationDropoutAndLinear, RelPositionMultiheadAttentionWeights, Balancer, Whiten)):
            module.use_triton = True
        if isinstance(module, Whiten) and "kernel_probability" not in module._buffers:
            module.register_buffer("kernel_probability", torch.tensor(module.max_prob, device=next(encoder.parameters()).device), persistent=False)
