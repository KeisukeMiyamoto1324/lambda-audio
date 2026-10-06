from torch import nn

from src.modeling.zipformer.attention import RelPositionMultiheadAttentionWeights
from src.modeling.zipformer.scaling import ActivationDropoutAndLinear, BiasNorm
from src.modeling.zipformer.utils import Fp32GroupNorm


def enable_training_kernels(encoder: nn.Module) -> None:
    for module in encoder.modules():
        if isinstance(module, Fp32GroupNorm):
            if module.num_groups != module.num_channels or not module.affine:
                raise ValueError("Training kernels require affine GroupNorm with one group per channel.")
            module.use_triton = True
        if isinstance(module, (BiasNorm, ActivationDropoutAndLinear, RelPositionMultiheadAttentionWeights)):
            module.use_triton = True
