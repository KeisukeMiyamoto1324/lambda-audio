from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx.model import LambdaAudioModel

CHECKPOINT_METADATA = {
    "format": "mlx",
    "lambda_audio_version": "1",
    "decoder_quantization": "affine",
    "decoder_bits": "4",
    "decoder_group_size": "64",
}


def _is_linear(_path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def load_checkpoint(model: LambdaAudioModel, checkpoint_path: Path) -> None:
    weights, metadata = mx.load(str(checkpoint_path), return_metadata=True)
    if metadata.get("format") not in (None, "pt", "mlx"):
        raise ValueError("Unsupported LambdaAudio checkpoint format.")
    quantized = metadata.get("format") == "mlx"
    if quantized and any(metadata.get(key) != value for key, value in CHECKPOINT_METADATA.items()):
        raise ValueError("Unsupported MLX LambdaAudio checkpoint metadata.")

    # ---------------------------------------------------------
    # Native MLX checkpoints already contain packed decoder weights.
    # Original PyTorch checkpoints need layout conversion and quantization.
    # ---------------------------------------------------------
    if not quantized:
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
    nn.quantize(model.decoder, group_size=64, bits=4, class_predicate=_is_linear)
    if quantized:
        model.load_weights(list(weights.items()), strict=True)


def save_checkpoint(model: LambdaAudioModel, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("MLX LambdaAudio requires a .safetensors checkpoint.")
    modules = [module for _, module in model.decoder.named_modules()]
    quantized = [module for module in modules if isinstance(module, nn.QuantizedLinear)]
    if not quantized or any(isinstance(module, nn.Linear) for module in modules):
        raise ValueError("Quantize all decoder linear layers before saving.")
    if any(module.bits != 4 or module.group_size != 64 or module.mode != "affine" for module in quantized):
        raise ValueError("Saving requires affine 4bit decoder weights with group size 64.")

    # ---------------------------------------------------------
    # Preserve native MLX layouts and quantization parameters exactly.
    # Exclusive creation protects the original checkpoint from overwrites.
    # ---------------------------------------------------------
    with checkpoint_path.open("xb") as output:
        mx.save_safetensors(output, dict(tree_flatten(model.parameters())), CHECKPOINT_METADATA)
