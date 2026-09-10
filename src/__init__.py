from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.modeling.lambda_audio import LambdaAudioModel

__all__ = ["LambdaAudioModel"]


def __getattr__(name: str) -> object:
    if name != "LambdaAudioModel":
        raise AttributeError(f"module 'src' has no attribute {name!r}")

    from src.modeling.lambda_audio import LambdaAudioModel

    return LambdaAudioModel
