from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = cast(
        dict[str, Any],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    return cast(dict[str, torch.Tensor], checkpoint["state_dict"])


def load_model_checkpoint(model: nn.Module, path: Path) -> None:
    state_dict = {
        name.removeprefix("model."): parameter
        for name, parameter in load_checkpoint_state(path).items()
    }
    model.load_state_dict(state_dict, strict=True)
