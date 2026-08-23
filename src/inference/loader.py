from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)

from src.encoders.zipformer import ZipformerModel
from src.inference.config import InferenceConfig
from src.shared.lambda_audio import LambdaAudioModel
from src.shared.lfm2_decoder import AudioLfm2ForCausalLM


def select_device() -> torch.device:
    # ---------------------------------------------------------
    # Select the fastest available device in the requested order.
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path) -> None:
    # ---------------------------------------------------------
    # Remove the Lightning module prefix and load every model weight.
    # ---------------------------------------------------------
    checkpoint = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    state_dict = cast(dict[str, torch.Tensor], checkpoint["state_dict"])
    model_state_dict = {
        name.removeprefix("model."): parameter
        for name, parameter in state_dict.items()
    }
    model.load_state_dict(model_state_dict, strict=True)


def load_components(
    config: InferenceConfig,
    device: torch.device,
) -> tuple[LambdaAudioModel, SequenceFeatureExtractor, PreTrainedTokenizerBase]:
    # ---------------------------------------------------------
    # Build the training architecture before restoring learned weights.
    # ---------------------------------------------------------
    feature_extractor = cast(
        SequenceFeatureExtractor,
        AutoFeatureExtractor.from_pretrained(config.encoder_model_id),
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(config.decoder_model_id),
    )
    encoder = ZipformerModel.from_pretrained(config.encoder_model_id)
    decoder = AudioLfm2ForCausalLM.from_pretrained(
        config.decoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )
    model = LambdaAudioModel(encoder, decoder)
    load_checkpoint_weights(model, config.checkpoint_path)
    model.to(device)
    model.eval()

    return model, feature_extractor, tokenizer
