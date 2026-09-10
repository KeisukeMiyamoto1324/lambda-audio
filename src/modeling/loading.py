from typing import cast

import torch
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)

from src.modeling.config import ModelConfig
from src.modeling.lambda_audio import LambdaAudioModel
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM
from src.modeling.zipformer import ZipformerModel


class ModelComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: LambdaAudioModel
    feature_extractor: SequenceFeatureExtractor
    tokenizer: PreTrainedTokenizerBase


def load_model_components(
    config: ModelConfig,
    token: str,
) -> ModelComponents:
    feature_extractor = cast(
        SequenceFeatureExtractor,
        AutoFeatureExtractor.from_pretrained(config.encoder_model_id, token=token),
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(config.decoder_model_id, token=token),
    )
    encoder = ZipformerModel.from_pretrained(config.encoder_model_id, token=token)
    decoder = AudioLfm2ForCausalLM.from_pretrained(
        config.decoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
        token=token,
    )
    model = LambdaAudioModel(encoder, decoder)
    return ModelComponents(
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
