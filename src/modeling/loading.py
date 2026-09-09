from typing import cast

import torch
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
    Wav2Vec2BertModel,
    Wav2Vec2CTCTokenizer,
)

from src.modeling.config import ModelConfig
from src.modeling.lambda_audio import LambdaAudioModel
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM


class ModelComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: LambdaAudioModel
    feature_extractor: SequenceFeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    ctc_tokenizer: PreTrainedTokenizerBase


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
    ctc_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        config.ctc_tokenizer_model_id,
        token=token,
    )
    if ctc_tokenizer.pad_token_id is None:
        raise ValueError("The CTC tokenizer must define a blank token.")

    encoder = Wav2Vec2BertModel.from_pretrained(
        config.encoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
        token=token,
    )
    decoder = AudioLfm2ForCausalLM.from_pretrained(
        config.decoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
        token=token,
    )
    model = LambdaAudioModel(
        encoder,
        decoder,
        ctc_vocab_size=len(ctc_tokenizer),
        ctc_blank_token_id=ctc_tokenizer.pad_token_id,
        ctc_loss_weight=config.ctc_loss_weight,
    )
    return ModelComponents(
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        ctc_tokenizer=ctc_tokenizer,
    )
