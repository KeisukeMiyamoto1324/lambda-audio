from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt


class ModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
    )

    encoder_model_id: str = "facebook/w2v-bert-2.0"
    decoder_model_id: str = "LiquidAI/LFM2.5-230M-Base"
    ctc_tokenizer_model_id: str = "reazon-research/japanese-zipformer-base-k2-rs35kh-bpe"
    ctc_loss_weight: PositiveFloat = 0.3
    sampling_rate: PositiveInt = 16_000
