from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class ModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
    )

    encoder_model_id: str = "reazon-research/japanese-zipformer-base-k2"
    decoder_model_id: str = "LiquidAI/LFM2.5-230M-Base"
    ctc_tokenizer_model_id: str = "reazon-research/japanese-zipformer-base-k2-rs35kh-bpe"
    ctc_loss_weight: PositiveFloat = 0.3
    time_mask_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    time_mask_length: PositiveInt = 10
    sampling_rate: PositiveInt = 16_000
