from pydantic import BaseModel, ConfigDict, PositiveInt


class ModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
    )

    encoder_model_id: str = "reazon-research/japanese-zipformer-base-k2"
    decoder_model_id: str = "KeisukeMiyamoto/lfm2.5-230m-cpt"
    sampling_rate: PositiveInt = 16_000
