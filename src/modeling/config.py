from pydantic import BaseModel, ConfigDict, PositiveInt


class ModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
    )

    encoder_model_id: str = "KeisukeMiyamoto/zipformer-cpt"
    decoder_model_id: str = "KeisukeMiyamoto/lfm-cpt-multilingual"
    sampling_rate: PositiveInt = 16_000
