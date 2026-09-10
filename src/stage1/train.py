import torch

from src.environment import load_hf_token
from src.modeling.loading import load_model_components
from src.stage1.cli import parse_arguments
from src.stage1.training import Stage1LightningModule
from src.training.data_module import LambdaAudioDataModule
from src.training.trainer import configure_training_environment, create_trainer


def main() -> None:
    config = parse_arguments()
    configure_training_environment(config.seed)
    token = load_hf_token()
    components = load_model_components(config, token)
    model = components.model
    model.set_backbones_trainable(False)
    model.cast_frozen_weights(torch.bfloat16)
    model.decoder.config.use_cache = False
    model.compile(dynamic=True)

    lightning_model = Stage1LightningModule(model, config)
    data_module = LambdaAudioDataModule(
        components.feature_extractor,
        components.tokenizer,
        components.ctc_tokenizer,
        config,
        token,
    )
    create_trainer(config).fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=config.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
